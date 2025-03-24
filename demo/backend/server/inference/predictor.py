# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import torch
from app_conf import APP_ROOT, MODEL_SIZE, DATA_PATH
from inference.data_types import (
    AddMaskRequest,
    AddPointsRequest,
    CancelPorpagateResponse,
    CancelPropagateInVideoRequest,
    ClearPointsInFrameRequest,
    ClearPointsInVideoRequest,
    ClearPointsInVideoResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    DownloadMasksRequest,
    DownloadMasksResponse,
    Mask,
    PropagateDataResponse,
    PropagateDataValue,
    PropagateInVideoRequest,
    RemoveObjectRequest,
    RemoveObjectResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from PIL import Image
from io import BytesIO
from pycocotools.mask import decode as decode_masks, encode as encode_masks
from sam2.build_sam import build_sam2_video_predictor

import cv2  # <--- We import OpenCV to read frames in original shape

logger = logging.getLogger(__name__)

class InferenceAPI:
    """
    Core class responsible for managing inference sessions, storing their state,
    and performing image/video segmentation. Also supports streaming frames to
    clients via /download_frames.
    """

    def __init__(self) -> None:
        super(InferenceAPI, self).__init__()

        self.session_states: Dict[str, Any] = {}
        self.score_thresh = 0

        # Choose checkpoint & config based on MODEL_SIZE
        if MODEL_SIZE == "tiny":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif MODEL_SIZE == "small":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif MODEL_SIZE == "large":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:  # base_plus (default)
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        # Select device
        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        if force_cpu_device:
            logger.info("Forcing CPU device for SAM 2 demo")
        if torch.cuda.is_available() and not force_cpu_device:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not force_cpu_device:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Turn on TF32 if on a supported GPU
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            logging.warning(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

        self.device = device
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=device
        )
        self.inference_lock = Lock()

    def autocast_context(self):
        """
        Returns the appropriate autocast (automatic mixed precision) context
        if running on CUDA, otherwise a no-op context.
        """
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def _read_raw_frames(self, path: str) -> List[np.ndarray]:
        """
        Utility to read original frames from a video (using OpenCV),
        preserving the original resolution and color arrangement (BGR).
        """
        video_path = str(Path(DATA_PATH) / path)  # The same path that start_session uses
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video at {video_path}")
            return []

        frames = []
        frame_index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            # 'frame_bgr' is 720×1280×3 if your final video is 1280×720
            frames.append(frame_bgr)
            frame_index += 1
        cap.release()
        logger.info(f"Loaded {len(frames)} raw frames from {video_path}")
        return frames

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        """
        Starts an inference session for the given video path, returning a new session_id.
        Also stores the original frames (in 1280×720) so the user can download them
        later without the model's resizing or normalization.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())

            # In case of MPS device, we offload frames to CPU to avoid memory fragmentation
            offload_video_to_cpu = self.device.type == "mps"
            inference_state = self.predictor.init_state(
                request.path,
                offload_video_to_cpu=offload_video_to_cpu,
            )

            # Additionally, read the raw frames from the same video path.
            raw_frames = self._read_raw_frames(request.path)
            # Store them in the same inference_state for later retrieval:
            inference_state["images_original"] = raw_frames

            self.session_states[session_id] = {
                "canceled": False,
                "state": inference_state,
            }
            return StartSessionResponse(session_id=session_id)

    def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        """
        Closes an inference session (frees memory) for the given session_id.
        """
        is_successful = self.__clear_session_state(request.session_id)
        return CloseSessionResponse(success=is_successful)

    def add_points(self, request: AddPointsRequest, test: str = "") -> PropagateDataResponse:
        """
        Add new point prompts to a specific frame. The model immediately updates
        that frame's segmentation mask for the targeted object.
        """
        with self.autocast_context(), self.inference_lock:
            session = self.__get_session(request.session_id)
            inference_state = session["state"]

            frame_idx = request.frame_index
            obj_id = request.object_id
            points = request.points
            labels = request.labels
            clear_old_points = request.clear_old_points

            # Add the new prompt
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=clear_old_points,
                normalize_coords=False,
            )

            masks_binary = (masks > self.score_thresh)[:, 0].cpu().numpy()
            rle_mask_list = self.__get_rle_mask_list(
                object_ids=object_ids, masks=masks_binary
            )

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def add_mask(self, request: AddMaskRequest) -> PropagateDataResponse:
        """
        Add a mask prompt directly (mask is an array of size [H, W], 1 for FG, 0 for BG).
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            obj_id = request.object_id
            rle_mask = {
                "counts": request.mask.counts,
                "size": request.mask.size,
            }

            mask = decode_masks(rle_mask)

            logger.info(
                f"add mask on frame {frame_idx} in session {session_id}: {obj_id=}, {mask.shape=}"
            )
            session = self.__get_session(session_id)
            inference_state = session["state"]

            frame_idx, obj_ids, video_res_masks = self.predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=torch.tensor(mask > 0),
            )
            masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(
                object_ids=obj_ids, masks=masks_binary
            )

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def clear_points_in_frame(
        self, request: ClearPointsInFrameRequest
    ) -> PropagateDataResponse:
        """
        Remove all point prompts in a specific frame for a specified object.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            obj_id = request.object_id

            logger.info(
                f"clear inputs on frame {frame_idx} in session {session_id}: {obj_id=}"
            )
            session = self.__get_session(session_id)
            inference_state = session["state"]
            frame_idx, obj_ids, video_res_masks = (
                self.predictor.clear_all_prompts_in_frame(
                    inference_state, frame_idx, obj_id
                )
            )
            masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(
                object_ids=obj_ids, masks=masks_binary
            )

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def clear_points_in_video(
        self, request: ClearPointsInVideoRequest
    ) -> ClearPointsInVideoResponse:
        """
        Remove all point prompts across all frames of the video.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            logger.info(f"clear all inputs across the video in session {session_id}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            self.predictor.reset_state(inference_state)
            return ClearPointsInVideoResponse(success=True)

    def remove_object(self, request: RemoveObjectRequest) -> RemoveObjectResponse:
        """
        Remove a specific object ID from the tracking state.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            obj_id = request.object_id
            logger.info(f"remove object in session {session_id}: {obj_id=}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            new_obj_ids, updated_frames = self.predictor.remove_object(
                inference_state, obj_id
            )

            results = []
            for frame_index, video_res_masks in updated_frames:
                masks = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                rle_mask_list = self.__get_rle_mask_list(
                    object_ids=new_obj_ids, masks=masks
                )
                results.append(
                    PropagateDataResponse(
                        frame_index=frame_index,
                        results=rle_mask_list,
                    )
                )

            return RemoveObjectResponse(results=results)

    def propagate_in_video(
        self, request: PropagateInVideoRequest
    ) -> Generator[PropagateDataResponse, None, None]:
        """
        Propagate existing point prompts throughout the video to track objects in all frames.
        Yields partial results as they are computed (streaming).
        """
        session_id = request.session_id
        start_frame_idx = request.start_frame_index
        propagation_direction = "both"
        max_frame_num_to_track = None

        with self.autocast_context(), self.inference_lock:
            logger.info(
                f"propagate in video in session {session_id}: "
                f"{propagation_direction=}, {start_frame_idx=}, {max_frame_num_to_track=}"
            )

            try:
                session = self.__get_session(session_id)
                session["canceled"] = False
                inference_state = session["state"]

                # Initialize masks cache if needed
                if "masks_cache" not in inference_state:
                    inference_state["masks_cache"] = {}

                # Forward pass
                if propagation_direction in ["both", "forward"]:
                    for outputs in self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frame_num_to_track,
                        reverse=False,
                    ):
                        if session["canceled"]:
                            return None

                        frame_idx, obj_ids, video_res_masks = outputs
                        masks_binary = (
                            (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                        )

                        rle_mask_list = self.__get_rle_mask_list(
                            object_ids=obj_ids, masks=masks_binary
                        )
                        response = PropagateDataResponse(
                            frame_index=frame_idx,
                            results=rle_mask_list,
                        )
                        inference_state["masks_cache"][frame_idx] = response
                        yield response

                # Backward pass
                if propagation_direction in ["both", "backward"]:
                    for outputs in self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frame_num_to_track,
                        reverse=True,
                    ):
                        if session["canceled"]:
                            return None

                        frame_idx, obj_ids, video_res_masks = outputs
                        masks_binary = (
                            (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                        )

                        rle_mask_list = self.__get_rle_mask_list(
                            object_ids=obj_ids, masks=masks_binary
                        )
                        response = PropagateDataResponse(
                            frame_index=frame_idx,
                            results=rle_mask_list,
                        )
                        inference_state["masks_cache"][frame_idx] = response
                        yield response
            finally:
                logger.info(
                    f"propagation ended in session {session_id}; {self.__get_session_stats()}"
                )

    def download_masks(self, request: DownloadMasksRequest) -> DownloadMasksResponse:
        """
        Retrieve all masks for all frames in the specified session. This uses
        a cache to avoid recomputing masks if they've already been generated.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            logger.info(f"Downloading all masks for session {session_id}")

            session = self.__get_session(session_id)
            inference_state = session["state"]

            if "masks_cache" not in inference_state:
                inference_state["masks_cache"] = {}

            num_frames = inference_state.get("num_frames", 0)
            if num_frames <= 0:
                logger.warning(f"No frames available in session {session_id}")
                return DownloadMasksResponse(results=[])

            results = []
            for frame_idx in range(num_frames):
                if frame_idx in inference_state["masks_cache"]:
                    results.append(inference_state["masks_cache"][frame_idx])
                else:
                    # Compute mask for the frame
                    outputs = self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=frame_idx,
                        max_frame_num_to_track=1,
                        reverse=False,
                    )
                    mask_response = None
                    for frame_idx_result, obj_ids, video_res_masks in outputs:
                        masks_binary = (
                            (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                        )
                        rle_mask_list = self.__get_rle_mask_list(
                            object_ids=obj_ids, masks=masks_binary
                        )
                        mask_response = PropagateDataResponse(
                            frame_index=frame_idx_result,
                            results=rle_mask_list,
                        )
                        break
                    if mask_response is None:
                        mask_response = PropagateDataResponse(
                            frame_index=frame_idx, results=[]
                        )
                    inference_state["masks_cache"][frame_idx] = mask_response
                    results.append(mask_response)

            logger.info(
                f"Completed downloading masks for {len(results)} frames in session {session_id}"
            )
            return DownloadMasksResponse(results=results)

    def download_frames(self, session_id: str) -> Generator[Tuple[int, bytes], None, None]:
        """
        Yield video frames as JPEG-encoded bytes, along with their frame indices, for a given session.

        Now we serve from "images_original" (the 1280×720 frames read with OpenCV),
        rather than the 1024×1024 model data. This ensures correct resolution and color.

        1. We retrieve the frames from "images_original" (shape is [H, W, 3] BGR).
        2. Convert from BGR to RGB, then encode to JPEG.
        3. Yield (frame_index, jpeg_bytes).

        Args:
            session_id: The ID of the session whose frames we want to stream.

        Yields:
            (frame_idx, jpeg_bytes) for each frame in the original resolution & color.

        Raises:
            RuntimeError: If the session or frames do not exist.
        """
        logger.info(f"Starting frame download for session {session_id}")
        session = self.__get_session(session_id)
        inference_state = session["state"]

        # We changed this: serve from "images_original"
        if "images_original" not in inference_state:
            logger.error(f"No original frames found for session {session_id}")
            raise RuntimeError(f"Session {session_id} has no 'images_original' data")

        video_frames = inference_state["images_original"]
        num_frames = len(video_frames)
        logger.info(
            f"Found {num_frames} raw frames in session {session_id}"
        )

        if num_frames == 0:
            logger.warning(f"Session {session_id} contains zero frames (original)")
            return

        for frame_idx, frame_bgr in enumerate(video_frames):
            try:
                # frame_bgr is a NumPy array in [H, W, 3] with BGR channels
                logger.debug(
                    f"Frame {frame_idx} BGR shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}"
                )

                # Convert BGR→RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Turn into a PIL image
                frame_pil = Image.fromarray(frame_rgb)
                buf = BytesIO()
                frame_pil.save(buf, format="JPEG", quality=85)
                jpeg_bytes = buf.getvalue()
                buf.close()

                logger.debug(
                    f"Encoded frame {frame_idx} as JPEG, size: {len(jpeg_bytes)} bytes"
                )
                yield frame_idx, jpeg_bytes

            except Exception as e:
                logger.error(
                    f"Failed to process frame {frame_idx} in session {session_id}: {str(e)}"
                )
                raise RuntimeError(f"Error processing frame {frame_idx}: {str(e)}")

        logger.info(
            f"Completed frame download for session {session_id}, total frames: {num_frames}"
        )

    def cancel_propagate_in_video(
        self, request: CancelPropagateInVideoRequest
    ) -> CancelPorpagateResponse:
        """
        Cancels an ongoing forward/backward propagation for the specified session.
        """
        session = self.__get_session(request.session_id)
        session["canceled"] = True
        return CancelPorpagateResponse(success=True)

    def __get_rle_mask_list(
        self, object_ids: List[int], masks: np.ndarray
    ) -> List[PropagateDataValue]:
        """
        Return a list of data values (object ID plus mask) for each object.
        """
        return [
            self.__get_mask_for_object(object_id=object_id, mask=mask)
            for object_id, mask in zip(object_ids, masks)
        ]

    def __get_mask_for_object(
        self, object_id: int, mask: np.ndarray
    ) -> PropagateDataValue:
        """
        Create a data value for an object/mask pair.
        """
        mask_rle = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
        mask_rle["counts"] = mask_rle["counts"].decode()
        return PropagateDataValue(
            object_id=object_id,
            mask=Mask(
                size=mask_rle["size"],
                counts=mask_rle["counts"],
            ),
        )

    def __get_session(self, session_id: str):
        """
        Safely retrieve a session by ID, or raise an error if it doesn't exist.
        """
        session = self.session_states.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def __get_session_stats(self):
        """
        Return a string summarizing all active sessions and GPU usage (for debugging).
        """
        live_session_strs = []
        for sid, sdata in self.session_states.items():
            frames = sdata["state"].get("num_frames", 0)
            objs = len(sdata["state"].get("obj_ids", []))
            live_session_strs.append(f"'{sid}' ({frames} frames, {objs} objects)")

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() // 1024**2
            mem_reserved = torch.cuda.memory_reserved() // 1024**2
            mem_alloc_max = torch.cuda.max_memory_allocated() // 1024**2
            mem_reserved_max = torch.cuda.max_memory_reserved() // 1024**2
            mem_str = (
                f"{mem_alloc} MiB used, {mem_reserved} MiB reserved (max: "
                f"{mem_alloc_max} / {mem_reserved_max})"
            )
        else:
            mem_str = "GPU not in use"

        session_stats_str = (
            f"Live sessions: [{', '.join(live_session_strs)}], GPU mem: {mem_str}"
        )
        return session_stats_str

    def __clear_session_state(self, session_id: str) -> bool:
        """
        Remove the session from memory. Returns True if successfully removed,
        otherwise False.
        """
        session = self.session_states.pop(session_id, None)
        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{self.__get_session_stats()}"
            )
            return False
        else:
            logger.info(f"removed session {session_id}; {self.__get_session_stats()}")
            return True