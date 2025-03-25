# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/inference_api.py
"""
This module provides the main InferenceAPI class that manages inference sessions,
delegating operations to specialized modules. This file replaces the original predictor.py.
"""

import contextlib
import logging
import os
import uuid
from pathlib import Path
from threading import Lock

import torch
import cv2
from PIL import Image
from io import BytesIO

from app_conf import APP_ROOT, MODEL_SIZE, DATA_PATH
from inference.data_types import (
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
    PropagateDataResponse,
    PropagateInVideoRequest,
    RemoveObjectRequest,
    RemoveObjectResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from typing import Generator, List

from sam2.build_sam import build_sam2_video_predictor

# Import our newly created modules
from . import inference_operations
from . import downloaders  # Updated import to reference downloaders module
from . import session_manager
from data.data_types import SessionInfo  # Added for list_sessions


class InferenceAPI:
    """
    Core class responsible for managing inference sessions and performing segmentation.
    Operations are delegated to helper modules for clarity and maintainability.
    """

    def __init__(self) -> None:
        super(InferenceAPI, self).__init__()
        self.session_states = {}
        self.score_thresh = 0

        # Choose checkpoint & configuration based on MODEL_SIZE.
        if MODEL_SIZE == "tiny":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif MODEL_SIZE == "small":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif MODEL_SIZE == "large":
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:  # default base_plus
            checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        # Select device.
        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        if force_cpu_device:
            logging.info("Forcing CPU device for SAM 2 demo")
        if torch.cuda.is_available() and not force_cpu_device:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not force_cpu_device:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logging.info(f"Using device: {device}")
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            logging.warning(
                "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might give "
                "numerically different outputs and sometimes degraded performance on MPS."
            )
        self.device = device
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        self.inference_lock = Lock()

    def autocast_context(self):
        """
        Returns an automatic mixed-precision context if on CUDA, otherwise a no-op context.
        """
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def _read_raw_frames(self, path: str):
        """
        Read raw frames from a video file using OpenCV.

        Args:
            path (str): Relative path to the video.

        Returns:
            List[np.ndarray]: List of frames read from the video.
        """
        video_path = str(Path(DATA_PATH) / path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video at {video_path}")
            return []
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        logging.info(f"Loaded {len(frames)} raw frames from {video_path}")
        return frames

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        """
        Start an inference session for a given video path.

        Args:
            request (StartSessionRequest): Request containing the video path.

        Returns:
            StartSessionResponse: Response containing the new session ID.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())
            offload_video_to_cpu = self.device.type == "mps"
            inference_state = self.predictor.init_state(
                request.path,
                offload_video_to_cpu=offload_video_to_cpu,
            )
            raw_frames = self._read_raw_frames(request.path)
            inference_state["images_original"] = raw_frames
            self.session_states[session_id] = {
                "canceled": False,
                "state": inference_state,
            }
            return StartSessionResponse(session_id=session_id)

    def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        """
        Close an inference session.

        Args:
            request (CloseSessionRequest): Request with the session ID.

        Returns:
            CloseSessionResponse: Response indicating success.
        """
        with self.inference_lock:
            success = session_manager.clear_session_state(self.session_states, request.session_id)
            return CloseSessionResponse(success=success)

    def add_points(self, request: AddPointsRequest) -> PropagateDataResponse:
        """
        Add point prompts and update segmentation masks.

        Args:
            request (AddPointsRequest): Request with point data.

        Returns:
            PropagateDataResponse: Updated masks for the frame.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            return inference_operations.add_points_operation(self.predictor, inference_state, request, score_thresh=self.score_thresh)

    def add_mask(self, request) -> PropagateDataResponse:
        """
        Add a mask prompt directly.

        Args:
            request: Request with mask data.

        Returns:
            PropagateDataResponse: Updated mask for the frame.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            return inference_operations.add_mask_operation(self.predictor, inference_state, request, score_thresh=self.score_thresh)

    def clear_points_in_frame(self, request: ClearPointsInFrameRequest) -> PropagateDataResponse:
        """
        Clear point prompts in a specific frame.

        Args:
            request (ClearPointsInFrameRequest): Request with frame and object data.

        Returns:
            PropagateDataResponse: Updated mask for the frame.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            return inference_operations.clear_points_in_frame_operation(self.predictor, inference_state, request, score_thresh=self.score_thresh)

    def clear_points_in_video(self, request: ClearPointsInVideoRequest) -> ClearPointsInVideoResponse:
        """
        Clear all point prompts in the video.

        Args:
            request (ClearPointsInVideoRequest): Request with the session ID.

        Returns:
            ClearPointsInVideoResponse: Confirmation of clearing.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            return inference_operations.clear_points_in_video_operation(self.predictor, inference_state, request)

    def remove_object(self, request: RemoveObjectRequest) -> RemoveObjectResponse:
        """
        Remove an object from the segmentation state.

        Args:
            request (RemoveObjectRequest): Request with the object ID.

        Returns:
            RemoveObjectResponse: Updated masks after removal.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            return inference_operations.remove_object_operation(self.predictor, inference_state, request, score_thresh=self.score_thresh)

    def propagate_in_video(self, request: PropagateInVideoRequest) -> Generator[PropagateDataResponse, None, None]:
        """
        Propagate prompts throughout the video and yield updated masks.

        Args:
            request (PropagateInVideoRequest): Request with propagation parameters.

        Returns:
            Generator[PropagateDataResponse]: Yields updated masks per frame.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            session["canceled"] = False
            inference_state = session["state"]
            logging.info(f"Propagating in session {request.session_id}: {session_manager.get_session_stats(self.session_states)}")
            return inference_operations.propagate_in_video_operation(self.predictor, inference_state, request, session, score_thresh=self.score_thresh)

    def cancel_propagate_in_video(self, request: CancelPropagateInVideoRequest) -> CancelPorpagateResponse:
        """
        Cancel an ongoing propagation process.

        Args:
            request (CancelPropagateInVideoRequest): Request with the session ID.

        Returns:
            CancelPorpagateResponse: Confirmation of cancellation.
        """
        with self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            return inference_operations.cancel_propagate_in_video_operation(session, request)

    def download_masks(self, request: DownloadMasksRequest) -> DownloadMasksResponse:
        """
        Download all masks for a session.

        Args:
            request (DownloadMasksRequest): Request with the session ID.

        Returns:
            DownloadMasksResponse: Response with masks for each frame.
        """
        with self.autocast_context(), self.inference_lock:
            session = session_manager.get_session(self.session_states, request.session_id)
            inference_state = session["state"]
            if "masks_cache" not in inference_state:
                inference_state["masks_cache"] = {}
            num_frames = inference_state.get("num_frames", 0)
            if num_frames <= 0:
                logging.warning(f"No frames available in session {request.session_id}")
                return DownloadMasksResponse(results=[])
            # Updated to use downloaders.download_masks_operation instead of inference_operations
            return downloaders.download_masks_operation(self.predictor, inference_state, num_frames, score_thresh=self.score_thresh)

    def download_frames(self, session_id: str):
        """
        Download raw video frames as JPEG-encoded images.

        Args:
            session_id (str): The session ID.

        Returns:
            Generator[Tuple[int, bytes]]: Yields tuples of frame index and JPEG bytes.
        """
        session = session_manager.get_session(self.session_states, session_id)
        return downloaders.download_frames_operation(session)

    def list_sessions(self) -> List[SessionInfo]:
        """
        Retrieve metadata for all active inference sessions.

        Returns:
            List[SessionInfo]: A list of SessionInfo objects containing metadata for each session.
        """
        with self.inference_lock:
            sessions = []
            for session_id, session_data in self.session_states.items():
                inference_state = session_data["state"]
                # Extract session metadata; fall back to 0 if keys are missing
                num_frames = inference_state.get("num_frames", len(inference_state.get("images_original", [])))
                num_objects = len(inference_state.get("obj_ids", []))
                # Use time.time() as a proxy if exact timestamps aren't stored
                start_time = inference_state.get("start_time", 0.0)
                last_use_time = inference_state.get("last_use_time", 0.0)
                sessions.append(
                    SessionInfo(
                        session_id=session_id,
                        start_time=start_time,
                        last_use_time=last_use_time,
                        num_frames=num_frames,
                        num_objects=num_objects,
                    )
                )
            logging.info(f"Listed {len(sessions)} active sessions")
            return sessions