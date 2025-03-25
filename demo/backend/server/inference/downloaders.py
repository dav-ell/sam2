# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/downloaders.py
"""
Module for downloading frames and masks.
Provides functions to generate JPEG-encoded frames and to compute/download masks.
"""

import logging
from typing import Generator, Tuple
import cv2
from PIL import Image
from io import BytesIO
from inference.data_types import DownloadMasksResponse, PropagateDataResponse

def download_frames_operation(session: dict) -> Generator[Tuple[int, bytes], None, None]:
    """
    Generate a stream of JPEG-encoded frames from the original video data.

    Args:
        session (dict): The session dictionary containing 'images_original' in its state.

    Yields:
        Tuple[int, bytes]: Frame index and JPEG-encoded frame bytes.

    Raises:
        RuntimeError: If no original frames are found.
    """
    logger = logging.getLogger(__name__)
    inference_state = session.get("state", {})
    if "images_original" not in inference_state:
        logger.error(f"No original frames found in session")
        raise RuntimeError("Session has no 'images_original' data")
    video_frames = inference_state["images_original"]
    num_frames = len(video_frames)
    logger.info(f"Found {num_frames} raw frames in session")
    if num_frames == 0:
        logger.warning("Session contains zero frames (original)")
        return
    for frame_idx, frame_bgr in enumerate(video_frames):
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            buf = BytesIO()
            frame_pil.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()
            buf.close()
            yield frame_idx, jpeg_bytes
        except Exception as e:
            logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
            raise RuntimeError(f"Error processing frame {frame_idx}: {str(e)}")

def download_masks_operation(predictor, inference_state: dict, num_frames: int, score_thresh=0) -> DownloadMasksResponse:
    """
    Retrieve masks for all frames. Uses a cache to avoid recomputation.

    Args:
        predictor: The SAM predictor instance.
        inference_state (dict): The current inference state.
        num_frames (int): Total number of frames.
        score_thresh: Score threshold to binarize masks.

    Returns:
        DownloadMasksResponse: Response containing masks for all frames.
    """
    logger = logging.getLogger(__name__)
    if "masks_cache" not in inference_state:
        inference_state["masks_cache"] = {}
    results = []
    from .inference_operations import PropagateDataResponse
    from .mask_utils import get_rle_mask_list
    for frame_idx in range(num_frames):
        if frame_idx in inference_state["masks_cache"]:
            results.append(inference_state["masks_cache"][frame_idx])
        else:
            outputs = predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
                reverse=False,
            )
            mask_response = None
            for frame_idx_result, obj_ids, video_res_masks in outputs:
                masks_binary = (video_res_masks > score_thresh)[:, 0].cpu().numpy()
                rle_mask_list = get_rle_mask_list(obj_ids, masks_binary)
                mask_response = PropagateDataResponse(frame_index=frame_idx_result, results=rle_mask_list)
                break
            if mask_response is None:
                mask_response = PropagateDataResponse(frame_index=frame_idx, results=[])
            inference_state["masks_cache"][frame_idx] = mask_response
            results.append(mask_response)
    return DownloadMasksResponse(results=results)