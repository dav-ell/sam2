# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import av
import strawberry
from app_conf import (
    DATA_PATH,
    DEFAULT_VIDEO_PATH,
    MAX_UPLOAD_VIDEO_DURATION,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.data_types import (
    AddPointsInput,
    CancelPropagateInVideo,
    CancelPropagateInVideoInput,
    ClearPointsInFrameInput,
    ClearPointsInVideo,
    ClearPointsInVideoInput,
    CloseSession,
    CloseSessionInput,
    DownloadMasksInput,
    DownloadMasksResponse,
    DownloadBoxesInput,
    YOLOBoxForObject,
    BoxesListOnFrame,
    DownloadBoxesResponse,
    RemoveObjectInput,
    RLEMask,
    RLEMaskForObject,
    RLEMaskListOnFrame,
    SessionInfo,  # Added new import
    StartSession,
    StartSessionInput,
    Video,
)
from data.loader import get_video
from data.store import get_videos
from data.transcoder import get_video_metadata, transcode, VideoMetadata
from inference.data_types import (
    AddPointsRequest,
    CancelPropagateInVideoRequest,
    ClearPointsInFrameRequest,
    ClearPointsInVideoRequest,
    CloseSessionRequest,
    DownloadMasksRequest,
    PropagateDataResponse,
    RemoveObjectRequest,
    StartSessionRequest,
)
from inference.inference_api import InferenceAPI
from strawberry import relay
from strawberry.file_uploads import Upload


@strawberry.type
class Query:

    @strawberry.field
    def default_video(self) -> Video:
        """
        Return the default video.

        The default video can be set with the DEFAULT_VIDEO_PATH environment
        variable. It will return the video that matches this path. If no video
        is found, it will return the first video.
        """
        all_videos = get_videos()

        # Find the video that matches the default path and return that as
        # default video.
        for _, v in all_videos.items():
            if v.path == DEFAULT_VIDEO_PATH:
                return v

        # Fallback is returning the first video
        return next(iter(all_videos.values()))

    @relay.connection(relay.ListConnection[Video])
    def videos(
        self,
    ) -> Iterable[Video]:
        """
        Return all available videos.
        """
        all_videos = get_videos()
        return all_videos.values()

    @strawberry.field
    def sessions(self, info: strawberry.Info) -> List[SessionInfo]:
        """
        Return a list of all active inference sessions.

        Args:
            info: Strawberry context info containing the inference_api.

        Returns:
            List[SessionInfo]: A list of metadata for each active session.
        """
        inference_api: InferenceAPI = info.context["inference_api"]
        return inference_api.list_sessions()


@strawberry.type
class Mutation:

    @strawberry.mutation
    def upload_video(
        self,
        file: Upload,
        start_time_sec: Optional[float] = None,
        duration_time_sec: Optional[float] = None,
    ) -> Video:
        """
        Receive a video file and store it in the configured S3 bucket.
        """
        max_time = MAX_UPLOAD_VIDEO_DURATION
        filepath, file_key, vm = process_video(
            file,
            max_time=max_time,
            start_time_sec=start_time_sec,
            duration_time_sec=duration_time_sec,
        )

        video = get_video(
            filepath,
            UPLOADS_PATH,
            file_key=file_key,
            width=vm.width,
            height=vm.height,
            generate_poster=False,
        )
        return video

    @strawberry.mutation
    def start_session(
        self, input: StartSessionInput, info: strawberry.Info
    ) -> StartSession:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = StartSessionRequest(
            type="start_session",
            path=f"{DATA_PATH}/{input.path}",
        )

        response = inference_api.start_session(request=request)

        return StartSession(session_id=response.session_id)

    @strawberry.mutation
    def close_session(
        self, input: CloseSessionInput, info: strawberry.Info
    ) -> CloseSession:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = CloseSessionRequest(
            type="close_session",
            session_id=input.session_id,
        )
        response = inference_api.close_session(request)
        return CloseSession(success=response.success)

    @strawberry.mutation
    def add_points(
        self, input: AddPointsInput, info: strawberry.Info
    ) -> RLEMaskListOnFrame:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = AddPointsRequest(
            type="add_points",
            session_id=input.session_id,
            frame_index=input.frame_index,
            object_id=input.object_id,
            points=input.points,
            labels=input.labels,
            clear_old_points=input.clear_old_points,
        )
        reponse = inference_api.add_points(request)

        return RLEMaskListOnFrame(
            frame_index=reponse.frame_index,
            rle_mask_list=[
                RLEMaskForObject(
                    object_id=r.object_id,
                    rle_mask=RLEMask(counts=r.mask.counts, size=r.mask.size, order="F"),
                )
                for r in reponse.results
            ],
        )

    @strawberry.mutation
    def remove_object(
        self, input: RemoveObjectInput, info: strawberry.Info
    ) -> List[RLEMaskListOnFrame]:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = RemoveObjectRequest(
            type="remove_object", session_id=input.session_id, object_id=input.object_id
        )

        response = inference_api.remove_object(request)

        return [
            RLEMaskListOnFrame(
                frame_index=res.frame_index,
                rle_mask_list=[
                    RLEMaskForObject(
                        object_id=r.object_id,
                        rle_mask=RLEMask(
                            counts=r.mask.counts, size=r.mask.size, order="F"
                        ),
                    )
                    for r in res.results
                ],
            )
            for res in response.results
        ]

    @strawberry.mutation
    def clear_points_in_frame(
        self, input: ClearPointsInFrameInput, info: strawberry.Info
    ) -> RLEMaskListOnFrame:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = ClearPointsInFrameRequest(
            type="clear_points_in_frame",
            session_id=input.session_id,
            frame_index=input.frame_index,
            object_id=input.object_id,
        )

        response = inference_api.clear_points_in_frame(request)

        return RLEMaskListOnFrame(
            frame_index=response.frame_index,
            rle_mask_list=[
                RLEMaskForObject(
                    object_id=r.object_id,
                    rle_mask=RLEMask(counts=r.mask.counts, size=r.mask.size, order="F"),
                )
                for r in response.results
            ],
        )

    @strawberry.mutation
    def clear_points_in_video(
        self, input: ClearPointsInVideoInput, info: strawberry.Info
    ) -> ClearPointsInVideo:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = ClearPointsInVideoRequest(
            type="clear_points_in_video",
            session_id=input.session_id,
        )
        response = inference_api.clear_points_in_video(request)
        return ClearPointsInVideo(success=response.success)

    @strawberry.mutation
    def cancel_propagate_in_video(
        self, input: CancelPropagateInVideoInput, info: strawberry.Info
    ) -> CancelPropagateInVideo:
        inference_api: InferenceAPI = info.context["inference_api"]

        request = CancelPropagateInVideoRequest(
            type="cancel_propagate_in_video",
            session_id=input.session_id,
        )
        response = inference_api.cancel_propagate_in_video(request)
        return CancelPropagateInVideo(success=response.success)

    @strawberry.mutation
    def download_masks(
        self, input: DownloadMasksInput, info: strawberry.Info
    ) -> DownloadMasksResponse:
        """
        Retrieve all masks for all frames in the specified session.

        Args:
            input: The DownloadMasksInput containing the session_id.
            info: Strawberry context info containing the inference_api.

        Returns:
            A DownloadMasksResponse with a list of RLEMaskListOnFrame objects.
        """
        inference_api: InferenceAPI = info.context["inference_api"]

        request = DownloadMasksRequest(
            type="download_masks",
            session_id=input.session_id,
        )
        response = inference_api.download_masks(request)

        return DownloadMasksResponse(
            masks=[
                RLEMaskListOnFrame(
                    frame_index=res.frame_index,
                    rle_mask_list=[
                        RLEMaskForObject(
                            object_id=r.object_id,
                            rle_mask=RLEMask(
                                counts=r.mask.counts, size=r.mask.size, order="F"
                            ),
                        )
                        for r in res.results
                    ],
                )
                for res in response.results
            ]
        )

    @strawberry.mutation
    def download_boxes(
        self, input: DownloadBoxesInput, info: strawberry.Info
    ) -> DownloadBoxesResponse:
        """
        Retrieve bounding boxes for all frames in the specified session in the requested format.
        Currently, only YOLO format is supported.
        
        Args:
            input: The DownloadBoxesInput containing the session_id and desired format.
            info: Strawberry context info containing the inference_api.
        
        Returns:
            A DownloadBoxesResponse with a list of BoxesListOnFrame objects.
        """
        inference_api: InferenceAPI = info.context["inference_api"]

        # Reuse the download_masks functionality to get the masks
        request = DownloadMasksRequest(
            type="download_masks",
            session_id=input.session_id,
        )
        response = inference_api.download_masks(request=request)
        
        import pycocotools.mask as maskUtils

        def rle_to_yolo(rle):
            # Convert the RLE mask (using counts and size) to YOLO bounding box format.
            # maskUtils.toBbox returns [x, y, width, height] in absolute coordinates.
            bbox = maskUtils.toBbox({"counts": rle.counts, "size": rle.size})
            x, y, w, h = bbox
            img_h, img_w = rle.size
            x_center = x + w / 2.0
            y_center = y + h / 2.0
            return [x_center / img_w, y_center / img_h, w / img_w, h / img_h]

        boxes_response = []
        for mask_list in response.results:  # Access 'results' instead of 'masks'
            boxes_for_frame = []
            for mask_obj in mask_list.results:  # Access 'results' instead of 'rle_mask_list'
                yolo_box = rle_to_yolo(mask_obj.mask)  # Use 'mask' from PropagateDataValue
                boxes_for_frame.append(
                    YOLOBoxForObject(object_id=mask_obj.object_id, box=yolo_box)
                )
            boxes_response.append(
                BoxesListOnFrame(frame_index=mask_list.frame_index, boxes=boxes_for_frame)
            )
        
        return DownloadBoxesResponse(boxes=boxes_response)


def get_file_hash(video_path_or_file) -> str:
    if isinstance(video_path_or_file, str):
        with open(video_path_or_file, "rb") as in_f:
            result = hashlib.sha256(in_f.read()).hexdigest()
    else:
        video_path_or_file.seek(0)
        result = hashlib.sha256(video_path_or_file.read()).hexdigest()
    return result


def _get_start_sec_duration_sec(
    start_time_sec: Union[float, None],
    duration_time_sec: Union[float, None],
    max_time: float,
) -> Tuple[float, float]:
    default_seek_t = int(os.environ.get("VIDEO_ENCODE_SEEK_TIME", "0"))
    if start_time_sec is None:
        start_time_sec = default_seek_t

    if duration_time_sec is not None:
        duration_time_sec = min(duration_time_sec, max_time)
    else:
        duration_time_sec = max_time
    return start_time_sec, duration_time_sec


def process_video(
    file: Upload,
    max_time: float,
    start_time_sec: Optional[float] = None,
    duration_time_sec: Optional[float] = None,
) -> Tuple[Optional[str], str, VideoMetadata]:
    """
    Process file upload including video trimming and content moderation checks.

    Returns the filepath, s3_file_key, hash & video metaedata as a tuple.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        in_path = f"{tempdir}/in.mp4"
        out_path = f"{tempdir}/out.mp4"
        with open(in_path, "wb") as in_f:
            in_f.write(file.read())

        try:
            video_metadata = get_video_metadata(in_path)
        except av.InvalidDataError:
            raise Exception("not valid video file")

        if video_metadata.num_video_streams == 0:
            raise Exception("video container does not contain a video stream")
        if video_metadata.width is None or video_metadata.height is None:
            raise Exception("video container does not contain width or height metadata")

        if video_metadata.duration_sec in (None, 0):
            raise Exception("video container does time duration metadata")

        start_time_sec, duration_time_sec = _get_start_sec_duration_sec(
            max_time=max_time,
            start_time_sec=start_time_sec,
            duration_time_sec=duration_time_sec,
        )

        # Transcode video to make sure videos returned to the app are all in
        # the same format, duration, resolution, fps.
        transcode(
            in_path,
            out_path,
            video_metadata,
            seek_t=start_time_sec,
            duration_time_sec=duration_time_sec,
        )

        os.remove(in_path)  # don't need original video now

        out_video_metadata = get_video_metadata(out_path)
        if out_video_metadata.num_video_frames == 0:
            raise Exception(
                "transcode produced empty video; check seek time or your input video"
            )

        filepath = None
        file_key = None
        with open(out_path, "rb") as file_data:
            file_hash = get_file_hash(file_data)
            file_data.seek(0)

            file_key = UPLOADS_PREFIX + "/" + f"{file_hash}.mp4"
            filepath = os.path.join(UPLOADS_PATH, f"{file_hash}.mp4")

        assert filepath is not None and file_key is not None
        shutil.move(out_path, filepath)

        return filepath, file_key, out_video_metadata


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
)