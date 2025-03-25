# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generator, Tuple

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import Flask, make_response, Request, request, Response, send_from_directory
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.multipart import MultipartResponseBuilder
from inference.inference_api import InferenceAPI
from strawberry.flask.views import GraphQLView

logger = logging.getLogger(__name__)
# Force DEBUG logs
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }
    # Required to see session-id for later download
    # TODO add list-sessions endpoint
    logger.info(f"propagate_in_video: {args}")

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, **args)
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    logger.info(f"gen_track_with_mask_stream: {session_id}")
    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        for chunk in inference_api.propagate_in_video(request=request):
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()


def gen_frames_stream(boundary: str, session_id: str) -> Generator[bytes, None, None]:
    """
    Generate a stream of multipart responses containing JPEG-encoded video frames.

    This function retrieves frames from the InferenceAPI, packages them into multipart
    responses with appropriate headers, and yields them for streaming to the client.

    Args:
        boundary: The multipart boundary string used to separate parts in the response.
        session_id: The ID of the session to retrieve frames from.

    Yields:
        bytes: Multipart response parts containing JPEG-encoded frame data.

    Raises:
        RuntimeError: Propagated from InferenceAPI.download_frames if the session is invalid.
    """
    logger.info(f"Starting frame streaming for session {session_id}")
    for frame_idx, jpeg_bytes in inference_api.download_frames(session_id):
        yield MultipartResponseBuilder.build(
            boundary=boundary,
            headers={
                "Content-Type": "image/jpeg",
                "Frame-Index": str(frame_idx),
            },
            body=jpeg_bytes,
        ).get_message()
    logger.info(f"Completed frame streaming for session {session_id}")


@app.route("/download_frames", methods=["POST"])
def download_frames() -> Response:
    """
    Endpoint to download video frames for a given session as a stream of JPEG images.

    Expects a JSON body with a 'session_id' field. Returns a multipart response where each
    part contains a JPEG-encoded frame and its index in the 'Frame-Index' header. This
    ensures frames are delivered exactly as processed by the server, aligning with masks.

    Returns:
        Response: A Flask Response object streaming frames with MIME type
                  'multipart/x-savi-stream'.

    Raises:
        400: If 'session_id' is missing in the request body.
        404: If the specified session_id does not exist.
    """
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        logger.error("Received request with missing session_id")
        return make_response("Missing session_id", 400)
    if session_id not in inference_api.session_states:
        logger.error(f"Session {session_id} not found")
        return make_response(f"Session {session_id} not found", 404)

    logger.info(f"Processing download_frames request for session {session_id}")
    boundary = "frame"
    frame_stream = gen_frames_stream(boundary, session_id)
    return Response(frame_stream, mimetype="multipart/x-savi-stream; boundary=" + boundary)


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)