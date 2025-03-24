#!/usr/bin/env python3

"""
Script to download all video frames for a given session from the SAM 2 backend and save them to disk as JPEG images.

Usage:
    python download_frames.py --session-id <SESSION_ID> --output-dir <OUTPUT_DIR> [--endpoint <ENDPOINT>]

Requirements:
    - pip install requests tqdm
"""

import argparse
import logging
import os
from typing import Generator, Tuple

import requests
from email.parser import BytesParser
from tqdm import tqdm


def download_frames(session_id: str, endpoint: str) -> Generator[Tuple[int, bytes], None, None]:
    """
    Download all video frames for a given session from the backend as a stream of JPEG images.

    This function sends a POST request to the /download_frames endpoint, processes the multipart
    response, and yields each frame's index and JPEG data.

    Args:
        session_id: The ID of the session to download frames for.
        endpoint: The endpoint URL (e.g., "http://localhost:5000/download_frames").

    Yields:
        Tuple[int, bytes]: A tuple containing the frame index (int) and JPEG data (bytes).

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the response format is invalid.
    """
    logging.info("Initiating frame download from endpoint %s for session %s", endpoint, session_id)
    url = endpoint
    payload = {"session_id": session_id}
    headers = {"Content-Type": "application/json"}

    # Stream the response to handle large videos efficiently
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=3600)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.exception("Failed to initiate frame download for session %s", session_id)
        raise requests.RequestException(f"Failed to download frames: {str(e)}")

    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("multipart/x-savi-stream"):
        # Extract the boundary string without CRLF
        boundary_str = response.headers["Content-Type"].split("boundary=")[1]
        boundary = b"--" + boundary_str.encode("utf-8")
    else:
        raise ValueError("Unexpected response Content-Type; expected multipart/x-savi-stream")

    buffer = b""
    frame_count = 0
    for chunk in response.iter_content(chunk_size=8192):
        buffer += chunk
        # Process complete parts as long as the boundary is found
        while boundary in buffer:
            part, buffer = buffer.split(boundary, 1)
            part = part.strip(b"\r\n")
            if not part:
                continue
            try:
                # Each part should have headers and body separated by a blank line.
                if b"\r\n\r\n" not in part:
                    logging.error("Incomplete multipart part encountered; skipping part")
                    continue
                headers_part, body = part.split(b"\r\n\r\n", 1)
                headers_obj = BytesParser().parsebytes(headers_part)
                frame_index_value = headers_obj.get("Frame-Index")
                if frame_index_value is None:
                    logging.error("Missing Frame-Index header in multipart part; skipping part")
                    continue
                frame_idx = int(frame_index_value)
                frame_data = body.strip()
                yield frame_idx, frame_data
                frame_count += 1
            except Exception as e:
                logging.error("Failed to parse multipart part: %s", str(e))
                continue
    logging.info("Downloaded %d frames for session %s", frame_count, session_id)


def save_frames_to_disk(frames: Generator[Tuple[int, bytes], None, None], output_dir: str) -> None:
    """
    Save the downloaded frames to disk as JPEG files.

    Args:
        frames: Generator yielding tuples of frame index and JPEG data.
        output_dir: Directory where frame files will be saved.

    Raises:
        OSError: If the output directory cannot be created or written to.
    """
    logging.info("Ensuring output directory %s exists", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Saving frames to disk with progress bar")
    frame_count = 0
    # Using tqdm to show progress; initial count is unknown, so no total is set
    with tqdm(desc="Saving frames", unit="frame") as pbar:
        for frame_idx, frame_data in frames:
            filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            try:
                with open(filename, "wb") as f:
                    f.write(frame_data)
                logging.debug("Saved frame %d to %s", frame_idx, filename)
                frame_count += 1
                pbar.update(1)
            except OSError as os_err:
                logging.error("Failed to write file %s: %s", filename, str(os_err))
                raise
    logging.info("Successfully saved %d frames to %s", frame_count, output_dir)


def main():
    """Main function to parse arguments and execute the frame download."""
    parser = argparse.ArgumentParser(
        description="Download video frames for a session from the SAM 2 backend and save them to disk as JPEG images."
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="The session ID to download frames for."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where frame files will be saved."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000/download_frames",
        help="Endpoint URL (default: http://localhost:5000/download_frames)."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting frame download for session %s", args.session_id)
    try:
        # Download frames from the backend
        frames = download_frames(args.session_id, args.endpoint)
        # Save frames to disk
        save_frames_to_disk(frames, args.output_dir)
        logging.info("Successfully downloaded and saved frames to %s", args.output_dir)
    except Exception as e:
        logging.exception("An error occurred during processing:")
        exit(1)


if __name__ == "__main__":
    main()