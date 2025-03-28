#!/usr/bin/env python3

"""
Script to download bounding boxes for a given session from the SAM 2 backend and save them to disk as YOLO format bounding boxes.

Usage:
    python download_boxes.py [--session-id <SESSION_ID>] --output-dir <OUTPUT_DIR> [--endpoint <ENDPOINT_DOMAIN>]

If --session-id is not provided, the script will query active sessions and either use the only available session or prompt the user to select one if multiple sessions exist.

The endpoint domain should point to the server where the backend is hosted. The script will automatically construct the GraphQL endpoint URL by appending '/graphql'.

Requirements:
    - pip install requests tqdm
"""

import argparse
import logging
import os
import math
from typing import List, Dict

import requests
from tqdm import tqdm


def list_sessions(graphql_endpoint: str) -> List[Dict]:
    """
    Retrieve a list of active inference sessions from the backend.

    Args:
        graphql_endpoint: The GraphQL endpoint URL (e.g., "http://localhost:5000/graphql").

    Returns:
        A list of dictionaries containing session metadata:
        [
            {
                "sessionId": str,
                "startTime": float,
                "lastUseTime": float,
                "numFrames": int,
                "numObjects": int
            },
            ...
        ]

    Raises:
        Exception: If the GraphQL request fails or the response is invalid.
    """
    logging.info("Fetching list of active sessions from %s", graphql_endpoint)
    query = """
    query {
        sessions {
            sessionId
            startTime
            lastUseTime
            numFrames
            numObjects
        }
    }
    """
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(graphql_endpoint, json={"query": query}, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise Exception(result["errors"])
        sessions = result["data"]["sessions"]
        logging.info("Found %d active sessions", len(sessions))
        return sessions
    except Exception as e:
        logging.exception("Failed to list sessions")
        raise Exception(f"Failed to list sessions: {str(e)}")


def download_boxes(session_id: str, graphql_endpoint: str) -> List[Dict]:
    """
    Download bounding boxes for a given session from the backend.

    Args:
        session_id: The ID of the session to download bounding boxes for.
        graphql_endpoint: The GraphQL endpoint URL (e.g., "http://localhost:5000/graphql").

    Returns:
        A list of dictionaries containing frame_index and boxes data.
        Each dictionary corresponds to a frame and has the structure:
        {
            "frameIndex": int,
            "boxes": [
                {
                    "objectId": int,
                    "box": [x_center_norm, y_center_norm, width_norm, height_norm]
                },
                ...
            ]
        }

    Raises:
        Exception: If the GraphQL request fails or the response is invalid.
    """
    logging.info("Setting up GraphQL query for endpoint %s", graphql_endpoint)
    query = """
    mutation DownloadBoxes($input: DownloadBoxesInput!) {
        downloadBoxes(input: $input) {
            boxes {
                frameIndex
                boxes {
                    objectId
                    box
                }
            }
        }
    }
    """
    variables = {
        "input": {
            "sessionId": session_id,
            "format": "yolo"
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(graphql_endpoint, json={"query": query, "variables": variables}, headers=headers, timeout=3600)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise Exception(result["errors"])
        boxes = result["data"]["downloadBoxes"]["boxes"]
        logging.info("Downloaded bounding boxes for %d frames", len(boxes))
        return boxes
    except Exception as e:
        logging.exception("Failed to download bounding boxes for session %s", session_id)
        raise Exception(f"Failed to download bounding boxes: {str(e)}")


def save_boxes_to_disk(boxes: List[Dict], output_dir: str) -> None:
    """
    Save the downloaded bounding boxes to disk as text files.
    Each file is named 'frame_%06d.txt' corresponding to the frame index.
    Boxes with invalid dimensions or out-of-bound values are filtered out.

    Invalid box checks:
      - The box must be a list of exactly 4 elements.
      - All elements must be numeric and finite.
      - x_center_norm and y_center_norm must be within [0, 1].
      - width_norm and height_norm must be > 0 and within (0, 1].
      
    Args:
        boxes: List of dictionaries containing bounding boxes per frame.
        output_dir: Directory where the text files will be saved.
    """
    logging.info("Ensuring output directory %s exists", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Saving bounding boxes to disk")
    for frame_data in tqdm(boxes, desc="Saving boxes", unit="frame"):
        frame_index = frame_data["frameIndex"]
        filename = os.path.join(output_dir, f"frame_{frame_index:06d}.txt")
        lines = []
        for box_obj in frame_data["boxes"]:
            object_id = box_obj["objectId"]
            box_values = box_obj["box"]

            # Check that box_values is a list of exactly 4 elements.
            if not isinstance(box_values, list) or len(box_values) != 4:
                logging.warning("Skipping invalid bounding box for frame %d: expected list of 4 elements, got %s", frame_index, box_values)
                continue

            # Check that all values are numeric and finite.
            if not all(isinstance(val, (int, float)) for val in box_values):
                logging.warning("Skipping invalid bounding box for frame %d: non-numeric values in %s", frame_index, box_values)
                continue
            if any(not math.isfinite(val) for val in box_values):
                logging.warning("Skipping invalid bounding box for frame %d: non-finite values in %s", frame_index, box_values)
                continue

            x_center, y_center, width, height = box_values

            # Check that center coordinates are within [0, 1].
            if not (0 <= x_center <= 1) or not (0 <= y_center <= 1):
                logging.warning("Skipping invalid bounding box for frame %d: center coordinates out of range (x=%.6f, y=%.6f)", frame_index, x_center, y_center)
                continue

            # Check that width and height are greater than 0 and within (0, 1].
            if width <= 0 or height <= 0:
                logging.warning("Skipping invalid bounding box for frame %d: non-positive dimensions (width=%.6f, height=%.6f)", frame_index, width, height)
                continue
            if width > 1 or height > 1:
                logging.warning("Skipping invalid bounding box for frame %d: dimensions exceed normalized range (width=%.6f, height=%.6f)", frame_index, width, height)
                continue

            line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(object_id, x_center, y_center, width, height)
            lines.append(line)
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        logging.info("Saved bounding boxes for frame %d to %s", frame_index, filename)


def select_session(sessions: List[Dict]) -> str:
    """
    Prompt the user to select a session from a list of active sessions.

    Args:
        sessions: List of session metadata dictionaries.

    Returns:
        str: The selected session ID.

    Raises:
        SystemExit: If the user input is invalid or they choose to exit.
    """
    print("\nMultiple active sessions found:")
    for i, session in enumerate(sessions, 1):
        print(f"{i}. Session ID: {session['sessionId']}, Frames: {session['numFrames']}, Objects: {session['numObjects']}")
    print(f"{len(sessions) + 1}. Exit")
    
    while True:
        try:
            choice = int(input(f"\nEnter the number of the session to download (1-{len(sessions) + 1}): "))
            if choice == len(sessions) + 1:
                logging.info("User chose to exit")
                raise SystemExit("Exiting at user request.")
            if 1 <= choice <= len(sessions):
                selected_session = sessions[choice - 1]["sessionId"]
                logging.info("User selected session %s", selected_session)
                return selected_session
            else:
                print(f"Please enter a number between 1 and {len(sessions) + 1}.")
        except ValueError:
            print("Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(
        description="Download bounding boxes for a session from the SAM 2 backend and save them as text files in YOLO format."
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="The session ID to download bounding boxes for. If not provided, the script will list active sessions."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the bounding box files will be saved."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000",
        help="Endpoint domain where the backend is hosted (default: http://localhost:5000)."
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Construct the GraphQL endpoint URL by appending '/graphql'
    graphql_endpoint = args.endpoint.rstrip("/") + "/graphql"
    
    try:
        # If session_id is provided, use it directly; otherwise, fetch active sessions.
        if args.session_id:
            session_id = args.session_id
            logging.info("Using provided session ID: %s", session_id)
        else:
            sessions = list_sessions(graphql_endpoint)
            if not sessions:
                logging.error("No active sessions found")
                print("No active sessions found. Please start a session first.")
                exit(1)
            elif len(sessions) == 1:
                session_id = sessions[0]["sessionId"]
                logging.info("Automatically selected the only session: %s", session_id)
                print(f"Using the only active session: {session_id}")
            else:
                session_id = select_session(sessions)
        
        logging.info("Starting bounding boxes download for session %s", session_id)
        boxes = download_boxes(session_id, graphql_endpoint)
        if not boxes:
            logging.warning("No bounding boxes found for session %s", session_id)
            print(f"No bounding boxes found for session {session_id}.")
            return
        save_boxes_to_disk(boxes, args.output_dir)
        logging.info("Successfully downloaded and saved bounding boxes for %d frames.", len(boxes))
        print(f"Successfully saved bounding boxes for {len(boxes)} frames to {args.output_dir}")
    except SystemExit as e:
        logging.info(str(e))
        exit(0)
    except Exception as e:
        logging.exception("An error occurred during processing:")
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()