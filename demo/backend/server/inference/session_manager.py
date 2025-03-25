# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/session_manager.py
"""
Module for managing inference sessions.
Provides functions for retrieving a session, clearing session state, 
and generating session statistics.
"""

import logging
import torch

def get_session(session_states: dict, session_id: str):
    """
    Retrieve a session from the session_states dictionary.

    Args:
        session_states (dict): Dictionary containing active sessions.
        session_id (str): The ID of the session to retrieve.

    Returns:
        dict: The session data.

    Raises:
        RuntimeError: If the session is not found.
    """
    session = session_states.get(session_id, None)
    if session is None:
        raise RuntimeError(f"Cannot find session {session_id}; it might have expired")
    return session

def get_session_stats(session_states: dict) -> str:
    """
    Generate a summary string for all active sessions and GPU memory usage.

    Args:
        session_states (dict): Dictionary containing active sessions.

    Returns:
        str: A summary string.
    """
    live_session_strs = []
    for sid, sdata in session_states.items():
        frames = sdata["state"].get("num_frames", 0)
        objs = len(sdata["state"].get("obj_ids", []))
        live_session_strs.append(f"'{sid}' ({frames} frames, {objs} objects)")
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() // 1024**2
        mem_reserved = torch.cuda.memory_reserved() // 1024**2
        mem_alloc_max = torch.cuda.max_memory_allocated() // 1024**2
        mem_reserved_max = torch.cuda.max_memory_reserved() // 1024**2
        mem_str = (f"{mem_alloc} MiB used, {mem_reserved} MiB reserved (max: "
                   f"{mem_alloc_max} / {mem_reserved_max})")
    else:
        mem_str = "GPU not in use"
    session_stats_str = f"Live sessions: [{', '.join(live_session_strs)}], GPU mem: {mem_str}"
    return session_stats_str

def clear_session_state(session_states: dict, session_id: str) -> bool:
    """
    Remove a session from the session_states dictionary.

    Args:
        session_states (dict): Dictionary of active sessions.
        session_id (str): The ID of the session to remove.

    Returns:
        bool: True if the session was successfully removed; False otherwise.
    """
    session = session_states.pop(session_id, None)
    logger = logging.getLogger(__name__)
    if session is None:
        logger.warning(
            f"Cannot close session {session_id} as it does not exist; {get_session_stats(session_states)}"
        )
        return False
    else:
        logger.info(f"Removed session {session_id}; {get_session_stats(session_states)}")
        return True