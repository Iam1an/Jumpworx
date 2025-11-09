from __future__ import annotations

"""
Centralized MediaPipe-style joint indices and names.

This is the single source of truth for:
- Landmark indices
- Common joint groups
- Human-readable names for metrics/coaching

All downstream code (metrics, IO, coaching) should import from here instead of
hardcoding numeric indices.
"""

# Raw landmark indices (MediaPipe-style)
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Useful joint groups
HIP_IDS = (LEFT_HIP, RIGHT_HIP)
SHOULDER_IDS = (LEFT_SHOULDER, RIGHT_SHOULDER)
KNEE_IDS = (LEFT_KNEE, RIGHT_KNEE)
ANKLE_IDS = (LEFT_ANKLE, RIGHT_ANKLE)
FOOT_INDEX_IDS = (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX)
WRIST_IDS = (LEFT_WRIST, RIGHT_WRIST)
ELBOW_IDS = (LEFT_ELBOW, RIGHT_ELBOW)
HAND_IDS = (
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_PINKY,
    RIGHT_PINKY,
    LEFT_INDEX,
    RIGHT_INDEX,
    LEFT_THUMB,
    RIGHT_THUMB,
)

# Human-readable names for logging, metrics, coaching
JOINT_NAME_BY_INDEX = {
    NOSE: "nose",
    LEFT_EYE_INNER: "left_eye_inner",
    LEFT_EYE: "left_eye",
    LEFT_EYE_OUTER: "left_eye_outer",
    RIGHT_EYE_INNER: "right_eye_inner",
    RIGHT_EYE: "right_eye",
    RIGHT_EYE_OUTER: "right_eye_outer",
    LEFT_EAR: "left_ear",
    RIGHT_EAR: "right_ear",
    MOUTH_LEFT: "mouth_left",
    MOUTH_RIGHT: "mouth_right",
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_WRIST: "left_wrist",
    RIGHT_WRIST: "right_wrist",
    LEFT_PINKY: "left_pinky",
    RIGHT_PINKY: "right_pinky",
    LEFT_INDEX: "left_index",
    RIGHT_INDEX: "right_index",
    LEFT_THUMB: "left_thumb",
    RIGHT_THUMB: "right_thumb",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
    LEFT_KNEE: "left_knee",
    RIGHT_KNEE: "right_knee",
    LEFT_ANKLE: "left_ankle",
    RIGHT_ANKLE: "right_ankle",
    LEFT_HEEL: "left_heel",
    RIGHT_HEEL: "right_heel",
    LEFT_FOOT_INDEX: "left_foot_index",
    RIGHT_FOOT_INDEX: "right_foot_index",
}

__all__ = [
    # Indices
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    # Groups
    "HIP_IDS", "SHOULDER_IDS", "KNEE_IDS",
    "ANKLE_IDS", "FOOT_INDEX_IDS",
    "WRIST_IDS", "ELBOW_IDS", "HAND_IDS",
    # Maps
    "JOINT_NAME_BY_INDEX",
]
