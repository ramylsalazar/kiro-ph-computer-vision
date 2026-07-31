"""Face feature extraction using MediaPipe Face Mesh.

Extracts head yaw (via solvePnP with nose-offset fallback) and expression
ratios normalized by face dimensions for scale invariance.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp


# Key landmark indices for Face Mesh (468 landmarks)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
LEFT_MOUTH_CORNER = 291
RIGHT_MOUTH_CORNER = 61
FOREHEAD = 10

# For solvePnP: 6 key points
SOLVEPNP_LANDMARKS = [
    NOSE_TIP,        # Nose tip
    CHIN,            # Chin
    LEFT_EYE_OUTER,  # Left eye outer corner
    RIGHT_EYE_OUTER, # Right eye outer corner
    LEFT_MOUTH_CORNER,  # Left mouth corner
    RIGHT_MOUTH_CORNER, # Right mouth corner
]

# Canonical 3D face model points (approximate, in arbitrary units)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye outer corner
    (43.3, 32.7, -26.0),      # Right eye outer corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1),     # Right mouth corner
], dtype=np.float64)

# Eye landmarks for aspect ratio
# Left eye (from viewer's perspective - actually subject's right eye)
LEFT_EYE_UPPER = [386, 374]
LEFT_EYE_LOWER = [380, 373]
LEFT_EYE_LEFT = 263
LEFT_EYE_RIGHT = 362

# Right eye (from viewer's perspective - actually subject's left eye)
RIGHT_EYE_UPPER = [159, 145]
RIGHT_EYE_LOWER = [153, 144]
RIGHT_EYE_LEFT = 133
RIGHT_EYE_RIGHT = 33

# Mouth landmarks
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
MOUTH_LEFT = 291
MOUTH_RIGHT = 61

# Brow landmarks
LEFT_BROW_INNER = 282
LEFT_BROW_OUTER = 276
RIGHT_BROW_INNER = 52
RIGHT_BROW_OUTER = 46

# Eye center landmarks (for brow-to-eye distance)
LEFT_EYE_CENTER = 473  # iris center if available, fallback to midpoint
RIGHT_EYE_CENTER = 468

# Upper eyelid landmarks for brow distance
LEFT_EYE_TOP = 386
RIGHT_EYE_TOP = 159


@dataclass
class FaceFeatures:
    """Raw face feature measurements from a single frame."""

    yaw_degrees: float  # Head yaw in degrees (negative = left, positive = right)
    mouth_aspect_ratio: float  # Mouth height / mouth width (normalized)
    mouth_corner_lift: float  # Mouth corner elevation relative to center
    left_brow_eye_dist: float  # Left brow to eye distance (normalized by face height)
    right_brow_eye_dist: float  # Right brow to eye distance (normalized by face height)
    left_eye_aspect_ratio: float  # Left eye openness
    right_eye_aspect_ratio: float  # Right eye openness
    face_width: float  # Inter-eye distance in pixels (for scale reference)
    face_height: float  # Forehead to chin distance in pixels
    nose_x_normalized: float = 0.5  # Nose X position as 0-1 fraction of frame width
    face_center_x_normalized: float = 0.5  # Face center X as 0-1 fraction of frame width
    landmarks_valid: bool = True


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _get_point(landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
    """Get landmark pixel coordinates."""
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def _get_point_normalized(landmarks, idx: int) -> Tuple[float, float]:
    """Get landmark in normalized 0..1 coordinates."""
    lm = landmarks[idx]
    return (lm.x, lm.y)


def estimate_yaw_solvepnp(
    landmarks, img_w: int, img_h: int, mirrored: bool = True
) -> Optional[float]:
    """Estimate head yaw using cv2.solvePnP.

    Returns yaw in degrees or None if solvePnP fails.

    On a horizontally-flipped (mirrored) frame, the sign is already correct:
    negative = physical left, positive = physical right. No additional negation
    needed because solvePnP computes geometry on the already-flipped image where
    the player's physical left turn moves their nose to the left side of the frame,
    producing a negative yaw naturally.

    Args:
        landmarks: MediaPipe face landmarks
        img_w: Frame width in pixels
        img_h: Frame height in pixels
        mirrored: Kept for API compatibility. No negation is applied regardless
                  of this value since the mirrored frame already produces correct signs.
    """
    image_points = np.array([
        _get_point(landmarks, idx, img_w, img_h)
        for idx in SOLVEPNP_LANDMARKS
    ], dtype=np.float64)

    # Camera matrix approximation
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose rotation matrix to Euler angles
    # We use the projection matrix approach
    proj_matrix = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        np.vstack((proj_matrix, [0, 0, 0, 1]))[:3, :]
    )

    yaw = euler_angles[1, 0]

    # Clamp to reasonable range
    if abs(yaw) > 90:
        yaw = max(-90.0, min(90.0, yaw))

    return float(yaw)


def estimate_yaw_nose_offset(
    landmarks, img_w: int, img_h: int, mirrored: bool = True
) -> float:
    """Fallback yaw estimation using nose offset from face center.

    Returns yaw in degrees (approximate).

    On a horizontally-flipped (mirrored) frame, the sign is already correct:
    negative = physical left, positive = physical right. When the player turns
    left physically, on the mirrored image their nose moves LEFT, producing a
    negative offset which maps to negative degrees. No additional negation needed.

    Args:
        landmarks: MediaPipe face landmarks
        img_w: Frame width in pixels
        img_h: Frame height in pixels
        mirrored: Kept for API compatibility. No negation is applied regardless
                  of this value since the mirrored frame already produces correct signs.
    """
    nose = _get_point(landmarks, NOSE_TIP, img_w, img_h)
    left_eye = _get_point(landmarks, LEFT_EYE_OUTER, img_w, img_h)
    right_eye = _get_point(landmarks, RIGHT_EYE_OUTER, img_w, img_h)

    face_center_x = (left_eye[0] + right_eye[0]) / 2
    face_width = abs(left_eye[0] - right_eye[0])

    if face_width < 1:
        return 0.0

    # Nose offset as fraction of face width
    offset_ratio = (nose[0] - face_center_x) / face_width

    # Convert to approximate degrees (empirical mapping)
    # offset_ratio of ~0.3 corresponds to roughly 30-40 degrees
    yaw = offset_ratio * 90.0

    yaw = max(-90.0, min(90.0, yaw))

    return yaw


def extract_features(
    landmarks, img_w: int, img_h: int, mirrored: bool = True
) -> FaceFeatures:
    """Extract all face features from MediaPipe Face Mesh landmarks.

    All expression ratios are normalized by face dimensions for scale invariance.

    On a horizontally-flipped (mirrored) frame, yaw sign is already correct:
    negative = physical left, positive = physical right.

    Args:
        landmarks: MediaPipe face landmarks
        img_w: Frame width in pixels
        img_h: Frame height in pixels
        mirrored: Kept for API compatibility (passed to yaw estimators).
    """
    # Face dimensions for normalization
    left_eye_outer = _get_point(landmarks, LEFT_EYE_OUTER, img_w, img_h)
    right_eye_outer = _get_point(landmarks, RIGHT_EYE_OUTER, img_w, img_h)
    forehead = _get_point(landmarks, FOREHEAD, img_w, img_h)
    chin = _get_point(landmarks, CHIN, img_w, img_h)

    face_width = _distance(left_eye_outer, right_eye_outer)
    face_height = _distance(forehead, chin)

    # Nose and face center for position validation
    nose_point = _get_point(landmarks, NOSE_TIP, img_w, img_h)
    face_center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2.0

    nose_x_normalized = nose_point[0] / max(img_w, 1)
    face_center_x_normalized = face_center_x / max(img_w, 1)

    if face_width < 1 or face_height < 1:
        return FaceFeatures(
            yaw_degrees=0.0,
            mouth_aspect_ratio=0.0,
            mouth_corner_lift=0.0,
            left_brow_eye_dist=0.0,
            right_brow_eye_dist=0.0,
            left_eye_aspect_ratio=0.0,
            right_eye_aspect_ratio=0.0,
            face_width=face_width,
            face_height=face_height,
            nose_x_normalized=nose_x_normalized,
            face_center_x_normalized=face_center_x_normalized,
            landmarks_valid=False,
        )

    # --- Yaw estimation ---
    yaw = estimate_yaw_solvepnp(landmarks, img_w, img_h, mirrored=mirrored)
    if yaw is None:
        yaw = estimate_yaw_nose_offset(landmarks, img_w, img_h, mirrored=mirrored)

    # --- Mouth aspect ratio ---
    upper_lip = _get_point(landmarks, UPPER_LIP_TOP, img_w, img_h)
    lower_lip = _get_point(landmarks, LOWER_LIP_BOTTOM, img_w, img_h)
    mouth_left = _get_point(landmarks, MOUTH_LEFT, img_w, img_h)
    mouth_right = _get_point(landmarks, MOUTH_RIGHT, img_w, img_h)

    mouth_height = _distance(upper_lip, lower_lip)
    mouth_width = _distance(mouth_left, mouth_right)

    mouth_aspect_ratio = mouth_height / max(mouth_width, 1.0)
    # Normalize by face height
    mouth_aspect_ratio_norm = mouth_aspect_ratio

    # --- Mouth corner lift ---
    # Measure how much corners are lifted relative to mouth center
    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    avg_corner_y = (mouth_left[1] + mouth_right[1]) / 2
    # Negative means corners above center (smiling)
    mouth_corner_lift = (mouth_center_y - avg_corner_y) / face_height

    # --- Brow to eye distance ---
    left_brow_mid = _midpoint(
        _get_point(landmarks, LEFT_BROW_INNER, img_w, img_h),
        _get_point(landmarks, LEFT_BROW_OUTER, img_w, img_h),
    )
    right_brow_mid = _midpoint(
        _get_point(landmarks, RIGHT_BROW_INNER, img_w, img_h),
        _get_point(landmarks, RIGHT_BROW_OUTER, img_w, img_h),
    )
    left_eye_top = _get_point(landmarks, LEFT_EYE_TOP, img_w, img_h)
    right_eye_top = _get_point(landmarks, RIGHT_EYE_TOP, img_w, img_h)

    left_brow_eye_dist = (left_eye_top[1] - left_brow_mid[1]) / face_height
    right_brow_eye_dist = (right_eye_top[1] - right_brow_mid[1]) / face_height

    # --- Eye aspect ratio ---
    # Left eye
    left_eye_l = _get_point(landmarks, LEFT_EYE_LEFT, img_w, img_h)
    left_eye_r = _get_point(landmarks, LEFT_EYE_RIGHT, img_w, img_h)
    left_eye_upper = _midpoint(
        _get_point(landmarks, LEFT_EYE_UPPER[0], img_w, img_h),
        _get_point(landmarks, LEFT_EYE_UPPER[1], img_w, img_h),
    )
    left_eye_lower = _midpoint(
        _get_point(landmarks, LEFT_EYE_LOWER[0], img_w, img_h),
        _get_point(landmarks, LEFT_EYE_LOWER[1], img_w, img_h),
    )
    left_eye_h = _distance(left_eye_upper, left_eye_lower)
    left_eye_w = _distance(left_eye_l, left_eye_r)
    left_eye_ar = left_eye_h / max(left_eye_w, 1.0)

    # Right eye
    right_eye_l = _get_point(landmarks, RIGHT_EYE_LEFT, img_w, img_h)
    right_eye_r = _get_point(landmarks, RIGHT_EYE_RIGHT, img_w, img_h)
    right_eye_upper = _midpoint(
        _get_point(landmarks, RIGHT_EYE_UPPER[0], img_w, img_h),
        _get_point(landmarks, RIGHT_EYE_UPPER[1], img_w, img_h),
    )
    right_eye_lower = _midpoint(
        _get_point(landmarks, RIGHT_EYE_LOWER[0], img_w, img_h),
        _get_point(landmarks, RIGHT_EYE_LOWER[1], img_w, img_h),
    )
    right_eye_h = _distance(right_eye_upper, right_eye_lower)
    right_eye_w = _distance(right_eye_l, right_eye_r)
    right_eye_ar = right_eye_h / max(right_eye_w, 1.0)

    return FaceFeatures(
        yaw_degrees=yaw,
        mouth_aspect_ratio=mouth_aspect_ratio_norm,
        mouth_corner_lift=mouth_corner_lift,
        left_brow_eye_dist=left_brow_eye_dist,
        right_brow_eye_dist=right_brow_eye_dist,
        left_eye_aspect_ratio=left_eye_ar,
        right_eye_aspect_ratio=right_eye_ar,
        face_width=face_width,
        face_height=face_height,
        nose_x_normalized=nose_x_normalized,
        face_center_x_normalized=face_center_x_normalized,
        landmarks_valid=True,
    )


class FaceTracker:
    """Manages MediaPipe Face Mesh and extracts features each frame."""

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(
        self, bgr_frame: np.ndarray, mirrored: bool = True
    ) -> Optional[FaceFeatures]:
        """Process a BGR frame and return face features, or None if no face found.

        Args:
            bgr_frame: BGR image from webcam
            mirrored: If True, the frame has been horizontally flipped (mirror mode).
                      On a mirrored frame, yaw sign is already correct without negation:
                      negative = physical left, positive = physical right.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = bgr_frame.shape[:2]

        return extract_features(landmarks, w, h, mirrored=mirrored)

    def release(self):
        """Release MediaPipe resources."""
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
