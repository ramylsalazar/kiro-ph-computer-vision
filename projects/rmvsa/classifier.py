"""Expression classification and yaw bucketing.

Pure Python module - testable without OpenCV, MediaPipe, or a webcam.
Uses a neutral baseline for relative expression detection.
Applies foreshortening correction using current frame's raw yaw.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict

from config import DifficultyConfig, YawConfig, ExpressionThresholds, CalibrationConfig


class CalibrationStatus(Enum):
    """Status of the calibration process."""

    IDLE = auto()
    COLLECTING = auto()
    WAITING_FOR_FACE = auto()
    FACE_TOO_SMALL = auto()
    FACE_TOO_LARGE = auto()
    FACE_OFF_CENTER = auto()
    FACE_NOT_FORWARD = auto()
    NOT_STILL = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class Baseline:
    """Neutral face baseline captured during calibration."""

    mouth_aspect_ratio: float = 0.0
    mouth_corner_lift: float = 0.0
    avg_brow_eye_dist: float = 0.0
    avg_eye_aspect_ratio: float = 0.0
    sample_count: int = 0

    def is_valid(self) -> bool:
        """Return True if enough samples were collected."""
        return self.sample_count >= 5


@dataclass
class ClassifierResult:
    """Output of the classifier for a single frame."""

    yaw_bucket: str  # "left", "forward", "right"
    expression: str  # "neutral", "happy", "surprised", "angry"
    confirmed_yaw: str  # Yaw bucket after hysteresis
    confirmed_expression: str  # Expression after persistence filter
    raw_yaw_degrees: float = 0.0
    smoothed_yaw_degrees: float = 0.0


class YawBucketer:
    """Buckets yaw into left/forward/right with hysteresis to prevent flicker."""

    def __init__(self, config: YawConfig):
        self._config = config
        self._current_bucket = "forward"

    @property
    def current_bucket(self) -> str:
        return self._current_bucket

    def update(self, yaw_degrees: float) -> str:
        """Update yaw bucket with hysteresis.

        Uses separate enter and release thresholds.
        """
        if self._current_bucket == "forward":
            # Need to exceed enter threshold to leave forward
            if yaw_degrees <= self._config.enter_left:
                self._current_bucket = "left"
            elif yaw_degrees >= self._config.enter_right:
                self._current_bucket = "right"
        elif self._current_bucket == "left":
            # Need to come back past release threshold to return to forward
            if yaw_degrees >= self._config.release_left:
                if yaw_degrees >= self._config.enter_right:
                    self._current_bucket = "right"
                else:
                    self._current_bucket = "forward"
        elif self._current_bucket == "right":
            # Need to come back past release threshold to return to forward
            if yaw_degrees <= self._config.release_right:
                if yaw_degrees <= self._config.enter_left:
                    self._current_bucket = "left"
                else:
                    self._current_bucket = "forward"

        return self._current_bucket

    def reset(self):
        """Reset to forward."""
        self._current_bucket = "forward"


class ExpressionPersistence:
    """Requires an expression to be detected stably before accepting it.

    Prevents brief noisy classifications from triggering state changes.
    """

    def __init__(self, persistence_time: float):
        self._persistence_time = persistence_time
        self._current_expression = "neutral"
        self._candidate_expression = "neutral"
        self._candidate_start: Optional[float] = None

    @property
    def current_expression(self) -> str:
        return self._current_expression

    def update(self, raw_expression: str, timestamp: float) -> str:
        """Update with a new raw classification and return the confirmed expression."""
        if raw_expression == self._current_expression:
            # Already confirmed, reset candidate
            self._candidate_expression = raw_expression
            self._candidate_start = None
            return self._current_expression

        if raw_expression == self._candidate_expression:
            # Same candidate continuing
            if self._candidate_start is not None:
                elapsed = timestamp - self._candidate_start
                if elapsed >= self._persistence_time:
                    # Confirmed new expression
                    self._current_expression = raw_expression
                    self._candidate_start = None
        else:
            # New candidate
            self._candidate_expression = raw_expression
            self._candidate_start = timestamp

        return self._current_expression

    def reset(self):
        """Reset to neutral."""
        self._current_expression = "neutral"
        self._candidate_expression = "neutral"
        self._candidate_start = None


def apply_foreshortening_correction(
    mouth_ar: float,
    brow_eye_dist: float,
    eye_ar: float,
    yaw_degrees: float,
    max_yaw: float = 55.0,
) -> Tuple[float, float, float]:
    """Correct vertical-to-horizontal ratios for yaw foreshortening.

    When the head turns, horizontal measurements shrink due to perspective,
    making vertical/horizontal ratios appear larger. We multiply by cos(yaw)
    to compensate.

    Uses the CURRENT FRAME'S UNSMOOTHED yaw, not delayed or smoothed.
    Clamps at extreme angles to avoid numerical instability.
    """
    # Clamp yaw for correction to avoid extreme values
    clamped_yaw = max(-max_yaw, min(max_yaw, yaw_degrees))
    correction = math.cos(math.radians(clamped_yaw))

    # Apply correction to ratios that compare vertical to horizontal
    corrected_mouth_ar = mouth_ar * correction
    corrected_brow_eye = brow_eye_dist * correction
    corrected_eye_ar = eye_ar * correction

    return corrected_mouth_ar, corrected_brow_eye, corrected_eye_ar


def classify_expression_raw(
    mouth_ar: float,
    mouth_corner_lift: float,
    avg_brow_eye_dist: float,
    avg_eye_ar: float,
    baseline: Baseline,
    thresholds: ExpressionThresholds,
) -> str:
    """Classify expression relative to baseline.

    Returns one of: "neutral", "happy", "surprised", "angry".
    All comparisons are relative to the player's calibrated neutral baseline.
    """
    if not baseline.is_valid():
        return "neutral"

    # Compute ratios relative to baseline (avoid division by zero)
    def safe_ratio(current: float, base: float) -> float:
        if abs(base) < 1e-6:
            return 1.0 if abs(current) < 1e-6 else 2.0
        return current / base

    mar_ratio = safe_ratio(mouth_ar, baseline.mouth_aspect_ratio)
    corner_ratio = safe_ratio(mouth_corner_lift, baseline.mouth_corner_lift)
    brow_ratio = safe_ratio(avg_brow_eye_dist, baseline.avg_brow_eye_dist)
    eye_ratio = safe_ratio(avg_eye_ar, baseline.avg_eye_aspect_ratio)

    # Check surprised first (most distinctive: wide mouth + raised brows + wide eyes)
    if (mar_ratio >= thresholds.surprised_mouth_aspect_ratio
            and brow_ratio >= thresholds.surprised_brow_raise
            and eye_ratio >= thresholds.surprised_eye_aspect_ratio):
        return "surprised"

    # Check happy (mouth corners up + mouth slightly open)
    if (corner_ratio >= thresholds.happy_mouth_corner_lift
            and mar_ratio >= thresholds.happy_mouth_aspect_ratio):
        return "happy"

    # Check angry (brows lowered + eyes narrowed + corners down)
    if (brow_ratio <= thresholds.angry_brow_lower
            and eye_ratio <= thresholds.angry_eye_squint):
        return "angry"

    # Also check angry with mouth corners down
    if (brow_ratio <= thresholds.angry_brow_lower
            and corner_ratio <= thresholds.angry_mouth_corner_lift):
        return "angry"

    return "neutral"


@dataclass
class FaceMeasurements:
    """Simplified face measurements for the classifier (decoupled from tracking)."""

    yaw_degrees: float
    mouth_aspect_ratio: float
    mouth_corner_lift: float
    left_brow_eye_dist: float
    right_brow_eye_dist: float
    left_eye_aspect_ratio: float
    right_eye_aspect_ratio: float


class Classifier:
    """Full expression + yaw classifier with calibration, hysteresis, and persistence.

    Pure Python - no OpenCV or MediaPipe dependency.
    """

    def __init__(self, config: DifficultyConfig):
        self._config = config
        self._yaw_bucketer = YawBucketer(config.yaw)
        self._expression_persistence = ExpressionPersistence(config.expression_persistence)
        self._baseline = Baseline()
        self._calibrating = False
        self._calibration_start: Optional[float] = None
        self._calibration_samples: List[Tuple[float, float, float, float]] = []
        self._calibration_yaw_samples: List[float] = []
        self._valid_sample_count: int = 0
        self._calibration_status: CalibrationStatus = CalibrationStatus.IDLE

        # Yaw smoothing (exponential moving average)
        self._smoothed_yaw: float = 0.0
        self._smoothing_alpha: float = config.yaw_smoothing_alpha
        self._smoothed_yaw_initialized: bool = False

        # Last raw expression for debug
        self._last_raw_expression: str = "neutral"
        self._last_raw_yaw: float = 0.0
        self._is_face_valid: bool = False

    @property
    def baseline(self) -> Baseline:
        return self._baseline

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def is_calibrated(self) -> bool:
        return self._baseline.is_valid()

    @property
    def calibration_progress(self) -> float:
        """Return calibration progress from 0.0 to 1.0."""
        if not self._calibrating:
            if self._calibration_status == CalibrationStatus.COMPLETE:
                return 1.0
            return 0.0
        min_samples = self._config.calibration.min_valid_samples
        if min_samples <= 0:
            return 1.0
        return min(1.0, self._valid_sample_count / min_samples)

    @property
    def calibration_status(self) -> CalibrationStatus:
        """Return current calibration status."""
        return self._calibration_status

    @property
    def valid_sample_count(self) -> int:
        """Return number of valid calibration samples collected."""
        return self._valid_sample_count

    @property
    def debug_info(self) -> Dict:
        """Return debug information about current classifier state."""
        return {
            "raw_yaw": self._last_raw_yaw,
            "smoothed_yaw": self._smoothed_yaw,
            "confirmed_direction": self._yaw_bucketer.current_bucket,
            "raw_expression": self._last_raw_expression,
            "confirmed_expression": self._expression_persistence.current_expression,
            "baseline_mouth_ar": self._baseline.mouth_aspect_ratio,
            "baseline_corner_lift": self._baseline.mouth_corner_lift,
            "baseline_brow_eye": self._baseline.avg_brow_eye_dist,
            "baseline_eye_ar": self._baseline.avg_eye_aspect_ratio,
            "baseline_samples": self._baseline.sample_count,
            "calibration_status": self._calibration_status.name,
            "calibration_progress": self.calibration_progress,
            "is_face_valid": self._is_face_valid,
        }

    def set_no_face(self):
        """Signal that no face was detected this frame.

        Updates internal state to reflect missing face. During calibration,
        sets status to WAITING_FOR_FACE so the UI can inform the player.
        """
        self._is_face_valid = False
        if self._calibrating:
            self._calibration_status = CalibrationStatus.WAITING_FOR_FACE

    def start_calibration(self, timestamp: float):
        """Begin calibration. Player should hold a neutral face."""
        self._calibrating = True
        self._calibration_start = timestamp
        self._calibration_samples = []
        self._calibration_yaw_samples = []
        self._valid_sample_count = 0
        self._baseline = Baseline()
        self._calibration_status = CalibrationStatus.COLLECTING
        self._expression_persistence.reset()
        self._yaw_bucketer.reset()
        self._smoothed_yaw = 0.0
        self._smoothed_yaw_initialized = False

    def update(
        self,
        measurements: FaceMeasurements,
        timestamp: float,
        face_width: Optional[float] = None,
        face_center_x: Optional[float] = None,
        frame_width: Optional[float] = None,
    ) -> ClassifierResult:
        """Process measurements and return classification result.

        Uses the current frame's RAW yaw for foreshortening correction (not smoothed).
        The SMOOTHED yaw is used for bucket classification.

        Args:
            measurements: Face measurements from tracker
            timestamp: Current time
            face_width: Face width in pixels (for calibration validation)
            face_center_x: Face center X normalized 0-1 (for calibration validation)
            frame_width: Frame width in pixels (for calibration validation)
        """
        raw_yaw = measurements.yaw_degrees
        self._last_raw_yaw = raw_yaw

        # Update smoothed yaw using exponential moving average
        # Only apply smoothing when NOT calibrating - during calibration the
        # smoothed value isn't used and would drag toward 0 on gameplay start
        if not self._calibrating:
            if not self._smoothed_yaw_initialized:
                self._smoothed_yaw = raw_yaw
                self._smoothed_yaw_initialized = True
            else:
                alpha = self._smoothing_alpha
                self._smoothed_yaw = alpha * raw_yaw + (1.0 - alpha) * self._smoothed_yaw

        # Validate face for calibration purposes
        self._is_face_valid = self._validate_face(
            raw_yaw, face_width, face_center_x, frame_width
        )

        # Average brow and eye measurements
        avg_brow = (measurements.left_brow_eye_dist + measurements.right_brow_eye_dist) / 2
        avg_eye = (measurements.left_eye_aspect_ratio + measurements.right_eye_aspect_ratio) / 2

        # Apply foreshortening correction using CURRENT FRAME'S unsmoothed yaw
        corrected_mar, corrected_brow, corrected_eye = apply_foreshortening_correction(
            measurements.mouth_aspect_ratio,
            avg_brow,
            avg_eye,
            raw_yaw,
            self._config.max_yaw_for_correction,
        )

        # Handle calibration
        if self._calibrating:
            was_calibrating = True
            self._handle_calibration(
                corrected_mar,
                measurements.mouth_corner_lift,
                corrected_brow,
                corrected_eye,
                timestamp,
                raw_yaw,
                face_width,
                face_center_x,
                frame_width,
            )
        else:
            was_calibrating = False

        # If calibration just completed on this frame, reset smoothing
        # so next frame's yaw initializes fresh without EMA drag from cal frames
        if was_calibrating and not self._calibrating:
            self._smoothed_yaw_initialized = False

        # Yaw bucketing uses SMOOTHED yaw (with hysteresis)
        yaw_bucket = self._yaw_bucketer.update(self._smoothed_yaw)

        # Expression classification (relative to baseline)
        raw_expression = classify_expression_raw(
            corrected_mar,
            measurements.mouth_corner_lift,
            corrected_brow,
            corrected_eye,
            self._baseline,
            self._config.expression,
        )
        self._last_raw_expression = raw_expression

        # Expression persistence filter
        confirmed_expression = self._expression_persistence.update(raw_expression, timestamp)

        return ClassifierResult(
            yaw_bucket=yaw_bucket,
            expression=raw_expression,
            confirmed_yaw=yaw_bucket,
            confirmed_expression=confirmed_expression,
            raw_yaw_degrees=raw_yaw,
            smoothed_yaw_degrees=self._smoothed_yaw,
        )

    def _validate_face(
        self,
        yaw: float,
        face_width: Optional[float],
        face_center_x: Optional[float],
        frame_width: Optional[float],
    ) -> bool:
        """Validate face meets calibration requirements. Returns True if valid."""
        cal = self._config.calibration

        # If no face metrics provided, assume valid (backwards compatibility)
        if face_width is None:
            return True

        # Face size check
        if face_width < cal.min_face_width:
            if self._calibrating:
                self._calibration_status = CalibrationStatus.FACE_TOO_SMALL
            return False

        if face_width > cal.max_face_width:
            if self._calibrating:
                self._calibration_status = CalibrationStatus.FACE_TOO_LARGE
            return False

        # Face centering check
        if face_center_x is not None:
            offset = abs(face_center_x - 0.5)
            if offset > cal.max_face_offset_ratio:
                if self._calibrating:
                    self._calibration_status = CalibrationStatus.FACE_OFF_CENTER
                return False

        # Yaw check (face should be roughly forward for calibration)
        if self._calibrating and abs(yaw) > cal.max_yaw_during_cal:
            self._calibration_status = CalibrationStatus.FACE_NOT_FORWARD
            return False

        return True

    def _handle_calibration(
        self,
        mouth_ar: float,
        corner_lift: float,
        brow_dist: float,
        eye_ar: float,
        timestamp: float,
        raw_yaw: float,
        face_width: Optional[float] = None,
        face_center_x: Optional[float] = None,
        frame_width: Optional[float] = None,
    ):
        """Accumulate calibration samples and finalize when enough are collected."""
        if self._calibration_start is None:
            return

        cal = self._config.calibration

        # Validate this frame for calibration
        frame_valid = self._validate_face(raw_yaw, face_width, face_center_x, frame_width)

        if frame_valid:
            # Accept this sample
            self._calibration_samples.append((mouth_ar, corner_lift, brow_dist, eye_ar))
            self._calibration_yaw_samples.append(raw_yaw)
            self._valid_sample_count = len(self._calibration_samples)
            self._calibration_status = CalibrationStatus.COLLECTING

        # Check if we have enough valid samples
        if self._valid_sample_count >= cal.min_valid_samples:
            # Stillness check: std dev of yaw during calibration
            if len(self._calibration_yaw_samples) >= 2:
                yaw_values = self._calibration_yaw_samples
                mean_yaw = sum(yaw_values) / len(yaw_values)
                variance = sum((y - mean_yaw) ** 2 for y in yaw_values) / len(yaw_values)
                std_dev = math.sqrt(variance)

                if std_dev > cal.stillness_threshold:
                    # Too much movement - reject samples and restart collection
                    self._calibration_status = CalibrationStatus.NOT_STILL
                    # Keep only the most recent half of samples
                    half = len(self._calibration_samples) // 2
                    self._calibration_samples = self._calibration_samples[half:]
                    self._calibration_yaw_samples = self._calibration_yaw_samples[half:]
                    self._valid_sample_count = len(self._calibration_samples)
                    return

            # All checks passed - finalize
            self._finalize_calibration()
            return

        # Timeout: if calibration takes too long without enough samples, fail
        elapsed = timestamp - self._calibration_start
        if elapsed > self._config.calibration_duration * 10:
            # Extended timeout - try to finalize with what we have
            if self._valid_sample_count >= 5:
                self._finalize_calibration()
            else:
                self._calibrating = False
                self._calibration_status = CalibrationStatus.FAILED

    def _finalize_calibration(self):
        """Compute baseline from collected samples."""
        if not self._calibration_samples:
            self._calibrating = False
            self._calibration_status = CalibrationStatus.FAILED
            return

        n = len(self._calibration_samples)
        sum_mar = sum(s[0] for s in self._calibration_samples)
        sum_corner = sum(s[1] for s in self._calibration_samples)
        sum_brow = sum(s[2] for s in self._calibration_samples)
        sum_eye = sum(s[3] for s in self._calibration_samples)

        self._baseline = Baseline(
            mouth_aspect_ratio=sum_mar / n,
            mouth_corner_lift=sum_corner / n,
            avg_brow_eye_dist=sum_brow / n,
            avg_eye_aspect_ratio=sum_eye / n,
            sample_count=n,
        )
        self._calibrating = False
        self._calibration_status = CalibrationStatus.COMPLETE

    def reset(self):
        """Reset all state."""
        self._yaw_bucketer.reset()
        self._expression_persistence.reset()
        self._baseline = Baseline()
        self._calibrating = False
        self._calibration_start = None
        self._calibration_samples = []
        self._calibration_yaw_samples = []
        self._valid_sample_count = 0
        self._calibration_status = CalibrationStatus.IDLE
        self._smoothed_yaw = 0.0
        self._smoothed_yaw_initialized = False
        self._last_raw_expression = "neutral"
        self._last_raw_yaw = 0.0
        self._is_face_valid = False
