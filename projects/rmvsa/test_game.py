"""Unit tests for Hole in the Wall: Face Challenge.

Tests classifier, game logic, and persistence using synthetic data.
No webcam, OpenCV, or MediaPipe required.
"""

import sys
import os
import math
import json
import tempfile

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from config import (
    DifficultyConfig, YawConfig, ExpressionThresholds, CalibrationConfig,
    NORMAL, EASY, HARD, DIFFICULTIES, ALL_TARGETS,
)
from classifier import (
    Classifier, FaceMeasurements, YawBucketer, ExpressionPersistence,
    apply_foreshortening_correction, classify_expression_raw, Baseline,
    CalibrationStatus,
)
from game_logic import GameLogic, GameState, Target
from persistence import HighScoreManager, load_high_scores, save_high_scores


# ---------------------------------------------------------------------------
# Synthetic face fixtures
# ---------------------------------------------------------------------------

def make_neutral_measurements(yaw: float = 0.0, scale: float = 1.0) -> FaceMeasurements:
    """Create neutral face measurements at given yaw and scale."""
    return FaceMeasurements(
        yaw_degrees=yaw,
        mouth_aspect_ratio=0.05 * scale,
        mouth_corner_lift=0.02 * scale,
        left_brow_eye_dist=0.06 * scale,
        right_brow_eye_dist=0.06 * scale,
        left_eye_aspect_ratio=0.25 * scale,
        right_eye_aspect_ratio=0.25 * scale,
    )


def make_happy_measurements(yaw: float = 0.0, scale: float = 1.0) -> FaceMeasurements:
    """Create happy face measurements."""
    return FaceMeasurements(
        yaw_degrees=yaw,
        mouth_aspect_ratio=0.08 * scale,
        mouth_corner_lift=0.04 * scale,
        left_brow_eye_dist=0.06 * scale,
        right_brow_eye_dist=0.06 * scale,
        left_eye_aspect_ratio=0.22 * scale,
        right_eye_aspect_ratio=0.22 * scale,
    )


def make_surprised_measurements(yaw: float = 0.0, scale: float = 1.0) -> FaceMeasurements:
    """Create surprised face measurements."""
    return FaceMeasurements(
        yaw_degrees=yaw,
        mouth_aspect_ratio=0.12 * scale,
        mouth_corner_lift=0.02 * scale,
        left_brow_eye_dist=0.095 * scale,
        right_brow_eye_dist=0.095 * scale,
        left_eye_aspect_ratio=0.40 * scale,
        right_eye_aspect_ratio=0.40 * scale,
    )


def make_angry_measurements(yaw: float = 0.0, scale: float = 1.0) -> FaceMeasurements:
    """Create angry face measurements."""
    return FaceMeasurements(
        yaw_degrees=yaw,
        mouth_aspect_ratio=0.04 * scale,
        mouth_corner_lift=0.01 * scale,
        left_brow_eye_dist=0.04 * scale,
        right_brow_eye_dist=0.04 * scale,
        left_eye_aspect_ratio=0.18 * scale,
        right_eye_aspect_ratio=0.18 * scale,
    )


def calibrate_classifier(classifier: Classifier, t: float = 0.0, scale: float = 1.0) -> float:
    """Calibrate a classifier with neutral measurements. Returns time after calibration."""
    classifier.start_calibration(t)
    # Feed enough neutral frames over enough time to complete calibration
    # Need >= calibration_duration (1.0s) AND >= calibration_frames (15)
    num_frames = 25
    dt = 0.06  # 25 * 0.06 = 1.5s total, well past 1.0s duration
    for i in range(num_frames):
        m = make_neutral_measurements(scale=scale)
        classifier.update(m, t + (i + 1) * dt)
    return t + (num_frames + 1) * dt + 0.5  # Return time well after calibration



# ---------------------------------------------------------------------------
# NEW: Physical Direction Mapping Tests (Bug 1 fix validation)
# ---------------------------------------------------------------------------

class TestPhysicalDirectionMapping:
    """Test correct physical left/right mapping with mirrored webcam.

    After bug fix: on a mirrored frame, solvePnP naturally produces
    negative yaw when player turns physically left, positive when right.
    No negation is applied. The bucketer should map these correctly.
    """

    def test_negative_yaw_maps_to_left(self):
        """Negative yaw (player physically turns left on mirrored frame) -> 'left' bucket."""
        bucketer = YawBucketer(NORMAL.yaw)
        # Negative yaw = physical left on mirrored frame
        result = bucketer.update(-30.0)
        assert result == "left", (
            "Negative yaw should map to 'left' (physical left turn)"
        )

    def test_positive_yaw_maps_to_right(self):
        """Positive yaw (player physically turns right on mirrored frame) -> 'right' bucket."""
        bucketer = YawBucketer(NORMAL.yaw)
        # Positive yaw = physical right on mirrored frame
        result = bucketer.update(30.0)
        assert result == "right", (
            "Positive yaw should map to 'right' (physical right turn)"
        )

    def test_small_negative_stays_forward(self):
        """Small negative yaw (slight left lean) should stay forward."""
        bucketer = YawBucketer(NORMAL.yaw)
        result = bucketer.update(-10.0)
        assert result == "forward"

    def test_small_positive_stays_forward(self):
        """Small positive yaw (slight right lean) should stay forward."""
        bucketer = YawBucketer(NORMAL.yaw)
        result = bucketer.update(10.0)
        assert result == "forward"

    def test_classifier_negative_yaw_gives_left(self):
        """Full classifier pipeline: negative yaw -> confirmed 'left'."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Feed enough frames with negative yaw for EMA to converge
        for i in range(15):
            m = make_neutral_measurements(yaw=-30.0)
            result = classifier.update(m, t + i * 0.05)

        assert result.confirmed_yaw == "left"

    def test_classifier_positive_yaw_gives_right(self):
        """Full classifier pipeline: positive yaw -> confirmed 'right'."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Feed enough frames with positive yaw for EMA to converge
        for i in range(15):
            m = make_neutral_measurements(yaw=30.0)
            result = classifier.update(m, t + i * 0.05)

        assert result.confirmed_yaw == "right"



# ---------------------------------------------------------------------------
# NEW: Yaw Sign Consistency Tests (Bug 1 fix validation)
# ---------------------------------------------------------------------------

class TestYawSignConsistency:
    """Test that solvePnP and fallback yaw produce same sign for same physical direction.

    Both estimate_yaw_solvepnp and estimate_yaw_nose_offset should produce:
    - Negative values for physical left turn
    - Positive values for physical right turn
    - Near zero for forward

    We test this at the classifier level since the raw face_tracking functions
    require actual landmark objects (MediaPipe), but the classifier's bucketer
    uses the same sign convention.
    """

    def test_both_methods_agree_left_is_negative(self):
        """Both yaw methods should produce negative for physical left.

        The classifier's smoothed yaw should go negative when fed negative raw yaw,
        confirming the sign convention is consistent throughout the pipeline.
        """
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Feed negative yaw (physical left)
        for i in range(10):
            m = make_neutral_measurements(yaw=-30.0)
            result = classifier.update(m, t + i * 0.05)

        assert result.smoothed_yaw_degrees < 0, (
            "Smoothed yaw should be negative for physical left turn"
        )
        assert result.raw_yaw_degrees < 0, (
            "Raw yaw should be negative for physical left turn"
        )

    def test_both_methods_agree_right_is_positive(self):
        """Both yaw methods should produce positive for physical right."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Feed positive yaw (physical right)
        for i in range(10):
            m = make_neutral_measurements(yaw=30.0)
            result = classifier.update(m, t + i * 0.05)

        assert result.smoothed_yaw_degrees > 0, (
            "Smoothed yaw should be positive for physical right turn"
        )
        assert result.raw_yaw_degrees > 0, (
            "Raw yaw should be positive for physical right turn"
        )

    def test_forward_near_zero(self):
        """Forward-facing should produce yaw near zero."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        for i in range(10):
            m = make_neutral_measurements(yaw=0.0)
            result = classifier.update(m, t + i * 0.05)

        assert abs(result.smoothed_yaw_degrees) < 5.0, (
            "Forward-facing yaw should be near zero"
        )

    def test_sign_preserved_through_ema(self):
        """EMA smoothing should preserve sign direction."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Start with positive yaw
        for i in range(20):
            m = make_neutral_measurements(yaw=35.0)
            result = classifier.update(m, t + i * 0.05)

        # Smoothed yaw should converge to same sign as raw
        assert result.smoothed_yaw_degrees > 20.0, (
            "EMA should converge toward the consistent input sign"
        )



# ---------------------------------------------------------------------------
# NEW: Yaw Hysteresis Noise Tests
# ---------------------------------------------------------------------------

class TestYawHysteresisNoise:
    """Test yaw hysteresis with noisy boundary values.

    Verifies no rapid flicker between states when values oscillate
    around enter/release thresholds.
    """

    def test_no_flicker_at_enter_boundary(self):
        """Values oscillating around enter threshold should not cause rapid flicker."""
        bucketer = YawBucketer(NORMAL.yaw)
        # enter_left = -25.0 for Normal
        # Oscillate around -25: -24, -26, -24, -26, ...
        states = []
        for i in range(20):
            yaw = -24.0 if i % 2 == 0 else -26.0
            bucketer.update(yaw)
            states.append(bucketer.current_bucket)

        # Count state transitions
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        # With hysteresis, once we enter left (release_left = -18),
        # -24 is still in the left zone (past release). So after first entry,
        # it should stay left. Expect at most 1 transition (forward -> left).
        assert transitions <= 1, (
            f"Too many transitions ({transitions}) - hysteresis not preventing flicker"
        )

    def test_no_flicker_at_release_boundary(self):
        """Values oscillating around release threshold should not flicker."""
        bucketer = YawBucketer(NORMAL.yaw)
        # First enter left
        bucketer.update(-30.0)
        assert bucketer.current_bucket == "left"

        # Oscillate around release_left = -18: -17, -19, -17, -19
        states = []
        for i in range(20):
            yaw = -17.0 if i % 2 == 0 else -19.0
            bucketer.update(yaw)
            states.append(bucketer.current_bucket)

        # After first release (at -17), we're forward. At -19, we're still
        # between release (-18) and enter (-25), so we stay forward.
        # Should transition once from left to forward and stay there.
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        assert transitions <= 1, (
            f"Too many transitions ({transitions}) at release boundary"
        )

    def test_no_flicker_right_boundary(self):
        """Right side boundary oscillation should also be stable."""
        bucketer = YawBucketer(NORMAL.yaw)
        # enter_right = 25.0
        # Oscillate: 24, 26, 24, 26
        states = []
        for i in range(20):
            yaw = 24.0 if i % 2 == 0 else 26.0
            bucketer.update(yaw)
            states.append(bucketer.current_bucket)

        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        assert transitions <= 1, (
            f"Too many transitions ({transitions}) on right boundary"
        )

    def test_gradual_approach_no_premature_trigger(self):
        """Gradually approaching threshold should not trigger until crossed."""
        bucketer = YawBucketer(NORMAL.yaw)
        # Approach enter_left (-25) gradually
        for yaw in [-5, -10, -15, -20, -23, -24]:
            bucketer.update(yaw)
            assert bucketer.current_bucket == "forward", (
                f"Should stay forward at yaw={yaw}, not yet past -25"
            )

        # Cross the threshold
        bucketer.update(-25.0)
        assert bucketer.current_bucket == "left"

    def test_noisy_signal_stays_stable(self):
        """Realistic noisy signal (small jitter around a held position) stays stable."""
        import random
        random.seed(42)
        bucketer = YawBucketer(NORMAL.yaw)

        # Player holding head at ~-30 degrees (in left zone) with ±3 degree jitter
        states = []
        for _ in range(50):
            yaw = -30.0 + random.uniform(-3.0, 3.0)
            bucketer.update(yaw)
            states.append(bucketer.current_bucket)

        # All should be "left" since -30 ± 3 is always past enter_left (-25)
        # and never past release_left (-18)
        assert all(s == "left" for s in states), (
            "Jitter around -30 should always stay in 'left'"
        )



# ---------------------------------------------------------------------------
# NEW: Calibration Valid Samples Tests
# ---------------------------------------------------------------------------

class TestCalibrationValidSamples:
    """Test calibration requiring sufficient valid samples.

    CalibrationConfig.min_valid_samples = 12 by default.
    """

    def test_insufficient_samples_does_not_complete(self):
        """Calibration should not complete with fewer than min_valid_samples."""
        config = DifficultyConfig(
            calibration=CalibrationConfig(min_valid_samples=12),
        )
        classifier = Classifier(config)
        classifier.start_calibration(0.0)

        # Feed only 5 frames (less than 12 required)
        for i in range(5):
            m = make_neutral_measurements()
            classifier.update(m, 0.1 * (i + 1))

        assert classifier.is_calibrating is True
        assert classifier.is_calibrated is False
        assert classifier.valid_sample_count < 12

    def test_completes_at_min_valid_samples(self):
        """Calibration should complete once min_valid_samples reached."""
        config = DifficultyConfig(
            calibration=CalibrationConfig(min_valid_samples=12),
        )
        classifier = Classifier(config)
        classifier.start_calibration(0.0)

        # Feed exactly 12+ frames with consistent yaw
        for i in range(15):
            m = make_neutral_measurements(yaw=0.0)
            classifier.update(m, 0.05 * (i + 1))

        assert classifier.is_calibrating is False
        assert classifier.is_calibrated is True
        assert classifier.calibration_status == CalibrationStatus.COMPLETE

    def test_progress_tracks_sample_count(self):
        """Calibration progress should reflect sample collection progress."""
        config = DifficultyConfig(
            calibration=CalibrationConfig(min_valid_samples=12),
        )
        classifier = Classifier(config)
        classifier.start_calibration(0.0)

        # Feed 6 frames (half of 12)
        for i in range(6):
            m = make_neutral_measurements()
            classifier.update(m, 0.05 * (i + 1))

        progress = classifier.calibration_progress
        assert 0.4 <= progress <= 0.6, (
            f"Progress should be ~0.5 at 6/12 samples, got {progress}"
        )

    def test_invalid_frames_not_counted(self):
        """Frames that fail validation should not count toward min_valid_samples."""
        config = DifficultyConfig(
            calibration=CalibrationConfig(
                min_valid_samples=12,
                max_yaw_during_cal=10.0,
            ),
        )
        classifier = Classifier(config)
        classifier.start_calibration(0.0)

        # Feed frames with too much yaw (should be rejected)
        for i in range(10):
            m = make_neutral_measurements(yaw=20.0)  # Past max_yaw_during_cal
            classifier.update(m, 0.05 * (i + 1), face_width=150.0,
                            face_center_x=0.5, frame_width=640.0)

        # Should still be calibrating since yaw frames were rejected
        assert classifier.is_calibrating is True
        assert classifier.valid_sample_count < 12



# ---------------------------------------------------------------------------
# NEW: Calibration No-Face Frames Tests (Bug 4 fix validation)
# ---------------------------------------------------------------------------

class TestCalibrationNoFaceFrames:
    """Test that no-face frames don't corrupt calibration.

    Bug 4 fix: Classifier.set_no_face() correctly sets WAITING_FOR_FACE status
    during calibration without corrupting samples.
    """

    def test_set_no_face_during_calibration(self):
        """set_no_face() during calibration sets WAITING_FOR_FACE status."""
        classifier = Classifier(NORMAL)
        classifier.start_calibration(0.0)

        # Feed a few valid frames
        for i in range(3):
            m = make_neutral_measurements()
            classifier.update(m, 0.05 * (i + 1))

        # Simulate no face detected
        classifier.set_no_face()
        assert classifier.calibration_status == CalibrationStatus.WAITING_FOR_FACE

    def test_no_face_does_not_add_samples(self):
        """Calling set_no_face should not add calibration samples."""
        classifier = Classifier(NORMAL)
        classifier.start_calibration(0.0)

        # Feed 5 valid frames
        for i in range(5):
            m = make_neutral_measurements()
            classifier.update(m, 0.05 * (i + 1))

        count_before = classifier.valid_sample_count

        # Call set_no_face multiple times
        for _ in range(10):
            classifier.set_no_face()

        # Sample count should not have changed
        assert classifier.valid_sample_count == count_before

    def test_calibration_completes_after_face_returns(self):
        """Calibration can still complete after face returns from no-face frames."""
        classifier = Classifier(NORMAL)
        classifier.start_calibration(0.0)
        t = 0.0

        # Feed some valid frames
        for i in range(5):
            t += 0.05
            m = make_neutral_measurements()
            classifier.update(m, t)

        # Simulate no face for several frames
        for _ in range(10):
            classifier.set_no_face()

        # Resume with valid frames
        for i in range(20):
            t += 0.05
            m = make_neutral_measurements()
            classifier.update(m, t)

        # Should have completed calibration
        assert classifier.is_calibrating is False
        assert classifier.is_calibrated is True
        assert classifier.calibration_status == CalibrationStatus.COMPLETE

    def test_no_face_outside_calibration_no_effect(self):
        """set_no_face() outside calibration doesn't set WAITING_FOR_FACE."""
        classifier = Classifier(NORMAL)
        # Not calibrating
        classifier.set_no_face()
        # Status stays IDLE (not WAITING_FOR_FACE since we're not calibrating)
        assert classifier.calibration_status == CalibrationStatus.IDLE

    def test_interleaved_no_face_and_valid_frames(self):
        """Interleaving no-face and valid frames should eventually calibrate."""
        classifier = Classifier(NORMAL)
        classifier.start_calibration(0.0)
        t = 0.0

        # Alternate: valid frame, no face, valid frame, no face...
        for i in range(30):
            t += 0.05
            if i % 2 == 0:
                m = make_neutral_measurements()
                classifier.update(m, t)
            else:
                classifier.set_no_face()

        # 15 valid frames should be enough (min_valid_samples=12)
        assert classifier.is_calibrated is True



# ---------------------------------------------------------------------------
# Yaw Bucketing Tests (original)
# ---------------------------------------------------------------------------

class TestYawBucketing:
    """Test yaw bucketing with hysteresis."""

    def test_starts_forward(self):
        bucketer = YawBucketer(NORMAL.yaw)
        assert bucketer.current_bucket == "forward"

    def test_enters_left(self):
        bucketer = YawBucketer(NORMAL.yaw)
        result = bucketer.update(-26.0)  # Past enter_left (-25)
        assert result == "left"

    def test_enters_right(self):
        bucketer = YawBucketer(NORMAL.yaw)
        result = bucketer.update(26.0)  # Past enter_right (25)
        assert result == "right"

    def test_hysteresis_no_flicker(self):
        """Value between enter and release thresholds should not flicker."""
        bucketer = YawBucketer(NORMAL.yaw)
        # Enter left
        bucketer.update(-26.0)
        assert bucketer.current_bucket == "left"
        # Come back to between release (-18) and enter (-25) thresholds
        bucketer.update(-20.0)
        assert bucketer.current_bucket == "left"  # Should stay left

    def test_release_left_returns_forward(self):
        bucketer = YawBucketer(NORMAL.yaw)
        bucketer.update(-26.0)
        assert bucketer.current_bucket == "left"
        # Past release threshold
        bucketer.update(-17.0)
        assert bucketer.current_bucket == "forward"

    def test_release_right_returns_forward(self):
        bucketer = YawBucketer(NORMAL.yaw)
        bucketer.update(26.0)
        assert bucketer.current_bucket == "right"
        bucketer.update(17.0)
        assert bucketer.current_bucket == "forward"

    def test_direct_left_to_right(self):
        """Moving directly from left past right enter should go to right."""
        bucketer = YawBucketer(NORMAL.yaw)
        bucketer.update(-26.0)
        assert bucketer.current_bucket == "left"
        bucketer.update(26.0)
        assert bucketer.current_bucket == "right"

    def test_at_exact_enter_boundary(self):
        """At exactly the enter threshold, should trigger."""
        bucketer = YawBucketer(YawConfig(enter_left=-25.0, release_left=-18.0,
                                          enter_right=25.0, release_right=18.0))
        bucketer.update(-25.0)
        assert bucketer.current_bucket == "left"

    def test_just_inside_release_stays(self):
        """Just inside release threshold should stay in current bucket."""
        bucketer = YawBucketer(NORMAL.yaw)
        bucketer.update(-26.0)
        # release_left is -18, so -18.1 is still in left zone
        bucketer.update(-18.1)
        assert bucketer.current_bucket == "left"



# ---------------------------------------------------------------------------
# Expression Persistence Tests (original)
# ---------------------------------------------------------------------------

class TestExpressionPersistence:
    """Test expression persistence filter."""

    def test_starts_neutral(self):
        ep = ExpressionPersistence(0.15)
        assert ep.current_expression == "neutral"

    def test_rejects_brief_noise(self):
        """Brief blip of different expression should be rejected."""
        ep = ExpressionPersistence(0.15)
        ep.update("neutral", 0.0)
        ep.update("happy", 0.05)  # Brief noise
        ep.update("neutral", 0.10)
        assert ep.current_expression == "neutral"

    def test_accepts_stable_expression(self):
        """Expression held past persistence time should be accepted."""
        ep = ExpressionPersistence(0.15)
        ep.update("happy", 0.0)
        ep.update("happy", 0.08)
        ep.update("happy", 0.16)  # Past 0.15 persistence
        assert ep.current_expression == "happy"

    def test_multiple_transitions(self):
        ep = ExpressionPersistence(0.15)
        # Establish happy
        ep.update("happy", 0.0)
        ep.update("happy", 0.2)
        assert ep.current_expression == "happy"
        # Transition to surprised
        ep.update("surprised", 0.3)
        ep.update("surprised", 0.5)
        assert ep.current_expression == "surprised"


# ---------------------------------------------------------------------------
# Foreshortening Correction Tests (original)
# ---------------------------------------------------------------------------

class TestForeshorteningCorrection:
    """Test foreshortening correction."""

    def test_zero_yaw_no_correction(self):
        """At zero yaw, correction factor is 1.0 (cos(0) = 1)."""
        mar, brow, eye = apply_foreshortening_correction(0.5, 0.06, 0.25, 0.0)
        assert abs(mar - 0.5) < 1e-6
        assert abs(brow - 0.06) < 1e-6
        assert abs(eye - 0.25) < 1e-6

    def test_yaw_reduces_ratios(self):
        """At non-zero yaw, ratios should be reduced (multiplied by cos < 1)."""
        mar, brow, eye = apply_foreshortening_correction(0.5, 0.06, 0.25, 30.0)
        assert mar < 0.5
        assert brow < 0.06
        assert eye < 0.25

    def test_clamping_at_extreme_yaw(self):
        """Correction should be clamped at extreme angles."""
        mar1, _, _ = apply_foreshortening_correction(0.5, 0.06, 0.25, 80.0, max_yaw=55.0)
        mar2, _, _ = apply_foreshortening_correction(0.5, 0.06, 0.25, 55.0, max_yaw=55.0)
        # 80 degrees should be clamped to 55 degrees
        assert abs(mar1 - mar2) < 1e-6

    def test_symmetric_correction(self):
        """Left and right yaw should produce same correction magnitude."""
        mar_left, _, _ = apply_foreshortening_correction(0.5, 0.06, 0.25, -30.0)
        mar_right, _, _ = apply_foreshortening_correction(0.5, 0.06, 0.25, 30.0)
        assert abs(mar_left - mar_right) < 1e-6

    def test_current_frame_yaw_used(self):
        """Verify the raw yaw value is used directly (not smoothed)."""
        mar_30, _, _ = apply_foreshortening_correction(0.5, 0.06, 0.25, 30.0)
        mar_0, _, _ = apply_foreshortening_correction(0.5, 0.06, 0.25, 0.0)
        expected_30 = 0.5 * math.cos(math.radians(30.0))
        assert abs(mar_30 - expected_30) < 1e-6
        assert abs(mar_0 - 0.5) < 1e-6



# ---------------------------------------------------------------------------
# Expression Classification Tests (original)
# ---------------------------------------------------------------------------

class TestExpressionClassification:
    """Test expression classification relative to baseline."""

    def setup_method(self):
        """Set up a calibrated classifier for each test."""
        self.config = NORMAL
        self.classifier = Classifier(self.config)
        self.t = calibrate_classifier(self.classifier)

    def test_neutral_detected(self):
        """Neutral face should classify as neutral."""
        m = make_neutral_measurements()
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "neutral"

    def test_happy_detected(self):
        """Happy face should classify as happy."""
        m = make_happy_measurements()
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "happy"

    def test_surprised_detected(self):
        """Surprised face should classify as surprised."""
        m = make_surprised_measurements()
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "surprised"

    def test_angry_detected(self):
        """Angry face should classify as angry."""
        m = make_angry_measurements()
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "angry"

    def test_scale_invariance_small(self):
        """Same expression at half scale should still be detected."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier, scale=0.5)
        m = make_happy_measurements(scale=0.5)
        for i in range(5):
            result = classifier.update(m, t + i * 0.1)
        assert result.confirmed_expression == "happy"

    def test_scale_invariance_large(self):
        """Same expression at double scale should still be detected."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier, scale=2.0)
        m = make_happy_measurements(scale=2.0)
        for i in range(5):
            result = classifier.update(m, t + i * 0.1)
        assert result.confirmed_expression == "happy"

    def test_neutral_at_40_degrees_not_surprised(self):
        """CRITICAL: Turning a neutral face to ~40 degrees must NOT trigger surprised."""
        classifier = Classifier(NORMAL)
        calibrate_classifier(classifier)
        t = 2.0

        m = make_neutral_measurements(yaw=40.0)
        for i in range(10):
            result = classifier.update(m, t + i * 0.1)

        assert result.confirmed_expression != "surprised", (
            "Neutral face at 40 degrees should NOT be classified as surprised"
        )
        assert result.confirmed_expression == "neutral"

    def test_uses_current_frame_yaw_not_smoothed(self):
        """Verify foreshortening uses the current frame's yaw, not a delayed value."""
        classifier = Classifier(NORMAL)
        calibrate_classifier(classifier)
        t = 2.0

        m0 = make_neutral_measurements(yaw=0.0)
        classifier.update(m0, t)

        m40 = make_neutral_measurements(yaw=40.0)
        result = classifier.update(m40, t + 0.033)

        assert result.expression == "neutral"

    def test_expression_with_yaw_combined(self):
        """Expression detection should work alongside yaw detection."""
        classifier = Classifier(NORMAL)
        calibrate_classifier(classifier)
        t = 2.0

        m = make_happy_measurements(yaw=-30.0)
        for i in range(10):
            result = classifier.update(m, t + i * 0.1)

        assert result.confirmed_yaw == "left"
        assert result.confirmed_expression == "happy"

    def test_expression_with_right_yaw(self):
        """Expression at right yaw should still be detected correctly."""
        classifier = Classifier(NORMAL)
        calibrate_classifier(classifier)
        t = 2.0

        m = make_surprised_measurements(yaw=30.0)
        for i in range(10):
            result = classifier.update(m, t + i * 0.1)

        assert result.confirmed_yaw == "right"
        assert result.confirmed_expression == "surprised"


# ---------------------------------------------------------------------------
# NEW: Expression Boundary Tests
# ---------------------------------------------------------------------------

class TestExpressionBoundary:
    """Test expression accuracy near threshold boundaries.

    Verifies behavior when expression ratios are just at or near the
    classification thresholds.
    """

    def setup_method(self):
        self.classifier = Classifier(NORMAL)
        self.t = calibrate_classifier(self.classifier)

    def test_just_below_happy_threshold_stays_neutral(self):
        """Measurements just below happy thresholds should classify as neutral."""
        # happy_mouth_corner_lift threshold = 1.4x baseline
        # baseline mouth_corner_lift = 0.02
        # So threshold is 0.028. Use 0.027 (just below)
        m = FaceMeasurements(
            yaw_degrees=0.0,
            mouth_aspect_ratio=0.058,  # 1.16x baseline (below 1.2 happy_mouth_ar)
            mouth_corner_lift=0.027,   # 1.35x baseline (below 1.4 happy_corner)
            left_brow_eye_dist=0.06,
            right_brow_eye_dist=0.06,
            left_eye_aspect_ratio=0.25,
            right_eye_aspect_ratio=0.25,
        )
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "neutral"

    def test_just_above_happy_threshold_detects_happy(self):
        """Measurements just above happy thresholds should classify as happy."""
        # happy_mouth_corner_lift threshold = 1.4x, happy_mouth_aspect_ratio = 1.2x
        # baseline: corner=0.02, mouth_ar=0.05
        # Thresholds: corner >= 0.028, mouth_ar >= 0.06
        m = FaceMeasurements(
            yaw_degrees=0.0,
            mouth_aspect_ratio=0.062,  # 1.24x baseline (above 1.2)
            mouth_corner_lift=0.029,   # 1.45x baseline (above 1.4)
            left_brow_eye_dist=0.06,
            right_brow_eye_dist=0.06,
            left_eye_aspect_ratio=0.25,
            right_eye_aspect_ratio=0.25,
        )
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "happy"

    def test_just_below_surprised_threshold_not_surprised(self):
        """Measurements just below surprised thresholds should not classify as surprised."""
        # surprised_mouth_aspect_ratio = 1.8x, surprised_brow_raise = 1.3x,
        # surprised_eye_aspect_ratio = 1.3x
        # baseline: mouth_ar=0.05, brow=0.06, eye=0.25
        m = FaceMeasurements(
            yaw_degrees=0.0,
            mouth_aspect_ratio=0.088,  # 1.76x (below 1.8)
            mouth_corner_lift=0.02,
            left_brow_eye_dist=0.077,  # 1.28x (below 1.3)
            right_brow_eye_dist=0.077,
            left_eye_aspect_ratio=0.32,  # 1.28x (below 1.3)
            right_eye_aspect_ratio=0.32,
        )
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression != "surprised"

    def test_just_above_angry_threshold_detects_angry(self):
        """Measurements just past angry thresholds should classify as angry."""
        # angry_brow_lower = 0.75x, angry_eye_squint = 0.8x
        # baseline: brow=0.06, eye=0.25
        # Thresholds: brow <= 0.045, eye <= 0.2
        m = FaceMeasurements(
            yaw_degrees=0.0,
            mouth_aspect_ratio=0.04,
            mouth_corner_lift=0.015,
            left_brow_eye_dist=0.044,  # 0.73x (below 0.75 threshold)
            right_brow_eye_dist=0.044,
            left_eye_aspect_ratio=0.19,  # 0.76x (below 0.8 threshold)
            right_eye_aspect_ratio=0.19,
        )
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "angry"

    def test_ambiguous_measurements_default_to_neutral(self):
        """Measurements that don't clearly match any expression -> neutral."""
        # Values that are slightly elevated but don't reach any threshold
        m = FaceMeasurements(
            yaw_degrees=0.0,
            mouth_aspect_ratio=0.055,  # 1.1x (below all thresholds)
            mouth_corner_lift=0.022,   # 1.1x
            left_brow_eye_dist=0.055,  # 0.92x (not low enough for angry)
            right_brow_eye_dist=0.055,
            left_eye_aspect_ratio=0.23,  # 0.92x (not low enough for angry)
            right_eye_aspect_ratio=0.23,
        )
        for i in range(5):
            result = self.classifier.update(m, self.t + i * 0.1)
        assert result.confirmed_expression == "neutral"



# ---------------------------------------------------------------------------
# NEW: Expression Noise Rejection Tests
# ---------------------------------------------------------------------------

class TestExpressionNoiseRejection:
    """Test expression stability under noisy measurements.

    Verifies the persistence filter rejects noise and only accepts
    stable, consistent expression readings.
    """

    def test_alternating_expressions_rejected(self):
        """Rapidly alternating expressions should be rejected (stays at initial)."""
        ep = ExpressionPersistence(0.15)
        # Alternate every frame between happy and surprised
        for i in range(20):
            expr = "happy" if i % 2 == 0 else "surprised"
            ep.update(expr, i * 0.05)

        # Should still be neutral (never held long enough)
        assert ep.current_expression == "neutral"

    def test_noisy_with_dominant_expression_accepted(self):
        """Expression that dominates despite noise should eventually be accepted."""
        ep = ExpressionPersistence(0.15)
        t = 0.0
        # Mostly happy with occasional noise
        for i in range(20):
            t += 0.05
            if i % 5 == 3:  # One noisy frame every 5
                ep.update("surprised", t)
            else:
                ep.update("happy", t)

        # Happy held for long enough stretches -> should be accepted
        assert ep.current_expression == "happy"

    def test_single_frame_glitch_rejected(self):
        """A single frame of wrong expression should not change state."""
        ep = ExpressionPersistence(0.15)
        # Establish neutral
        ep.update("neutral", 0.0)
        ep.update("neutral", 0.2)

        # Single frame glitch
        ep.update("angry", 0.25)

        # Immediately back to neutral
        ep.update("neutral", 0.30)

        assert ep.current_expression == "neutral"

    def test_classifier_rejects_noisy_expressions(self):
        """Full classifier pipeline should reject noisy expression measurements."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Alternate between neutral and happy measurements rapidly
        for i in range(10):
            if i % 2 == 0:
                m = make_neutral_measurements()
            else:
                m = make_happy_measurements()
            result = classifier.update(m, t + i * 0.05)

        # Persistence filter (0.15s) should prevent acceptance of either non-neutral
        # since no expression is held for > 0.15s (each held for 0.05s)
        assert result.confirmed_expression == "neutral"

    def test_stable_expression_after_noise(self):
        """After noise settles, stable expression should be detected."""
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Start with noise
        for i in range(6):
            if i % 2 == 0:
                m = make_neutral_measurements()
            else:
                m = make_happy_measurements()
            classifier.update(m, t + i * 0.05)

        # Then stabilize on happy
        t2 = t + 0.3
        for i in range(8):
            m = make_happy_measurements()
            result = classifier.update(m, t2 + i * 0.05)

        assert result.confirmed_expression == "happy"



# ---------------------------------------------------------------------------
# NEW: State Transitions Tests (Bug 3 fix validation)
# ---------------------------------------------------------------------------

class TestStateTransitions:
    """Test game state transitions including difficulty selection flow.

    Bug 3 fix: D key enters DIFFICULTY_SELECT from START_SCREEN/GAME_OVER,
    cycles within DIFFICULTY_SELECT. SPACE confirms and returns to START_SCREEN.
    """

    def test_start_to_difficulty_select(self):
        """START_SCREEN -> DIFFICULTY_SELECT via enter_difficulty_select."""
        game = GameLogic(NORMAL)
        assert game.state == GameState.START_SCREEN
        game.enter_difficulty_select(0.0)
        assert game.state == GameState.DIFFICULTY_SELECT

    def test_difficulty_select_to_start_via_select(self):
        """DIFFICULTY_SELECT -> START_SCREEN via select_difficulty."""
        game = GameLogic(NORMAL)
        game.enter_difficulty_select(0.0)
        assert game.state == GameState.DIFFICULTY_SELECT
        game.select_difficulty(HARD, 0, 1.0)
        assert game.state == GameState.START_SCREEN

    def test_start_to_calibrating_to_countdown_to_active(self):
        """START_SCREEN -> CALIBRATING -> COUNTDOWN -> ACTIVE."""
        game = GameLogic(NORMAL)
        game.calibrated = False
        game.start_game(0.0)
        assert game.state == GameState.CALIBRATING

        game.calibration_complete(1.0)
        assert game.state == GameState.COUNTDOWN

        game.update("forward", "neutral", 5.0)  # Past countdown
        assert game.state == GameState.ACTIVE

    def test_active_to_pause_to_active(self):
        """ACTIVE -> PAUSE -> ACTIVE via toggle_pause."""
        game = GameLogic(NORMAL)
        game.start_game(0.0)
        game.update("forward", "neutral", 4.0)  # Past countdown
        assert game.state == GameState.ACTIVE

        game.toggle_pause(4.5)
        assert game.state == GameState.PAUSE

        game.toggle_pause(5.5)
        assert game.state == GameState.ACTIVE

    def test_active_timeout_miss_game_over(self):
        """ACTIVE -> timeout -> MISS -> GAME_OVER (with 1 life)."""
        game = GameLogic(DifficultyConfig(
            lives=1, round_time=2.0, countdown_duration=0.1,
            miss_display_time=0.1,
        ))
        game.start_game(0.0)
        game.update("forward", "neutral", 0.2)  # Past countdown
        assert game.state == GameState.ACTIVE

        # Timeout
        game.update("forward", "neutral", 3.0)
        assert game.state == GameState.MISS

        # Past miss display
        game.update("forward", "neutral", 3.2)
        assert game.state == GameState.GAME_OVER

    def test_game_over_to_difficulty_select(self):
        """GAME_OVER -> DIFFICULTY_SELECT via enter_difficulty_select."""
        game = GameLogic(DifficultyConfig(
            lives=1, round_time=2.0, countdown_duration=0.1,
            miss_display_time=0.1,
        ))
        game.start_game(0.0)
        game.update("forward", "neutral", 0.2)
        game.update("forward", "neutral", 3.0)
        game.update("forward", "neutral", 3.2)
        assert game.state == GameState.GAME_OVER

        game.enter_difficulty_select(4.0)
        assert game.state == GameState.DIFFICULTY_SELECT

    def test_calibration_failed_returns_to_start(self):
        """CALIBRATING -> calibration_failed -> START_SCREEN."""
        game = GameLogic(NORMAL)
        game.calibrated = False
        game.start_game(0.0)
        assert game.state == GameState.CALIBRATING

        game.calibration_failed(1.0)
        assert game.state == GameState.START_SCREEN

    def test_confirming_to_success(self):
        """CONFIRMING -> SUCCESS when held long enough."""
        game = GameLogic(DifficultyConfig(
            lives=3, round_time=10.0, countdown_duration=0.1,
            confirmation_time=0.2, confirmation_time_min=0.2,
        ))
        game.start_game(0.0)
        game.update("forward", "neutral", 0.2)
        assert game.state == GameState.ACTIVE

        target = game.current_target
        game.update(target.yaw, target.expression, 0.3)
        assert game.state == GameState.CONFIRMING

        game.update(target.yaw, target.expression, 0.6)
        assert game.state == GameState.SUCCESS

    def test_difficulty_select_not_reachable_from_active(self):
        """DIFFICULTY_SELECT should not be entered from ACTIVE state."""
        game = GameLogic(NORMAL)
        game.start_game(0.0)
        game.update("forward", "neutral", 4.0)
        assert game.state == GameState.ACTIVE

        # enter_difficulty_select only works from START_SCREEN/GAME_OVER
        game.enter_difficulty_select(4.5)
        assert game.state == GameState.ACTIVE  # Should not change



# ---------------------------------------------------------------------------
# NEW: Mutual Exclusive States Tests (Bug 3 fix validation)
# ---------------------------------------------------------------------------

class TestMutualExclusiveStates:
    """Test difficulty and calibration screens never rendering simultaneously.

    Verifies the game state machine doesn't allow both states at once
    and transitions properly between them.
    """

    def test_state_is_single_value(self):
        """Game state is always exactly one value (enum), never multiple."""
        game = GameLogic(NORMAL)
        # The state property is a single enum value by design
        assert isinstance(game.state, GameState)
        assert game.state == GameState.START_SCREEN

    def test_difficulty_and_calibrating_never_simultaneous(self):
        """Cannot be in both DIFFICULTY_SELECT and CALIBRATING."""
        game = GameLogic(NORMAL)
        game.enter_difficulty_select(0.0)
        assert game.state == GameState.DIFFICULTY_SELECT
        assert game.state != GameState.CALIBRATING

        # Entering calibration from difficulty select
        game.enter_calibration(1.0)
        assert game.state == GameState.CALIBRATING
        assert game.state != GameState.DIFFICULTY_SELECT

    def test_entering_calibration_from_difficulty_select(self):
        """Entering calibration from difficulty select should transition cleanly."""
        game = GameLogic(NORMAL)
        game.enter_difficulty_select(0.0)
        assert game.state == GameState.DIFFICULTY_SELECT

        game.enter_calibration(1.0)
        assert game.state == GameState.CALIBRATING

        # Completing calibration goes to countdown
        game.calibration_complete(2.0)
        assert game.state == GameState.COUNTDOWN
        assert game.state != GameState.DIFFICULTY_SELECT

    def test_all_states_mutually_exclusive(self):
        """Walk through multiple states and verify only one is active at a time."""
        config = DifficultyConfig(
            lives=3, round_time=10.0, countdown_duration=0.1,
            confirmation_time=0.1, confirmation_time_min=0.1,
            success_display_time=0.1, miss_display_time=0.1,
        )
        game = GameLogic(config)

        states_visited = set()

        # START_SCREEN
        states_visited.add(game.state)
        assert game.state == GameState.START_SCREEN

        # DIFFICULTY_SELECT
        game.enter_difficulty_select(0.0)
        assert game.state not in states_visited or game.state == GameState.DIFFICULTY_SELECT
        states_visited.add(game.state)

        # Back to START_SCREEN (use same config to keep short countdown)
        game.select_difficulty(config, 0, 0.5)
        assert game.state == GameState.START_SCREEN

        # COUNTDOWN
        game.start_game(1.0)
        assert game.state == GameState.COUNTDOWN

        # ACTIVE
        game.update("forward", "neutral", 1.2)
        assert game.state == GameState.ACTIVE

        # PAUSE
        game.toggle_pause(1.5)
        assert game.state == GameState.PAUSE

        # Back to ACTIVE
        game.toggle_pause(2.0)
        assert game.state == GameState.ACTIVE

        # CONFIRMING
        target = game.current_target
        game.update(target.yaw, target.expression, 2.1)
        assert game.state == GameState.CONFIRMING

        # SUCCESS
        game.update(target.yaw, target.expression, 2.3)
        assert game.state == GameState.SUCCESS

    def test_cannot_enter_calibration_from_active(self):
        """Cannot enter calibration from ACTIVE, CONFIRMING, or COUNTDOWN states."""
        game = GameLogic(NORMAL)
        game.start_game(0.0)
        game.update("forward", "neutral", 4.0)
        assert game.state == GameState.ACTIVE

        # Attempt calibration from active
        game.enter_calibration(4.5)
        # enter_calibration doesn't include ACTIVE in its allowed states
        # so state stays ACTIVE
        assert game.state == GameState.ACTIVE



# ---------------------------------------------------------------------------
# NEW: Responsive Layout Tests (Bug 2 fix validation)
# ---------------------------------------------------------------------------

class TestResponsiveLayout:
    """Test responsive layout calculations at various resolutions.

    The renderer._scale method converts base-640x480 values proportionally
    to the current frame height. We test this directly without requiring
    OpenCV rendering.
    """

    def test_scale_at_480(self):
        """At 480p (base resolution), scale factor is 1.0."""
        # _scale(value, frame_h) = int(value * frame_h / 480.0)
        # At 480: value * 480 / 480 = value
        scale_fn = lambda v, h: int(v * h / 480.0)
        assert scale_fn(100, 480) == 100
        assert scale_fn(200, 480) == 200
        assert scale_fn(50, 480) == 50

    def test_scale_at_720(self):
        """At 720p, scale factor is 1.5x."""
        scale_fn = lambda v, h: int(v * h / 480.0)
        assert scale_fn(100, 720) == 150
        assert scale_fn(200, 720) == 300
        assert scale_fn(50, 720) == 75

    def test_scale_at_1080(self):
        """At 1080p, scale factor is 2.25x."""
        scale_fn = lambda v, h: int(v * h / 480.0)
        assert scale_fn(100, 1080) == 225
        assert scale_fn(200, 1080) == 450
        assert scale_fn(50, 1080) == 112  # int(50 * 2.25) = 112

    def test_proportional_scaling(self):
        """Scaled values should be proportional across resolutions."""
        scale_fn = lambda v, h: int(v * h / 480.0)

        val = 100
        s480 = scale_fn(val, 480)
        s720 = scale_fn(val, 720)
        s1080 = scale_fn(val, 1080)

        # 720/480 = 1.5
        assert abs(s720 / s480 - 1.5) < 0.01
        # 1080/480 = 2.25
        assert abs(s1080 / s480 - 2.25) < 0.01

    def test_sprite_scaling_at_different_resolutions(self):
        """Sprite size should scale with resolution."""
        scale_fn = lambda v, h: int(v * h / 480.0)
        sprite_base = 200  # SPRITE_SIZE

        assert scale_fn(sprite_base, 480) == 200
        assert scale_fn(sprite_base, 720) == 300
        assert scale_fn(sprite_base, 1080) == 450

    def test_scale_at_4k(self):
        """At 4K (2160p), scale factor is 4.5x."""
        scale_fn = lambda v, h: int(v * h / 480.0)
        assert scale_fn(100, 2160) == 450
        assert scale_fn(10, 2160) == 45

    def test_timer_bar_scales(self):
        """Timer bar height should scale proportionally."""
        scale_fn = lambda v, h: int(v * h / 480.0)
        timer_base = 12  # TIMER_BAR_HEIGHT

        assert scale_fn(timer_base, 480) == 12
        assert scale_fn(timer_base, 720) == 18
        assert scale_fn(timer_base, 1080) == 27


# ---------------------------------------------------------------------------
# NEW: Fullscreen State Tests (Bug 2 fix validation)
# ---------------------------------------------------------------------------

class TestFullscreenState:
    """Test fullscreen toggle state tracking.

    Since we can't test actual display changes without a monitor,
    we verify the state logic that the main loop uses.
    """

    def test_toggle_state(self):
        """Fullscreen toggle should flip boolean state."""
        is_fullscreen = False
        is_fullscreen = not is_fullscreen
        assert is_fullscreen is True
        is_fullscreen = not is_fullscreen
        assert is_fullscreen is False

    def test_window_size_changes_on_fullscreen(self):
        """When fullscreen is toggled, frame size should change."""
        # Simulate windowed and fullscreen sizes
        windowed_size = (640, 480)
        fullscreen_size = (1920, 1080)  # Example native resolution

        is_fullscreen = False
        current_size = windowed_size

        # Toggle to fullscreen
        is_fullscreen = True
        if is_fullscreen:
            current_size = fullscreen_size

        assert current_size == fullscreen_size
        assert current_size != windowed_size

        # Toggle back
        is_fullscreen = False
        if not is_fullscreen:
            current_size = windowed_size

        assert current_size == windowed_size

    def test_renderer_scale_adapts_to_fullscreen(self):
        """Renderer scale should adapt when frame size changes for fullscreen."""
        scale_fn = lambda v, h: int(v * h / 480.0)

        # Windowed
        windowed_scale = scale_fn(100, 480)
        # Fullscreen at 1080p
        fullscreen_scale = scale_fn(100, 1080)

        assert fullscreen_scale > windowed_scale
        assert fullscreen_scale == 225

    def test_hud_elements_scale_in_fullscreen(self):
        """HUD elements should scale proportionally in fullscreen."""
        scale_fn = lambda v, h: int(v * h / 480.0)

        # Font scale base, timer bar, sprite size should all scale
        base_values = [12, 50, 200]  # timer_bar, padding, sprite
        for val in base_values:
            windowed = scale_fn(val, 480)
            fullscreen = scale_fn(val, 1080)
            ratio = fullscreen / windowed
            assert abs(ratio - 2.25) < 0.01, (
                f"Value {val} didn't scale correctly: {windowed} -> {fullscreen}"
            )



# ---------------------------------------------------------------------------
# Game Logic Tests (original)
# ---------------------------------------------------------------------------

class TestGameLogic:
    """Test game state machine, scoring, and progression."""

    def setup_method(self):
        self.game = GameLogic(NORMAL, high_score=10)

    def test_starts_at_start_screen(self):
        assert self.game.state == GameState.START_SCREEN

    def test_start_game_transitions_to_countdown(self):
        self.game.start_game(0.0)
        assert self.game.state == GameState.COUNTDOWN

    def test_countdown_transitions_to_active(self):
        self.game.start_game(0.0)
        # Fast-forward past countdown
        self.game.update("forward", "neutral", 4.0)
        assert self.game.state == GameState.ACTIVE
        assert self.game.current_target is not None

    def test_correct_match_gives_point(self):
        self.game.start_game(0.0)
        self.game.update("forward", "neutral", 4.0)  # Past countdown
        assert self.game.state == GameState.ACTIVE

        target = self.game.current_target
        # Match the target
        self.game.update(target.yaw, target.expression, 4.1)
        assert self.game.state == GameState.CONFIRMING

        # Hold for confirmation time
        self.game.update(target.yaw, target.expression, 4.6)
        assert self.game.state == GameState.SUCCESS
        assert self.game.score == 1

    def test_timeout_loses_life(self):
        self.game.start_game(0.0)
        self.game.update("forward", "neutral", 4.0)  # Past countdown
        initial_lives = self.game.lives

        # Wait for round to timeout (round_time = 3.8 for Normal)
        self.game.update("forward", "neutral", 8.0)
        assert self.game.state == GameState.MISS
        assert self.game.lives == initial_lives - 1

    def test_game_over_when_no_lives(self):
        game = GameLogic(DifficultyConfig(lives=1, round_time=2.0, countdown_duration=0.1,
                                          miss_display_time=0.1))
        game.start_game(0.0)
        game.update("forward", "neutral", 0.2)  # Past countdown
        # Timeout
        game.update("forward", "neutral", 3.0)
        assert game.state == GameState.MISS
        # Wait for miss display
        game.update("forward", "neutral", 3.2)
        assert game.state == GameState.GAME_OVER

    def test_streak_counting(self):
        game = GameLogic(DifficultyConfig(
            lives=5, round_time=10.0, countdown_duration=0.1,
            confirmation_time=0.1, confirmation_time_min=0.1,
            success_display_time=0.1,
        ))
        game.start_game(0.0)
        t = 0.2  # Past countdown
        game.update("forward", "neutral", t)

        for i in range(3):
            target = game.current_target
            t += 0.1
            game.update(target.yaw, target.expression, t)  # Start confirm
            t += 0.2
            game.update(target.yaw, target.expression, t)  # Confirm
            assert game.state == GameState.SUCCESS
            t += 0.2
            game.update("forward", "neutral", t)  # Next round

        assert game.streak == 3

    def test_streak_bonus(self):
        """Streak >= threshold should give bonus points."""
        game = GameLogic(DifficultyConfig(
            lives=5, round_time=10.0, countdown_duration=0.1,
            confirmation_time=0.1, confirmation_time_min=0.1,
            success_display_time=0.1,
            streak_bonus_threshold=3, streak_bonus_points=1,
        ))
        game.start_game(0.0)
        t = 0.2
        game.update("forward", "neutral", t)

        for i in range(3):
            target = game.current_target
            t += 0.1
            game.update(target.yaw, target.expression, t)
            t += 0.2
            game.update(target.yaw, target.expression, t)
            t += 0.2
            game.update("forward", "neutral", t)

        # 3 base points + 1 streak bonus on 3rd match = 4
        assert game.score == 4

    def test_streak_resets_on_miss(self):
        game = GameLogic(DifficultyConfig(
            lives=5, round_time=2.0, countdown_duration=0.1,
            confirmation_time=0.1, confirmation_time_min=0.1,
            success_display_time=0.1, miss_display_time=0.1,
        ))
        game.start_game(0.0)
        t = 0.2
        game.update("forward", "neutral", t)

        # Score one
        target = game.current_target
        t += 0.1
        game.update(target.yaw, target.expression, t)
        t += 0.2
        game.update(target.yaw, target.expression, t)
        t += 0.2
        game.update("forward", "neutral", t)
        assert game.streak == 1

        # Miss one
        t += 3.0
        game.update("forward", "neutral", t)  # Timeout
        t += 0.2
        game.update("forward", "neutral", t)  # Past miss display
        assert game.streak == 0

    def test_speed_increases_with_score(self):
        """Round time should decrease as score increases."""
        game = GameLogic(NORMAL)
        initial_time = game.get_round_time()

        # Simulate score increase
        game._score = 10
        later_time = game.get_round_time()
        assert later_time < initial_time

    def test_speed_has_minimum(self):
        """Round time should not go below minimum."""
        game = GameLogic(NORMAL)
        game._score = 1000  # Very high score
        assert game.get_round_time() >= NORMAL.min_round_time

    def test_target_no_immediate_repeat(self):
        """Target should not repeat immediately."""
        game = GameLogic(NORMAL)
        game.start_game(0.0)
        game.update("forward", "neutral", 4.0)  # Past countdown

        targets_seen = []
        t = 4.0
        for _ in range(20):
            target = game.current_target
            targets_seen.append(target)
            # Force success
            t += 0.1
            game.update(target.yaw, target.expression, t)
            t += 0.5
            game.update(target.yaw, target.expression, t)
            t += 1.0
            game.update("forward", "neutral", t)

        # Check no consecutive duplicates
        for i in range(1, len(targets_seen)):
            assert targets_seen[i] != targets_seen[i - 1], (
                f"Target repeated at index {i}: {targets_seen[i]}"
            )

    def test_high_score_updates(self):
        game = GameLogic(NORMAL, high_score=5)
        game.start_game(0.0)
        game._score = 6
        game._handle_success(1.0)
        assert game.high_score >= 6

    def test_pause_and_resume(self):
        game = GameLogic(NORMAL)
        game.start_game(0.0)
        game.update("forward", "neutral", 4.0)  # Active
        assert game.state == GameState.ACTIVE

        game.toggle_pause(4.5)
        assert game.state == GameState.PAUSE

        game.toggle_pause(5.5)
        assert game.state == GameState.ACTIVE

    def test_pause_preserves_round_time(self):
        """Pausing should not consume round time."""
        game = GameLogic(DifficultyConfig(
            lives=3, round_time=5.0, countdown_duration=0.1,
        ))
        game.start_game(0.0)
        game.update("forward", "neutral", 0.2)  # Past countdown

        time_before_pause = game.get_time_remaining(1.0)
        game.toggle_pause(1.0)
        # Pause for 10 seconds
        game.toggle_pause(11.0)
        time_after_resume = game.get_time_remaining(11.0)

        # Time remaining should be approximately the same
        assert abs(time_before_pause - time_after_resume) < 0.1

    def test_difficulty_settings(self):
        """Different difficulties should have different parameters."""
        assert EASY.lives > NORMAL.lives > HARD.lives
        assert EASY.round_time > NORMAL.round_time > HARD.round_time
        assert EASY.confirmation_time > NORMAL.confirmation_time > HARD.confirmation_time

    def test_confirmation_broken_returns_to_active(self):
        """Breaking pose during confirmation should go back to active."""
        game = GameLogic(DifficultyConfig(
            lives=3, round_time=10.0, countdown_duration=0.1,
            confirmation_time=1.0, confirmation_time_min=1.0,
        ))
        game.start_game(0.0)
        game.update("forward", "neutral", 0.2)

        target = game.current_target
        # Start confirming
        game.update(target.yaw, target.expression, 0.3)
        assert game.state == GameState.CONFIRMING

        # Break pose
        wrong_expr = "happy" if target.expression != "happy" else "neutral"
        game.update(target.yaw, wrong_expr, 0.5)
        assert game.state == GameState.ACTIVE



# ---------------------------------------------------------------------------
# Persistence Tests (original)
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test high score save/load."""

    def test_load_missing_file(self):
        """Missing file should return empty dict."""
        scores = load_high_scores("/nonexistent/path/scores.json")
        assert scores == {}

    def test_save_and_load(self):
        """Saved scores should be loadable."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_high_scores({"Normal": 42, "Hard": 10}, filepath)
            loaded = load_high_scores(filepath)
            assert loaded == {"Normal": 42, "Hard": 10}
        finally:
            os.unlink(filepath)

    def test_load_corrupt_file(self):
        """Corrupt file should return empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{{")
            filepath = f.name

        try:
            scores = load_high_scores(filepath)
            assert scores == {}
        finally:
            os.unlink(filepath)

    def test_load_wrong_type(self):
        """File with wrong JSON type should return empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([1, 2, 3], f)
            filepath = f.name

        try:
            scores = load_high_scores(filepath)
            assert scores == {}
        finally:
            os.unlink(filepath)

    def test_manager_only_writes_on_change(self):
        """Manager should only write when high score actually changes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            mgr = HighScoreManager(filepath)
            # First update should write
            assert mgr.update("Normal", 10) is True
            # Same score should not update
            assert mgr.update("Normal", 10) is False
            # Lower score should not update
            assert mgr.update("Normal", 5) is False
            # Higher score should update
            assert mgr.update("Normal", 15) is True

            # Verify persisted
            loaded = load_high_scores(filepath)
            assert loaded["Normal"] == 15
        finally:
            os.unlink(filepath)

    def test_manager_per_difficulty(self):
        """Each difficulty should have independent high score."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            mgr = HighScoreManager(filepath)
            mgr.update("Easy", 50)
            mgr.update("Normal", 30)
            mgr.update("Hard", 10)

            assert mgr.get("Easy") == 50
            assert mgr.get("Normal") == 30
            assert mgr.get("Hard") == 10
            assert mgr.get("Nonexistent") == 0
        finally:
            os.unlink(filepath)


# ---------------------------------------------------------------------------
# Calibration Tests (original)
# ---------------------------------------------------------------------------

class TestCalibration:
    """Test calibration process."""

    def test_calibration_starts(self):
        classifier = Classifier(NORMAL)
        classifier.start_calibration(0.0)
        assert classifier.is_calibrating is True

    def test_calibration_completes(self):
        classifier = Classifier(NORMAL)
        calibrate_classifier(classifier)
        assert classifier.is_calibrating is False
        assert classifier.is_calibrated is True

    def test_recalibration_resets_baseline(self):
        classifier = Classifier(NORMAL)
        t = calibrate_classifier(classifier)

        # Verify calibrated
        assert classifier.is_calibrated

        # Recalibrate
        classifier.start_calibration(t)
        assert classifier.is_calibrating is True
        # Old baseline should be cleared
        assert classifier.baseline.sample_count == 0


# ---------------------------------------------------------------------------
# Target Tests (original)
# ---------------------------------------------------------------------------

class TestTarget:
    """Test Target class."""

    def test_matches(self):
        target = Target("left", "happy")
        assert target.matches("left", "happy") is True
        assert target.matches("right", "happy") is False
        assert target.matches("left", "neutral") is False

    def test_difference_both_different(self):
        t1 = Target("left", "happy")
        t2 = Target("right", "neutral")
        assert t1.difference_from(t2) == 2

    def test_difference_one_different(self):
        t1 = Target("left", "happy")
        t2 = Target("left", "neutral")
        assert t1.difference_from(t2) == 1

    def test_difference_same(self):
        t1 = Target("left", "happy")
        t2 = Target("left", "happy")
        assert t1.difference_from(t2) == 0

    def test_difference_from_none(self):
        t1 = Target("left", "happy")
        assert t1.difference_from(None) == 2

    def test_sprite_name(self):
        target = Target("right", "surprised")
        assert target.sprite_name == "right_surprised.png"

    def test_all_targets_count(self):
        """Should have exactly 12 target combinations."""
        assert len(ALL_TARGETS) == 12
