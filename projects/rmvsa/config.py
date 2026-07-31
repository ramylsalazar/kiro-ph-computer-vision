"""Configuration dataclasses for Hole in the Wall: Face Challenge."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class YawConfig:
    """Hysteresis thresholds for yaw bucketing (degrees)."""

    enter_left: float = -25.0
    release_left: float = -18.0
    enter_right: float = 25.0
    release_right: float = 18.0


@dataclass(frozen=True)
class ExpressionThresholds:
    """Relative thresholds for expression detection (multiplied against baseline).

    Values represent how much above baseline a ratio must be to trigger.
    """

    # Happy: mouth corner lift relative to baseline
    happy_mouth_corner_lift: float = 1.4
    happy_mouth_aspect_ratio: float = 1.2

    # Surprised: mouth open + brow raise
    surprised_mouth_aspect_ratio: float = 1.8
    surprised_brow_raise: float = 1.3
    surprised_eye_aspect_ratio: float = 1.3

    # Angry: brow lower + narrow eyes
    angry_brow_lower: float = 0.75
    angry_eye_squint: float = 0.8
    angry_mouth_corner_lift: float = 0.85


@dataclass(frozen=True)
class CalibrationConfig:
    """Validation parameters for face calibration."""

    min_face_width: float = 80.0
    max_face_width: float = 400.0
    max_face_offset_ratio: float = 0.3  # how far off-center face can be
    max_yaw_during_cal: float = 10.0
    min_valid_samples: int = 12
    stillness_threshold: float = 2.0  # max std dev of yaw during calibration
    progress_display: bool = True


@dataclass(frozen=True)
class DifficultyConfig:
    """All tunables for a difficulty level."""

    name: str = "Normal"

    # Lives
    lives: int = 3

    # Timing (seconds)
    round_time: float = 3.8
    min_round_time: float = 1.8
    time_reduction_per_point: float = 0.08
    countdown_duration: float = 3.0
    success_display_time: float = 0.8
    miss_display_time: float = 1.0

    # Confirmation: player must hold pose this long (seconds)
    confirmation_time: float = 0.35
    confirmation_time_min: float = 0.2

    # Expression persistence: must detect same expression for this long (seconds)
    expression_persistence: float = 0.15

    # Speed milestone interval (every N points triggers "SPEED UP!")
    speed_milestone_interval: int = 5

    # Streak bonus: extra points for consecutive correct
    streak_bonus_threshold: int = 3
    streak_bonus_points: int = 1

    # Yaw config
    yaw: YawConfig = field(default_factory=YawConfig)

    # Expression thresholds
    expression: ExpressionThresholds = field(default_factory=ExpressionThresholds)

    # Calibration
    calibration_duration: float = 1.0
    calibration_frames: int = 15

    # Foreshortening correction
    max_yaw_for_correction: float = 55.0  # clamp beyond this

    # Target selection: minimum difference from previous target
    min_target_difference: int = 1  # at least 1 attribute must differ

    # Calibration validation config
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Yaw smoothing (exponential moving average alpha)
    yaw_smoothing_alpha: float = 0.3


EASY = DifficultyConfig(
    name="Easy",
    lives=5,
    round_time=5.0,
    min_round_time=2.5,
    time_reduction_per_point=0.05,
    confirmation_time=0.5,
    confirmation_time_min=0.3,
    expression_persistence=0.2,
    speed_milestone_interval=7,
    yaw=YawConfig(
        enter_left=-30.0,
        release_left=-22.0,
        enter_right=30.0,
        release_right=22.0,
    ),
    expression=ExpressionThresholds(
        happy_mouth_corner_lift=1.5,
        happy_mouth_aspect_ratio=1.3,
        surprised_mouth_aspect_ratio=2.0,
        surprised_brow_raise=1.4,
        surprised_eye_aspect_ratio=1.4,
        angry_brow_lower=0.7,
        angry_eye_squint=0.75,
        angry_mouth_corner_lift=0.8,
    ),
)

NORMAL = DifficultyConfig(name="Normal")

HARD = DifficultyConfig(
    name="Hard",
    lives=2,
    round_time=2.7,
    min_round_time=1.4,
    time_reduction_per_point=0.12,
    confirmation_time=0.25,
    confirmation_time_min=0.15,
    expression_persistence=0.1,
    speed_milestone_interval=4,
    yaw=YawConfig(
        enter_left=-20.0,
        release_left=-14.0,
        enter_right=20.0,
        release_right=14.0,
    ),
    expression=ExpressionThresholds(
        happy_mouth_corner_lift=1.3,
        happy_mouth_aspect_ratio=1.15,
        surprised_mouth_aspect_ratio=1.6,
        surprised_brow_raise=1.2,
        surprised_eye_aspect_ratio=1.2,
        angry_brow_lower=0.8,
        angry_eye_squint=0.85,
        angry_mouth_corner_lift=0.9,
    ),
)

DIFFICULTIES: Dict[str, DifficultyConfig] = {
    "Easy": EASY,
    "Normal": NORMAL,
    "Hard": HARD,
}

DIFFICULTY_ORDER = ["Easy", "Normal", "Hard"]


# Sprite / rendering constants
SPRITE_SIZE = 200  # pixels, square
TIMER_BAR_HEIGHT = 12
HUD_FONT_SCALE = 0.7
HUD_COLOR = (255, 255, 255)
HUD_SHADOW_COLOR = (0, 0, 0)
FEEDBACK_FONT_SCALE = 1.5

# Yaw labels
YAW_LABELS = ["left", "forward", "right"]
EXPRESSION_LABELS = ["neutral", "happy", "surprised", "angry"]

# All 12 target combinations
ALL_TARGETS = [
    (yaw, expr) for yaw in YAW_LABELS for expr in EXPRESSION_LABELS
]

# Background modes
BG_WEBCAM = 0
BG_DARK = 1
BG_CHROMA = 2
BG_MODES = ["webcam", "dark", "chroma"]

# Debug and fullscreen key constants
DEBUG_OVERLAY_KEY = ord('v')
FULLSCREEN_KEY = ord('f')
