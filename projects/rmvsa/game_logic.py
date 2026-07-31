"""Game logic for Hole in the Wall: Face Challenge.

Pure Python state machine - testable without OpenCV, MediaPipe, or a webcam.
Handles rounds, scoring, lives, streaks, speed scaling, target selection.
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List

from config import DifficultyConfig, ALL_TARGETS, DIFFICULTY_ORDER, DIFFICULTIES


class GameState(Enum):
    """Game states."""

    START_SCREEN = auto()
    DIFFICULTY_SELECT = auto()
    CALIBRATING = auto()
    COUNTDOWN = auto()
    ACTIVE = auto()
    CONFIRMING = auto()  # Player holding correct pose
    SUCCESS = auto()
    MISS = auto()
    PAUSE = auto()
    GAME_OVER = auto()


@dataclass
class Target:
    """A target pose the player must match."""

    yaw: str  # "left", "forward", "right"
    expression: str  # "neutral", "happy", "surprised", "angry"

    def matches(self, player_yaw: str, player_expression: str) -> bool:
        """Check if player matches this target."""
        return self.yaw == player_yaw and self.expression == player_expression

    def difference_from(self, other: Optional["Target"]) -> int:
        """Count how many attributes differ from another target."""
        if other is None:
            return 2
        diff = 0
        if self.yaw != other.yaw:
            diff += 1
        if self.expression != other.expression:
            diff += 1
        return diff

    @property
    def sprite_name(self) -> str:
        """Return the sprite filename for this target."""
        return f"{self.yaw}_{self.expression}.png"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Target):
            return NotImplemented
        return self.yaw == other.yaw and self.expression == other.expression

    def __hash__(self) -> int:
        return hash((self.yaw, self.expression))


@dataclass
class GameStats:
    """Current game statistics."""

    score: int = 0
    high_score: int = 0
    lives: int = 3
    streak: int = 0
    best_streak: int = 0
    round_number: int = 0
    difficulty_name: str = "Normal"


class GameLogic:
    """Game state machine with scoring, lives, streaks, and difficulty scaling.

    Pure Python - no rendering or I/O dependencies.
    """

    def __init__(self, config: DifficultyConfig, high_score: int = 0):
        self._config = config
        self._state = GameState.START_SCREEN
        self._score = 0
        self._high_score = high_score
        self._lives = config.lives
        self._streak = 0
        self._best_streak = 0
        self._round_number = 0

        self._current_target: Optional[Target] = None
        self._previous_target: Optional[Target] = None

        # Timing
        self._state_start_time: float = 0.0
        self._round_start_time: float = 0.0
        self._confirm_start_time: float = 0.0

        # Speed milestone tracking
        self._last_milestone_score: int = 0
        self._milestone_message: Optional[str] = None
        self._milestone_time: float = 0.0

        # Confirmation progress (0.0 to 1.0)
        self._confirm_progress: float = 0.0

        # State before pause
        self._pre_pause_state: Optional[GameState] = None

        # Track if calibration has been done (default True for backward compat;
        # main.py sets to False and manages the calibration flow explicitly)
        self._calibrated: bool = True

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def score(self) -> int:
        return self._score

    @property
    def high_score(self) -> int:
        return self._high_score

    @property
    def lives(self) -> int:
        return self._lives

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def best_streak(self) -> int:
        return self._best_streak

    @property
    def round_number(self) -> int:
        return self._round_number

    @property
    def current_target(self) -> Optional[Target]:
        return self._current_target

    @property
    def confirm_progress(self) -> float:
        return self._confirm_progress

    @property
    def milestone_message(self) -> Optional[str]:
        return self._milestone_message

    @property
    def config(self) -> DifficultyConfig:
        return self._config

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @calibrated.setter
    def calibrated(self, value: bool):
        self._calibrated = value

    def get_stats(self) -> GameStats:
        """Return current game statistics."""
        return GameStats(
            score=self._score,
            high_score=self._high_score,
            lives=self._lives,
            streak=self._streak,
            best_streak=self._best_streak,
            round_number=self._round_number,
            difficulty_name=self._config.name,
        )

    def get_round_time(self) -> float:
        """Get current round time limit based on score."""
        reduction = self._score * self._config.time_reduction_per_point
        round_time = self._config.round_time - reduction
        return max(round_time, self._config.min_round_time)

    def get_confirmation_time(self) -> float:
        """Get current confirmation time (decreases slightly with score)."""
        base = self._config.confirmation_time
        reduction = self._score * 0.005
        return max(base - reduction, self._config.confirmation_time_min)

    def get_time_remaining(self, current_time: float) -> float:
        """Get seconds remaining in current round."""
        if self._state not in (GameState.ACTIVE, GameState.CONFIRMING):
            return 0.0
        elapsed = current_time - self._round_start_time
        return max(0.0, self.get_round_time() - elapsed)

    def get_time_fraction(self, current_time: float) -> float:
        """Get fraction of time remaining (1.0 = full, 0.0 = expired)."""
        round_time = self.get_round_time()
        if round_time <= 0:
            return 0.0
        remaining = self.get_time_remaining(current_time)
        return remaining / round_time

    def enter_difficulty_select(self, timestamp: float):
        """Transition to difficulty selection screen."""
        if self._state in (GameState.START_SCREEN, GameState.GAME_OVER):
            self._transition_to(GameState.DIFFICULTY_SELECT, timestamp)

    def select_difficulty(self, config: DifficultyConfig, high_score: int, timestamp: float):
        """Apply a difficulty selection and return to start screen."""
        self._config = config
        self._lives = config.lives
        self._high_score = high_score
        self._transition_to(GameState.START_SCREEN, timestamp)

    def enter_calibration(self, timestamp: float):
        """Transition to calibration state."""
        if self._state in (
            GameState.START_SCREEN, GameState.GAME_OVER,
            GameState.DIFFICULTY_SELECT, GameState.CALIBRATING,
        ):
            self._transition_to(GameState.CALIBRATING, timestamp)

    def calibration_complete(self, timestamp: float):
        """Calibration finished successfully - transition to countdown."""
        self._calibrated = True
        if self._state == GameState.CALIBRATING:
            # Reset game stats for fresh start
            self._score = 0
            self._lives = self._config.lives
            self._streak = 0
            self._best_streak = 0
            self._round_number = 0
            self._current_target = None
            self._previous_target = None
            self._last_milestone_score = 0
            self._milestone_message = None
            self._transition_to(GameState.COUNTDOWN, timestamp)

    def calibration_failed(self, timestamp: float):
        """Calibration failed - return to start screen."""
        if self._state == GameState.CALIBRATING:
            self._transition_to(GameState.START_SCREEN, timestamp)

    def start_game(self, timestamp: float):
        """Start a new game from the start screen.

        If not already calibrated, go to CALIBRATING instead of COUNTDOWN.
        """
        if self._state not in (GameState.START_SCREEN, GameState.GAME_OVER):
            return

        self._score = 0
        self._lives = self._config.lives
        self._streak = 0
        self._best_streak = 0
        self._round_number = 0
        self._current_target = None
        self._previous_target = None
        self._last_milestone_score = 0
        self._milestone_message = None

        if not self._calibrated:
            self._transition_to(GameState.CALIBRATING, timestamp)
        else:
            self._transition_to(GameState.COUNTDOWN, timestamp)

    def toggle_pause(self, timestamp: float):
        """Pause or resume the game."""
        if self._state == GameState.PAUSE:
            # Resume to pre-pause state
            if self._pre_pause_state is not None:
                self._state = self._pre_pause_state
                # Adjust round start time to account for pause duration
                pause_duration = timestamp - self._state_start_time
                self._round_start_time += pause_duration
                if self._pre_pause_state == GameState.CONFIRMING:
                    self._confirm_start_time += pause_duration
                self._pre_pause_state = None
        elif self._state in (GameState.ACTIVE, GameState.CONFIRMING, GameState.COUNTDOWN):
            self._pre_pause_state = self._state
            self._transition_to(GameState.PAUSE, timestamp)

    def update(self, player_yaw: str, player_expression: str, timestamp: float):
        """Update game state each frame.

        Args:
            player_yaw: Current confirmed yaw bucket ("left", "forward", "right")
            player_expression: Current confirmed expression
            timestamp: Current time in seconds
        """
        # Clear milestone message after display time
        if self._milestone_message and timestamp - self._milestone_time > 1.5:
            self._milestone_message = None

        if self._state == GameState.COUNTDOWN:
            self._update_countdown(timestamp)
        elif self._state == GameState.ACTIVE:
            self._update_active(player_yaw, player_expression, timestamp)
        elif self._state == GameState.CONFIRMING:
            self._update_confirming(player_yaw, player_expression, timestamp)
        elif self._state == GameState.SUCCESS:
            self._update_success(timestamp)
        elif self._state == GameState.MISS:
            self._update_miss(timestamp)

    def _update_countdown(self, timestamp: float):
        """Handle countdown state."""
        elapsed = timestamp - self._state_start_time
        if elapsed >= self._config.countdown_duration:
            self._start_new_round(timestamp)

    def get_countdown_value(self, timestamp: float) -> int:
        """Get current countdown number (3, 2, 1)."""
        elapsed = timestamp - self._state_start_time
        remaining = self._config.countdown_duration - elapsed
        return max(1, int(remaining) + 1)

    def _update_active(self, player_yaw: str, player_expression: str, timestamp: float):
        """Handle active round - check for match or timeout."""
        # Check timeout
        if self.get_time_remaining(timestamp) <= 0:
            self._handle_miss(timestamp)
            return

        # Check if player matches target
        if self._current_target and self._current_target.matches(player_yaw, player_expression):
            self._confirm_start_time = timestamp
            self._confirm_progress = 0.0
            self._transition_to(GameState.CONFIRMING, timestamp)

    def _update_confirming(self, player_yaw: str, player_expression: str, timestamp: float):
        """Handle confirmation - player must hold pose."""
        # Check timeout first
        if self.get_time_remaining(timestamp) <= 0:
            self._handle_miss(timestamp)
            return

        # Check if player still matches
        if not (self._current_target and self._current_target.matches(player_yaw, player_expression)):
            # Player broke the pose, back to active
            self._confirm_progress = 0.0
            self._transition_to(GameState.ACTIVE, timestamp)
            # Preserve round timing
            return

        # Update confirmation progress
        elapsed = timestamp - self._confirm_start_time
        confirmation_time = self.get_confirmation_time()
        self._confirm_progress = min(1.0, elapsed / confirmation_time)

        if elapsed >= confirmation_time:
            self._handle_success(timestamp)

    def _update_success(self, timestamp: float):
        """Handle success feedback display."""
        elapsed = timestamp - self._state_start_time
        if elapsed >= self._config.success_display_time:
            self._start_new_round(timestamp)

    def _update_miss(self, timestamp: float):
        """Handle miss feedback display."""
        elapsed = timestamp - self._state_start_time
        if elapsed >= self._config.miss_display_time:
            if self._lives <= 0:
                self._transition_to(GameState.GAME_OVER, timestamp)
            else:
                self._start_new_round(timestamp)

    def _handle_success(self, timestamp: float):
        """Process a successful match."""
        self._score += 1
        self._streak += 1
        self._best_streak = max(self._best_streak, self._streak)

        # Streak bonus
        if self._streak >= self._config.streak_bonus_threshold:
            self._score += self._config.streak_bonus_points

        # Update high score
        if self._score > self._high_score:
            self._high_score = self._score

        # Check for speed milestone
        milestone_interval = self._config.speed_milestone_interval
        if (self._score // milestone_interval) > (self._last_milestone_score // milestone_interval):
            self._milestone_message = "SPEED UP!"
            self._milestone_time = timestamp
            self._last_milestone_score = self._score

        self._confirm_progress = 0.0
        self._transition_to(GameState.SUCCESS, timestamp)

    def _handle_miss(self, timestamp: float):
        """Process a missed round (timeout)."""
        self._lives -= 1
        self._streak = 0
        self._confirm_progress = 0.0
        self._transition_to(GameState.MISS, timestamp)

    def _start_new_round(self, timestamp: float):
        """Select a new target and begin the round."""
        self._round_number += 1
        self._previous_target = self._current_target
        self._current_target = self._select_target()
        self._round_start_time = timestamp
        self._confirm_progress = 0.0
        self._transition_to(GameState.ACTIVE, timestamp)

    def _select_target(self) -> Target:
        """Select a new target that differs meaningfully from the previous one.

        Avoids immediate repetition. Prefers targets with maximum difference.
        """
        all_targets = [Target(yaw, expr) for yaw, expr in ALL_TARGETS]

        if self._previous_target is None:
            return random.choice(all_targets)

        # Filter out exact repeat
        candidates = [t for t in all_targets if t != self._previous_target]

        if not candidates:
            return random.choice(all_targets)

        # Prefer targets that differ maximally (both yaw AND expression differ)
        max_diff_targets = [
            t for t in candidates
            if t.difference_from(self._previous_target) >= 2
        ]

        if max_diff_targets:
            return random.choice(max_diff_targets)

        # Fall back to any target that differs by at least min_target_difference
        min_diff = self._config.min_target_difference
        good_targets = [
            t for t in candidates
            if t.difference_from(self._previous_target) >= min_diff
        ]

        if good_targets:
            return random.choice(good_targets)

        return random.choice(candidates)

    def _transition_to(self, new_state: GameState, timestamp: float):
        """Transition to a new game state."""
        self._state = new_state
        self._state_start_time = timestamp

    def set_high_score(self, high_score: int):
        """Set the high score (loaded from persistence)."""
        self._high_score = high_score

    def set_config(self, config: DifficultyConfig, high_score: int = 0):
        """Change difficulty configuration (only from start screen)."""
        self._config = config
        self._lives = config.lives
        self._high_score = high_score
