"""Rendering module for Hole in the Wall: Face Challenge.

Handles HUD display, sprite compositing, timer bar, feedback overlays,
and background switching. Uses OpenCV drawing primitives.
Responsive layout system supporting 640x480 to 1920x1080+.

UI improvements:
- Partial pose match indicators (checkmark/X for direction and expression)
- Pulsing confirmation bar effect
- 'Pose detected - hold it!' status during CONFIRMING state
- Enhanced start screen with prominent title
- Improved game-over screen with stats panel
"""

import os
import time as _time
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

from config import (
    SPRITE_SIZE, TIMER_BAR_HEIGHT, HUD_FONT_SCALE, HUD_COLOR,
    HUD_SHADOW_COLOR, FEEDBACK_FONT_SCALE, BG_WEBCAM, BG_DARK, BG_CHROMA,
    BG_MODES, YAW_LABELS, EXPRESSION_LABELS, DIFFICULTY_ORDER,
)
from classifier import CalibrationStatus
from game_logic import GameState, GameStats, Target
from sprite_generator import load_sprite, ensure_sprites_exist, SPRITE_DIR


# Visual theme colors (BGR format)
COLOR_CYAN = (255, 221, 0)       # #00DDFF
COLOR_GREEN = (136, 255, 0)     # #00FF88
COLOR_RED = (85, 68, 255)       # #FF4455
COLOR_YELLOW = (0, 221, 255)    # #FFDD00
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY_LIGHT = (200, 200, 200)
COLOR_GRAY = (140, 140, 140)
COLOR_GRAY_DARK = (80, 80, 80)
COLOR_BG_DARK = (35, 30, 25)
COLOR_PANEL = (50, 45, 40)
COLOR_PANEL_BORDER = (80, 75, 70)
COLOR_ORANGE = (0, 165, 255)    # #FFA500


def alpha_composite(
    background: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Alpha-composite a BGRA overlay onto a BGR background at position (x, y)."""
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return background

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_region = overlay[oy1:oy2, ox1:ox2]
    bg_region = background[y1:y2, x1:x2]

    alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
    fg = overlay_region[:, :, :3].astype(np.float32)
    bg = bg_region.astype(np.float32)

    blended = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    background[y1:y2, x1:x2] = blended

    return background


def put_text_shadowed(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = HUD_FONT_SCALE,
    color: Tuple[int, int, int] = HUD_COLOR,
    thickness: int = 2,
):
    """Draw text with a dark shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), font, scale, HUD_SHADOW_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def put_text_centered(
    img: np.ndarray,
    text: str,
    cy: int,
    scale: float = 1.0,
    color: Tuple[int, int, int] = HUD_COLOR,
    thickness: int = 2,
):
    """Draw text centered horizontally at vertical position cy."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - text_w) // 2
    put_text_shadowed(img, text, (x, cy), scale, color, thickness)


def draw_panel(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    fill_color: Tuple[int, int, int] = COLOR_PANEL,
    border_color: Tuple[int, int, int] = COLOR_PANEL_BORDER,
    alpha: float = 0.8,
):
    """Draw a semi-transparent panel with border."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), fill_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)


class Renderer:
    """Handles all rendering for the game with responsive layout."""

    def __init__(self, sprite_dir: str = SPRITE_DIR):
        self._sprite_dir = sprite_dir
        self._sprites: Dict[str, np.ndarray] = {}
        self._load_sprites()
        self._start_time = _time.time()

    def _load_sprites(self):
        """Load all sprites into memory."""
        ensure_sprites_exist(self._sprite_dir)
        for yaw in YAW_LABELS:
            for expr in EXPRESSION_LABELS:
                key = f"{yaw}_{expr}"
                self._sprites[key] = load_sprite(yaw, expr, self._sprite_dir)

    def _scale(self, value: float, frame_h: int) -> int:
        """Scale a base-640x480 value to current frame size."""
        return int(value * frame_h / 480.0)

    def _pulse(self, speed: float = 2.0) -> float:
        """Return a pulsing value between 0.0 and 1.0."""
        import math
        t = _time.time() - self._start_time
        return (math.sin(t * speed * math.pi) + 1.0) / 2.0

    def get_sprite(self, yaw: str, expression: str) -> np.ndarray:
        """Get a sprite by yaw and expression."""
        key = f"{yaw}_{expression}"
        if key in self._sprites:
            return self._sprites[key]
        return load_sprite(yaw, expression, self._sprite_dir)

    def render_frame(
        self,
        webcam_frame: Optional[np.ndarray],
        game_state: GameState,
        stats: GameStats,
        target: Optional[Target],
        player_yaw: str,
        player_expression: str,
        time_fraction: float,
        confirm_progress: float,
        bg_mode: int,
        countdown_value: int = 3,
        milestone_message: Optional[str] = None,
        frame_size: Tuple[int, int] = (640, 480),
        calibration_status: CalibrationStatus = CalibrationStatus.IDLE,
        calibration_progress: float = 0.0,
        debug_info: Optional[Dict] = None,
        show_debug: bool = False,
        face_status: str = "",
        difficulty_idx: int = 1,
    ) -> np.ndarray:
        """Render a complete frame based on game state."""
        w, h = frame_size
        frame = self._create_background(webcam_frame, bg_mode, w, h)

        if game_state == GameState.START_SCREEN:
            self._render_start_screen(frame, stats)
        elif game_state == GameState.DIFFICULTY_SELECT:
            self._render_difficulty_select(frame, stats, difficulty_idx)
        elif game_state == GameState.CALIBRATING:
            self._render_calibrating(frame, calibration_status, calibration_progress)
        elif game_state == GameState.COUNTDOWN:
            self._render_countdown(frame, countdown_value)
        elif game_state in (GameState.ACTIVE, GameState.CONFIRMING):
            self._render_active(frame, target, player_yaw, player_expression,
                              time_fraction, confirm_progress, stats,
                              is_confirming=(game_state == GameState.CONFIRMING))
        elif game_state == GameState.SUCCESS:
            self._render_success(frame, stats)
        elif game_state == GameState.MISS:
            self._render_miss(frame, stats)
        elif game_state == GameState.PAUSE:
            self._render_pause(frame, stats)
        elif game_state == GameState.GAME_OVER:
            self._render_game_over(frame, stats)

        # Face status message during gameplay or calibration
        if face_status and game_state in (GameState.ACTIVE, GameState.CONFIRMING, GameState.CALIBRATING):
            self._render_face_status(frame, face_status)

        # Milestone message
        if milestone_message:
            put_text_centered(frame, milestone_message, h // 2 + self._scale(80, h), 1.2, COLOR_YELLOW, 3)

        # Debug overlay
        if show_debug and debug_info:
            self._render_debug_overlay(frame, debug_info)

        return frame

    def _create_background(
        self, webcam_frame: Optional[np.ndarray], bg_mode: int, w: int, h: int
    ) -> np.ndarray:
        """Create background based on mode with subtle gradient."""
        if bg_mode == BG_WEBCAM and webcam_frame is not None:
            frame = cv2.resize(webcam_frame, (w, h))
            frame = (frame * 0.6).astype(np.uint8)
        elif bg_mode == BG_CHROMA:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = (0, 177, 64)
        else:
            # Dark background with subtle vertical gradient
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            for row in range(h):
                t = row / max(h - 1, 1)
                b = int(25 + t * 15)
                g = int(25 + t * 10)
                r = int(20 + t * 10)
                frame[row, :] = (b, g, r)
        return frame


    def _render_start_screen(self, frame: np.ndarray, stats: GameStats):
        """Render start screen with prominent title and instructions."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Title panel - larger and more prominent
        panel_h = s(200)
        panel_y = s(40)
        draw_panel(frame, w // 8, panel_y, w * 3 // 4, panel_h, alpha=0.7)

        # Game title with double-thickness for prominence
        title_y = panel_y + s(60)
        put_text_centered(frame, "HOLE IN THE WALL", title_y, 1.4, COLOR_WHITE, 3)

        # Subtitle
        subtitle_y = panel_y + s(100)
        put_text_centered(frame, "FACE CHALLENGE", subtitle_y, 1.0, COLOR_CYAN, 2)

        # Decorative line under title
        line_y = panel_y + s(130)
        line_w = s(200)
        cv2.line(frame, (w // 2 - line_w, line_y), (w // 2 + line_w, line_y), COLOR_CYAN, 1, cv2.LINE_AA)

        # Version/mode indicator
        mode_y = panel_y + s(160)
        put_text_centered(frame, f"Mode: {stats.difficulty_name}", mode_y, 0.6, COLOR_GREEN, 1)

        # Info section
        info_y = panel_y + panel_h + s(40)
        put_text_centered(frame, f"High Score: {stats.high_score}", info_y, 0.8, COLOR_YELLOW, 2)

        # Main controls panel
        ctrl_panel_y = info_y + s(50)
        ctrl_panel_h = s(120)
        draw_panel(frame, w // 6, ctrl_panel_y, w * 2 // 3, ctrl_panel_h,
                  fill_color=(40, 35, 30), alpha=0.6)

        # Pulsing "SPACE to start" text
        pulse = self._pulse(1.5)
        start_color = (
            int(COLOR_WHITE[0] * (0.6 + 0.4 * pulse)),
            int(COLOR_WHITE[1] * (0.6 + 0.4 * pulse)),
            int(COLOR_WHITE[2] * (0.6 + 0.4 * pulse)),
        )
        put_text_centered(frame, "[ SPACE ] to start", ctrl_panel_y + s(35), 0.8, start_color, 2)
        put_text_centered(frame, "D = difficulty  |  C = calibrate  |  B = background",
                         ctrl_panel_y + s(70), 0.5, COLOR_GRAY_LIGHT, 1)
        put_text_centered(frame, "F = fullscreen  |  V = debug  |  Q = quit",
                         ctrl_panel_y + s(95), 0.5, COLOR_GRAY, 1)

    def _render_difficulty_select(self, frame: np.ndarray, stats: GameStats, difficulty_idx: int):
        """Render difficulty selection screen."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        put_text_centered(frame, "SELECT DIFFICULTY", s(80), 1.2, COLOR_CYAN, 3)

        # Difficulty options
        for i, name in enumerate(DIFFICULTY_ORDER):
            y_pos = s(160) + i * s(70)
            is_selected = (i == difficulty_idx)

            if is_selected:
                # Highlighted panel with pulse effect
                pulse = self._pulse(2.0)
                border_color = (
                    int(COLOR_CYAN[0] * (0.7 + 0.3 * pulse)),
                    int(COLOR_CYAN[1] * (0.7 + 0.3 * pulse)),
                    int(COLOR_CYAN[2] * (0.7 + 0.3 * pulse)),
                )
                panel_w = s(300)
                panel_x = (w - panel_w) // 2
                draw_panel(frame, panel_x, y_pos - s(20), panel_w, s(50),
                          fill_color=(60, 50, 40), border_color=border_color, alpha=0.7)
                color = COLOR_CYAN
                prefix = "> "
            else:
                color = COLOR_GRAY_LIGHT
                prefix = "  "

            put_text_centered(frame, f"{prefix}{name}", y_pos + s(10), 0.8, color, 2)

        # Instructions
        put_text_centered(frame, "D = cycle  |  SPACE = confirm", h - s(60), 0.6, COLOR_GRAY, 1)

    def _render_calibrating(self, frame: np.ndarray, status: CalibrationStatus, progress: float):
        """Render calibration screen with progress and status."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Title
        put_text_centered(frame, "CALIBRATION", s(80), 1.2, COLOR_CYAN, 3)
        put_text_centered(frame, "Hold a neutral face and look at camera", s(120), 0.6, COLOR_GRAY_LIGHT, 1)

        # Progress bar
        bar_w = w * 2 // 3
        bar_h = s(25)
        bar_x = (w - bar_w) // 2
        bar_y = h // 2 - bar_h // 2

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_GRAY_DARK, -1)
        # Progress fill
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            color = COLOR_GREEN if progress >= 1.0 else COLOR_CYAN
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_PANEL_BORDER, 1)

        # Percentage text
        pct_text = f"{int(progress * 100)}%"
        put_text_centered(frame, pct_text, bar_y + bar_h + s(30), 0.7, COLOR_WHITE, 2)

        # Status message
        status_y = bar_y + bar_h + s(70)
        status_msg, status_color = self._get_calibration_message(status)
        put_text_centered(frame, status_msg, status_y, 0.7, status_color, 2)

        # Bottom hint
        put_text_centered(frame, "ESC to cancel", h - s(30), 0.5, COLOR_GRAY, 1)

    def _get_calibration_message(self, status: CalibrationStatus) -> Tuple[str, Tuple[int, int, int]]:
        """Get display message and color for calibration status."""
        messages = {
            CalibrationStatus.IDLE: ("Preparing...", COLOR_GRAY_LIGHT),
            CalibrationStatus.COLLECTING: ("Calibrating... hold still", COLOR_GREEN),
            CalibrationStatus.WAITING_FOR_FACE: ("No face detected - look at camera", COLOR_YELLOW),
            CalibrationStatus.FACE_TOO_SMALL: ("Move closer to camera", COLOR_YELLOW),
            CalibrationStatus.FACE_TOO_LARGE: ("Move farther from camera", COLOR_YELLOW),
            CalibrationStatus.FACE_OFF_CENTER: ("Center your face in frame", COLOR_YELLOW),
            CalibrationStatus.FACE_NOT_FORWARD: ("Face forward", COLOR_YELLOW),
            CalibrationStatus.NOT_STILL: ("Hold still!", COLOR_ORANGE),
            CalibrationStatus.COMPLETE: ("Calibration complete!", COLOR_GREEN),
            CalibrationStatus.FAILED: ("Calibration failed - try again", COLOR_RED),
        }
        return messages.get(status, ("", COLOR_WHITE))

    def _render_countdown(self, frame: np.ndarray, value: int):
        """Render countdown number with color coding and pulse."""
        h, w = frame.shape[:2]
        colors = {3: COLOR_RED, 2: COLOR_YELLOW, 1: COLOR_GREEN}
        color = colors.get(value, COLOR_WHITE)

        # Pulse effect on the number
        pulse = self._pulse(3.0)
        scale = 3.0 + pulse * 0.3
        put_text_centered(frame, str(value), h // 2 + 30, scale, color, 5)
        put_text_centered(frame, "GET READY!", h // 2 - self._scale(60, h), 0.8, COLOR_GRAY_LIGHT, 2)


    def _render_active(
        self,
        frame: np.ndarray,
        target: Optional[Target],
        player_yaw: str,
        player_expression: str,
        time_fraction: float,
        confirm_progress: float,
        stats: GameStats,
        is_confirming: bool = False,
    ):
        """Render active game state with target, timer, HUD, and match indicators."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Draw target sprite (large, centered top)
        if target:
            sprite = self.get_sprite(target.yaw, target.expression)
            # Scale sprite based on frame size
            target_size = min(s(220), w // 3)
            sprite_resized = cv2.resize(sprite, (target_size, target_size))
            sx = (w - target_size) // 2
            sy = s(30)
            alpha_composite(frame, sprite_resized, sx, sy)

            # Target label
            target_text = f"Match: {target.yaw.upper()} + {target.expression.upper()}"
            put_text_centered(frame, target_text, sy + target_size + s(25), 0.6, COLOR_YELLOW, 1)

            # Partial match indicators
            self._render_match_indicators(frame, target, player_yaw, player_expression,
                                         sx, sy, target_size)

        # Draw player's current detection (small, bottom-right)
        player_sprite = self.get_sprite(player_yaw, player_expression)
        small_size = s(100)
        player_sprite_small = cv2.resize(player_sprite, (small_size, small_size))
        px = w - small_size - s(15)
        py = h - small_size - s(60)
        alpha_composite(frame, player_sprite_small, px, py)

        # Player label
        player_text = f"You: {player_yaw} + {player_expression}"
        put_text_shadowed(frame, player_text, (px - s(10), py + small_size + s(20)), 0.45, COLOR_CYAN)

        # Timer bar
        self._render_timer_bar(frame, time_fraction, w, h)

        # Confirmation progress bar (with pulsing effect)
        if confirm_progress > 0:
            self._render_confirm_bar(frame, confirm_progress, w, h, is_confirming)

        # "Pose detected - hold it!" message during confirming
        if is_confirming:
            pulse = self._pulse(4.0)
            msg_color = (
                int(COLOR_GREEN[0] * (0.7 + 0.3 * pulse)),
                int(COLOR_GREEN[1] * (0.7 + 0.3 * pulse)),
                int(COLOR_GREEN[2] * (0.7 + 0.3 * pulse)),
            )
            put_text_centered(frame, "Pose detected - hold it!", h - s(95), 0.6, msg_color, 2)

        # HUD
        self._render_hud(frame, stats)

    def _render_match_indicators(
        self,
        frame: np.ndarray,
        target: Target,
        player_yaw: str,
        player_expression: str,
        sprite_x: int,
        sprite_y: int,
        sprite_size: int,
    ):
        """Render checkmarks/X indicators showing which parts of pose match."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Position indicators to the right of the target sprite
        ind_x = sprite_x + sprite_size + s(15)
        ind_y_dir = sprite_y + sprite_size // 3
        ind_y_expr = sprite_y + sprite_size * 2 // 3

        # Direction match indicator
        dir_matches = (target.yaw == player_yaw)
        dir_color = COLOR_GREEN if dir_matches else COLOR_RED
        dir_symbol = "OK" if dir_matches else "X"
        dir_label = f"Dir: {dir_symbol}"
        put_text_shadowed(frame, dir_label, (ind_x, ind_y_dir), 0.45, dir_color, 1)

        # Draw a small checkmark or X icon
        icon_x = ind_x - s(12)
        if dir_matches:
            # Checkmark
            cv2.line(frame, (icon_x, ind_y_dir - 2), (icon_x + 4, ind_y_dir + 3), COLOR_GREEN, 2, cv2.LINE_AA)
            cv2.line(frame, (icon_x + 4, ind_y_dir + 3), (icon_x + 10, ind_y_dir - 5), COLOR_GREEN, 2, cv2.LINE_AA)
        else:
            # X mark
            cv2.line(frame, (icon_x, ind_y_dir - 5), (icon_x + 8, ind_y_dir + 3), COLOR_RED, 2, cv2.LINE_AA)
            cv2.line(frame, (icon_x + 8, ind_y_dir - 5), (icon_x, ind_y_dir + 3), COLOR_RED, 2, cv2.LINE_AA)

        # Expression match indicator
        expr_matches = (target.expression == player_expression)
        expr_color = COLOR_GREEN if expr_matches else COLOR_RED
        expr_symbol = "OK" if expr_matches else "X"
        expr_label = f"Expr: {expr_symbol}"
        put_text_shadowed(frame, expr_label, (ind_x, ind_y_expr), 0.45, expr_color, 1)

        # Draw icon for expression
        icon_x2 = ind_x - s(12)
        if expr_matches:
            cv2.line(frame, (icon_x2, ind_y_expr - 2), (icon_x2 + 4, ind_y_expr + 3), COLOR_GREEN, 2, cv2.LINE_AA)
            cv2.line(frame, (icon_x2 + 4, ind_y_expr + 3), (icon_x2 + 10, ind_y_expr - 5), COLOR_GREEN, 2, cv2.LINE_AA)
        else:
            cv2.line(frame, (icon_x2, ind_y_expr - 5), (icon_x2 + 8, ind_y_expr + 3), COLOR_RED, 2, cv2.LINE_AA)
            cv2.line(frame, (icon_x2 + 8, ind_y_expr - 5), (icon_x2, ind_y_expr + 3), COLOR_RED, 2, cv2.LINE_AA)

    def _render_timer_bar(self, frame: np.ndarray, fraction: float, width: int, height: int):
        """Render shrinking timer bar at top of screen."""
        bar_h = self._scale(TIMER_BAR_HEIGHT, height)
        bar_width = int(width * fraction)

        # Color transitions: green -> yellow -> red
        if fraction > 0.5:
            color = COLOR_GREEN
        elif fraction > 0.25:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED

        cv2.rectangle(frame, (0, 0), (bar_width, bar_h), color, -1)
        cv2.rectangle(frame, (0, 0), (width, bar_h), COLOR_GRAY_DARK, 1)

    def _render_confirm_bar(self, frame: np.ndarray, progress: float, width: int, height: int,
                           is_confirming: bool = True):
        """Render confirmation progress bar at bottom with pulsing effect."""
        s = lambda v: self._scale(v, height)
        bar_y = height - s(40)
        bar_h = s(15)
        bar_margin = s(50)
        bar_max_w = width - 2 * bar_margin
        bar_w = int(bar_max_w * progress)

        # Background
        cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_max_w, bar_y + bar_h), COLOR_GRAY_DARK, -1)

        # Progress with pulsing brightness
        pulse = self._pulse(4.0) if is_confirming else 0.5
        if progress >= 1.0:
            color = COLOR_GREEN
        else:
            # Pulsing cyan
            color = (
                int(COLOR_CYAN[0] * (0.7 + 0.3 * pulse)),
                int(COLOR_CYAN[1] * (0.7 + 0.3 * pulse)),
                int(COLOR_CYAN[2] * (0.7 + 0.3 * pulse)),
            )

        if bar_w > 0:
            cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_w, bar_y + bar_h), color, -1)

        # Pulsing border
        border_color = (
            int(COLOR_PANEL_BORDER[0] + (COLOR_CYAN[0] - COLOR_PANEL_BORDER[0]) * pulse * 0.3),
            int(COLOR_PANEL_BORDER[1] + (COLOR_CYAN[1] - COLOR_PANEL_BORDER[1]) * pulse * 0.3),
            int(COLOR_PANEL_BORDER[2] + (COLOR_CYAN[2] - COLOR_PANEL_BORDER[2]) * pulse * 0.3),
        )
        cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_max_w, bar_y + bar_h), border_color, 1)

        # Label
        label = "HOLD POSE..." if is_confirming else "CONFIRMING..."
        put_text_shadowed(frame, label, (bar_margin, bar_y - s(5)), 0.5, COLOR_GREEN)


    def _render_hud(self, frame: np.ndarray, stats: GameStats):
        """Render score, lives, streak, difficulty."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)
        y = s(35)
        x = s(10)

        put_text_shadowed(frame, f"Score: {stats.score}", (x, y), 0.6, COLOR_WHITE)
        put_text_shadowed(frame, f"Best: {stats.high_score}", (x, y + s(28)), 0.5, COLOR_GRAY_LIGHT)

        # Lives as hearts/stars
        lives_text = "Lives: " + "* " * stats.lives
        lives_color = COLOR_RED if stats.lives <= 1 else COLOR_CYAN
        put_text_shadowed(frame, lives_text, (x, y + s(56)), 0.6, lives_color)

        if stats.streak > 0:
            streak_color = COLOR_YELLOW if stats.streak >= 3 else COLOR_GRAY_LIGHT
            put_text_shadowed(frame, f"Streak: {stats.streak}", (x, y + s(84)), 0.5, streak_color)

        # Difficulty (top right)
        put_text_shadowed(frame, stats.difficulty_name, (w - s(100), y), 0.5, COLOR_GREEN)

    def _render_success(self, frame: np.ndarray, stats: GameStats):
        """Render success feedback with animation-like effect."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Greenish tint overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 40, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        put_text_centered(frame, "MATCHED!", h // 2, 1.5, COLOR_GREEN, 3)
        put_text_centered(frame, f"+1 point  (Score: {stats.score})", h // 2 + s(50), 0.8, COLOR_WHITE, 2)

        if stats.streak >= 3:
            put_text_centered(frame, f"STREAK x{stats.streak}!", h // 2 + s(90), 0.7, COLOR_YELLOW, 2)

        self._render_hud(frame, stats)

    def _render_miss(self, frame: np.ndarray, stats: GameStats):
        """Render miss feedback."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Reddish tint overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 40), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        put_text_centered(frame, "MISS!", h // 2, 1.5, COLOR_RED, 3)
        lives_color = COLOR_RED if stats.lives <= 1 else COLOR_YELLOW
        put_text_centered(frame, f"Lives remaining: {stats.lives}", h // 2 + s(50), 0.8, lives_color, 2)

        self._render_hud(frame, stats)

    def _render_pause(self, frame: np.ndarray, stats: GameStats):
        """Render pause screen with darkened overlay."""
        h, w = frame.shape[:2]

        # Semi-transparent dark overlay
        frame[:] = (frame * 0.3).astype(np.uint8)

        # Pause panel
        panel_w = w // 2
        panel_h = h // 4
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2
        draw_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.9)

        put_text_centered(frame, "PAUSED", h // 2 - 10, 1.5, COLOR_WHITE, 3)
        put_text_centered(frame, "P to resume  |  Q to quit", h // 2 + self._scale(40, h), 0.6, COLOR_GRAY_LIGHT, 1)

        self._render_hud(frame, stats)

    def _render_game_over(self, frame: np.ndarray, stats: GameStats):
        """Render game over screen with final scores and stats."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Dark overlay for emphasis
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 5, 20), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # GAME OVER title with red glow effect
        put_text_centered(frame, "GAME OVER", s(100), 1.8, COLOR_RED, 4)

        # Score panel
        panel_w = w * 2 // 3
        panel_h = s(180)
        panel_x = (w - panel_w) // 2
        panel_y = s(130)
        draw_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.75)

        # Final score (large)
        put_text_centered(frame, f"Final Score: {stats.score}", panel_y + s(45), 1.1, COLOR_WHITE, 2)

        # Stats
        put_text_centered(frame, f"Best Streak: {stats.best_streak}", panel_y + s(85), 0.7, COLOR_GRAY_LIGHT, 2)
        put_text_centered(frame, f"Rounds: {stats.round_number}", panel_y + s(115), 0.7, COLOR_GRAY_LIGHT, 2)

        # High score notification
        if stats.score >= stats.high_score and stats.score > 0:
            pulse = self._pulse(2.0)
            hs_color = (
                int(COLOR_YELLOW[0] * (0.7 + 0.3 * pulse)),
                int(COLOR_YELLOW[1] * (0.7 + 0.3 * pulse)),
                int(COLOR_YELLOW[2] * (0.7 + 0.3 * pulse)),
            )
            put_text_centered(frame, "*** NEW HIGH SCORE! ***", panel_y + s(150), 0.9, hs_color, 2)
        else:
            put_text_centered(frame, f"High Score: {stats.high_score}", panel_y + s(150), 0.7, COLOR_GRAY_LIGHT, 2)

        # Controls at bottom
        ctrl_y = h - s(80)
        put_text_centered(frame, "SPACE to restart  |  Q to quit", ctrl_y, 0.6, COLOR_GRAY_LIGHT, 1)
        put_text_centered(frame, "D = change difficulty", ctrl_y + s(30), 0.5, COLOR_GRAY, 1)

    def _render_face_status(self, frame: np.ndarray, status: str):
        """Render face status message during gameplay."""
        if not status:
            return
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Semi-transparent banner at bottom
        banner_y = h - s(80)
        banner_h = s(30)
        draw_panel(frame, 0, banner_y, w, banner_h,
                  fill_color=(0, 0, 60), border_color=COLOR_YELLOW, alpha=0.7)
        put_text_centered(frame, status, banner_y + s(20), 0.55, COLOR_YELLOW, 1)

    def _render_debug_overlay(self, frame: np.ndarray, debug_info: Dict):
        """Render debug information overlay."""
        h, w = frame.shape[:2]
        s = lambda v: self._scale(v, h)

        # Semi-transparent panel on the left
        panel_w = s(280)
        panel_h = s(250)
        draw_panel(frame, 5, h - panel_h - 5, panel_w, panel_h,
                  fill_color=(20, 20, 20), border_color=COLOR_CYAN, alpha=0.85)

        x = 15
        y_start = h - panel_h + s(15)
        line_h = s(18)
        font_scale = 0.4

        lines = [
            f"Raw Yaw: {debug_info.get('raw_yaw', 0):.1f} deg",
            f"Smoothed Yaw: {debug_info.get('smoothed_yaw', 0):.1f} deg",
            f"Direction: {debug_info.get('confirmed_direction', '?')}",
            f"Raw Expr: {debug_info.get('raw_expression', '?')}",
            f"Confirmed: {debug_info.get('confirmed_expression', '?')}",
            f"Baseline MAR: {debug_info.get('baseline_mouth_ar', 0):.4f}",
            f"Baseline Brow: {debug_info.get('baseline_brow_eye', 0):.4f}",
            f"Baseline Eye: {debug_info.get('baseline_eye_ar', 0):.4f}",
            f"Cal Status: {debug_info.get('calibration_status', '?')}",
            f"Cal Progress: {debug_info.get('calibration_progress', 0):.0%}",
            f"Face Valid: {debug_info.get('is_face_valid', False)}",
        ]

        for i, line in enumerate(lines):
            y = y_start + i * line_h
            put_text_shadowed(frame, line, (x, y), font_scale, COLOR_GREEN, 1)
