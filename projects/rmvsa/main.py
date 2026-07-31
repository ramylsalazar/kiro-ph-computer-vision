"""Main entry point for Hole in the Wall: Face Challenge.

Webcam loop with keyboard controls, integrating face tracking,
classification, game logic, rendering, and persistence.

Controls:
    SPACE  - Start / Restart / Continue
    C      - Recalibrate neutral baseline
    B      - Cycle background (webcam / dark / chroma-green)
    D      - Cycle difficulty (when not playing)
    P      - Pause / Resume
    F      - Toggle fullscreen
    V      - Toggle debug overlay
    1-4    - Preview expressions (neutral / happy / surprised / angry)
    Q/ESC  - Quit
"""

import sys
import time
import os

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from config import (
    DIFFICULTIES, DIFFICULTY_ORDER, BG_WEBCAM, BG_DARK, BG_CHROMA, BG_MODES,
    EXPRESSION_LABELS, DifficultyConfig, DEBUG_OVERLAY_KEY, FULLSCREEN_KEY,
)
from face_tracking import FaceTracker, FaceFeatures
from classifier import Classifier, FaceMeasurements, CalibrationStatus
from game_logic import GameLogic, GameState
from persistence import HighScoreManager
from renderer import Renderer
from sprite_generator import ensure_sprites_exist


WINDOW_NAME = "Hole in the Wall: Face Challenge"
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _get_screen_size() -> tuple:
    """Try to get screen resolution. Falls back to 1920x1080."""
    try:
        # Try using ctypes on Windows
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        pass

    try:
        # Try tkinter as fallback (cross-platform)
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return (w, h)
    except Exception:
        pass

    return (1920, 1080)


def main():
    """Run the game."""
    # Initialize components
    print("Initializing Face Challenge...")
    print("Generating sprites if needed...")
    ensure_sprites_exist()

    high_score_mgr = HighScoreManager()

    # Difficulty selection
    difficulty_idx = 1  # Start on Normal
    difficulty_name = DIFFICULTY_ORDER[difficulty_idx]
    config = DIFFICULTIES[difficulty_name]

    # Initialize game
    game = GameLogic(config, high_score_mgr.get(difficulty_name))
    game.calibrated = False  # Require calibration before first game
    classifier = Classifier(config)
    renderer = Renderer()

    # State flags
    bg_mode = BG_WEBCAM
    fullscreen = False
    show_debug = False
    preview_expression: str = ""

    # Get screen size with error handling
    try:
        screen_w, screen_h = _get_screen_size()
    except Exception:
        screen_w, screen_h = 1920, 1080

    # Open webcam
    print(f"Opening webcam (index {WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Initialize face tracker
    tracker = FaceTracker()

    # Window setup
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    print("Game ready! Press SPACE to start.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Face tracking (mirrored=True because we flipped the frame)
            features = tracker.process_frame(frame, mirrored=True)

            # Classification
            player_yaw = "forward"
            player_expression = "neutral"
            face_status = ""

            if features and features.landmarks_valid:
                measurements = FaceMeasurements(
                    yaw_degrees=features.yaw_degrees,
                    mouth_aspect_ratio=features.mouth_aspect_ratio,
                    mouth_corner_lift=features.mouth_corner_lift,
                    left_brow_eye_dist=features.left_brow_eye_dist,
                    right_brow_eye_dist=features.right_brow_eye_dist,
                    left_eye_aspect_ratio=features.left_eye_aspect_ratio,
                    right_eye_aspect_ratio=features.right_eye_aspect_ratio,
                )
                result = classifier.update(
                    measurements,
                    current_time,
                    face_width=features.face_width,
                    face_center_x=features.face_center_x_normalized,
                    frame_width=float(frame.shape[1]),
                )
                player_yaw = result.confirmed_yaw
                player_expression = result.confirmed_expression
            else:
                # No face detected - notify classifier
                classifier.set_no_face()
                face_status = "No face detected - look at camera"

            # Generate face status from calibration status during gameplay
            if features and features.landmarks_valid and not classifier._is_face_valid:
                cal_status = classifier.calibration_status
                if cal_status == CalibrationStatus.FACE_TOO_SMALL:
                    face_status = "Move closer to camera"
                elif cal_status == CalibrationStatus.FACE_TOO_LARGE:
                    face_status = "Move farther from camera"
                elif cal_status == CalibrationStatus.FACE_OFF_CENTER:
                    face_status = "Center your face in frame"

            # Preview override
            if preview_expression:
                player_expression = preview_expression

            # Handle calibration state transitions
            if game.state == GameState.CALIBRATING:
                if classifier.calibration_status == CalibrationStatus.COMPLETE:
                    game.calibration_complete(current_time)
                elif classifier.calibration_status == CalibrationStatus.FAILED:
                    game.calibration_failed(current_time)

            # Update game logic
            if game.state in (GameState.ACTIVE, GameState.CONFIRMING,
                             GameState.COUNTDOWN, GameState.SUCCESS, GameState.MISS):
                game.update(player_yaw, player_expression, current_time)

            # Get game info for rendering
            stats = game.get_stats()
            time_fraction = game.get_time_fraction(current_time)
            confirm_progress = game.confirm_progress
            countdown_val = game.get_countdown_value(current_time) if game.state == GameState.COUNTDOWN else 3

            # Determine output frame size
            if fullscreen:
                output_size = (screen_w, screen_h)
            else:
                output_size = (FRAME_WIDTH, FRAME_HEIGHT)

            # Render
            display_frame = renderer.render_frame(
                webcam_frame=frame,
                game_state=game.state,
                stats=stats,
                target=game.current_target,
                player_yaw=player_yaw,
                player_expression=player_expression,
                time_fraction=time_fraction,
                confirm_progress=confirm_progress,
                bg_mode=bg_mode,
                countdown_value=countdown_val,
                milestone_message=game.milestone_message,
                frame_size=output_size,
                calibration_status=classifier.calibration_status,
                calibration_progress=classifier.calibration_progress,
                debug_info=classifier.debug_info if show_debug else None,
                show_debug=show_debug,
                face_status=face_status if game.state in (GameState.ACTIVE, GameState.CONFIRMING, GameState.CALIBRATING) else "",
                difficulty_idx=difficulty_idx,
            )

            cv2.imshow(WINDOW_NAME, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == ord(' '):  # SPACE
                if game.state == GameState.START_SCREEN:
                    if classifier.is_calibrated:
                        game.calibrated = True
                        game.start_game(current_time)
                    else:
                        # Need calibration first
                        game.enter_calibration(current_time)
                        classifier.start_calibration(current_time)
                elif game.state == GameState.DIFFICULTY_SELECT:
                    # Confirm difficulty selection and go back to start
                    game.select_difficulty(config, high_score_mgr.get(difficulty_name), current_time)
                elif game.state == GameState.GAME_OVER:
                    # Save high score and restart
                    high_score_mgr.update(difficulty_name, stats.score)
                    if classifier.is_calibrated:
                        game.calibrated = True
                        game.start_game(current_time)
                    else:
                        game.enter_calibration(current_time)
                        classifier.start_calibration(current_time)

            elif key == ord('c') or key == ord('C'):  # Calibrate
                # Only allow calibration from appropriate states
                # NOT during active gameplay, confirming, or countdown
                if game.state not in (GameState.CALIBRATING, GameState.ACTIVE,
                                      GameState.CONFIRMING, GameState.COUNTDOWN):
                    game.enter_calibration(current_time)
                    classifier.start_calibration(current_time)

            elif key == ord('b') or key == ord('B'):  # Background
                bg_mode = (bg_mode + 1) % 3

            elif key == ord('d') or key == ord('D'):  # Difficulty
                if game.state == GameState.DIFFICULTY_SELECT:
                    # Cycle through difficulties within the select screen
                    difficulty_idx = (difficulty_idx + 1) % len(DIFFICULTY_ORDER)
                    difficulty_name = DIFFICULTY_ORDER[difficulty_idx]
                    config = DIFFICULTIES[difficulty_name]
                elif game.state in (GameState.START_SCREEN, GameState.GAME_OVER):
                    # Enter difficulty select mode
                    game.enter_difficulty_select(current_time)

            elif key == ord('p') or key == ord('P'):  # Pause
                game.toggle_pause(current_time)

            elif key == FULLSCREEN_KEY or key == ord('F'):  # Fullscreen
                fullscreen = not fullscreen
                try:
                    if fullscreen:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
                except cv2.error:
                    # Some OpenCV builds don't support fullscreen toggle
                    fullscreen = not fullscreen  # revert

            elif key == DEBUG_OVERLAY_KEY or key == ord('V'):  # Debug
                show_debug = not show_debug

            elif key == ord('1'):  # Preview neutral
                preview_expression = "neutral" if preview_expression != "neutral" else ""
            elif key == ord('2'):  # Preview happy
                preview_expression = "happy" if preview_expression != "happy" else ""
            elif key == ord('3'):  # Preview surprised
                preview_expression = "surprised" if preview_expression != "surprised" else ""
            elif key == ord('4'):  # Preview angry
                preview_expression = "angry" if preview_expression != "angry" else ""

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        print("\nShutting down...")
        # Save any pending high score
        if game.state == GameState.GAME_OVER or game.score > 0:
            high_score_mgr.update(difficulty_name, game.score)

        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
