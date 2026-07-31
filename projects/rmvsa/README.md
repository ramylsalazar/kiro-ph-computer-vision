# Hole in the Wall: Face Challenge

A webcam-based mini-game where you match target poses using head direction and facial expressions.

Each round shows a "hole" (target pose) combining a head direction (left / forward / right) and an expression (neutral / happy / surprised / angry). Match it before time runs out!

## Setup

Requires Python 3.10–3.12 (mediapipe constraint).

```bash
cd projects/rmvsa
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

A webcam window opens. The game guides you through calibration before gameplay begins.

## Controls

| Key | Action |
|-----|--------|
| Space | Start / Restart / Continue |
| C | Recalibrate neutral baseline |
| B | Cycle background (webcam / dark / chroma-green) |
| D | Cycle difficulty (from start or game-over screen) |
| F | Toggle fullscreen (uses native display resolution) |
| V | Toggle debug overlay (shows raw yaw, expression values, calibration state) |
| P | Pause / Resume |
| 1–4 | Preview neutral / happy / surprised / angry |
| Q | Quit |
| Esc | Quit (same as Q) |

## How to Play

1. Press SPACE on the start screen
2. Calibration runs automatically — hold a neutral face forward
3. After countdown, match the displayed target pose (direction + expression)
4. Hold the pose for the confirmation period
5. Score points and build streaks for consecutive matches
6. Lose a life when time runs out
7. Game ends when all lives are gone

## Calibration Flow

The game follows this sequence:

**Start Screen → Choose Difficulty (D) → Start Game (SPACE) → Calibration → Countdown → Game**

During calibration, hold a neutral face looking forward for approximately 1 second. A progress bar shows calibration progress, along with status messages:

- **No face detected** — move into frame
- **Move closer/farther** — adjust distance to camera
- **Center your face** — position face in center of frame
- **Face forward** — look straight at the camera
- **Hold still** — remain steady while baseline is captured
- **Calibration complete** — ready to play

Press C anytime from the menu or game-over screen to recalibrate your baseline.

## Fullscreen

Press F to toggle fullscreen using your native display resolution. All UI elements (HUD, timer, sprites, overlays) scale proportionally to fill the screen. Press F again to return to windowed mode.

The game supports any resolution from 640×480 to 4K.

## Debug Overlay

Press V to toggle the debug overlay, which displays real-time detection data:

- Raw yaw degrees and smoothed yaw value
- Confirmed direction (left / forward / right)
- Raw expression ratios and confirmed expression
- Calibrated baseline values
- Calibration status
- Face validity state

Useful for tuning thresholds in `config.py` or troubleshooting detection issues.

## Difficulty Modes

| | Easy | Normal | Hard |
|--|------|--------|------|
| Lives | 5 | 3 | 2 |
| Starting time | 5.0s | 3.8s | 2.7s |
| Speed increase | Slow | Moderate | Fast |
| Confirmation | Longer | Standard | Shorter |
| Yaw angles | Wider | Standard | Stricter |

Press D on the start/game-over screen to cycle.

## Replacing Sprites

Sprites live in `sprites/` as 200×200 BGRA PNGs:
```
left_neutral.png    forward_neutral.png    right_neutral.png
left_happy.png      forward_happy.png      right_happy.png
left_surprised.png  forward_surprised.png  right_surprised.png
left_angry.png      forward_angry.png      right_angry.png
```

Replace any with your own artwork. Missing sprites are auto-generated as programmatic placeholders featuring:

- Directional head rotation with visible nose and ears
- Expression-specific features (teeth, brow angles, blush marks)
- Direction arrows and pose labels below each sprite
- Alpha-transparent backgrounds

To regenerate all placeholders:
```bash
python sprite_generator.py
```

## Tuning Difficulty

All tunables live in `config.py` as frozen dataclasses. Key parameters:

- `round_time` / `min_round_time` — starting and minimum round duration
- `time_reduction_per_point` — how fast rounds get shorter
- `confirmation_time` — how long the player must hold the correct pose
- `expression_persistence` — how long an expression must be stable before accepting
- `YawConfig.enter_*` / `release_*` — hysteresis angles for left/right detection
- `ExpressionThresholds.*` — relative multipliers against calibrated baseline

## Testing

```bash
pytest test_game.py -v
```

Tests run without a webcam — they use synthetic face landmarks to verify:

- Physical left/right direction mapping
- Yaw sign consistency
- Yaw hysteresis boundaries
- Calibration validation
- State transition mutual exclusivity
- Responsive layout scaling
- Expression noise rejection
- Foreshortening correction
- Scale invariance
- All four expressions against baseline
- Scoring, streaks, timeouts, lives, speed scaling
- Target non-repetition
- High score persistence

## Architecture

| Module | Responsibility |
|--------|---------------|
| `config.py` | All tunables as dataclasses |
| `face_tracking.py` | MediaPipe Face Mesh + solvePnP yaw + expression ratios |
| `classifier.py` | Calibration, yaw bucketing, expression classification (pure Python) |
| `game_logic.py` | State machine, scoring, target selection (pure Python) |
| `persistence.py` | High score JSON save/load |
| `sprite_generator.py` | Programmatic placeholder sprites |
| `renderer.py` | HUD, timer, overlays, sprite compositing |
| `main.py` | Webcam loop + keyboard integration |
| `test_game.py` | Unit tests |
