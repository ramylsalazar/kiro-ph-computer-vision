"""Programmatic placeholder sprite generation.

Creates 12 attractive PNG sprites named {yaw}_{expression}.png with alpha transparency.
These are used when artwork files are missing.

Improvements:
- Proper nose that rotates with yaw
- Ears that show/hide based on direction
- V-shaped brows for angry
- Teeth for surprised (O-mouth with teeth)
- Blush marks for happy
- Hair/head shape that rotates with yaw
- Anti-aliased drawing (cv2.LINE_AA)
- Labels below sprites showing the target pose
- Gradient fills for the background circle
- Inner glow effect
"""

import os
import math
from typing import Tuple

import numpy as np
import cv2

from config import YAW_LABELS, EXPRESSION_LABELS, SPRITE_SIZE


SPRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")

# Color scheme for expressions
EXPRESSION_COLORS = {
    "neutral": (200, 200, 210),   # Light gray-blue
    "happy": (80, 200, 255),      # Warm yellow/orange in BGR
    "surprised": (255, 200, 80),  # Blue-ish cyan
    "angry": (60, 60, 230),       # Red
}

# Background gradient colors (inner, outer) per expression
BG_GRADIENT = {
    "neutral": ((70, 70, 80), (30, 30, 40)),
    "happy": ((60, 80, 90), (25, 40, 50)),
    "surprised": ((80, 70, 50), (35, 30, 25)),
    "angry": ((60, 30, 60), (30, 15, 35)),
}

# Skin tone (BGR)
SKIN_COLOR = (180, 200, 220)
SKIN_SHADOW = (140, 160, 180)

# Hair color
HAIR_COLOR = (40, 50, 70)


def _lerp_color(
    c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    """Linear interpolate between two BGR colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _draw_gradient_circle(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    inner_color: Tuple[int, int, int],
    outer_color: Tuple[int, int, int],
):
    """Draw a radial gradient filled circle onto BGRA image."""
    h, w = img.shape[:2]
    # Create distance map from center
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2).astype(np.float32)

    # Normalize distance (0 at center, 1 at radius)
    t = np.clip(dist / max(radius, 1), 0, 1)

    # Create mask for circle area
    mask = (dist <= radius).astype(np.float32)

    # Interpolate colors
    for c in range(3):
        channel = inner_color[c] * (1 - t) + outer_color[c] * t
        img[:, :, c] = np.where(mask > 0, channel.astype(np.uint8), img[:, :, c])

    # Set alpha for the circle
    # Soft edge (anti-aliasing)
    edge_softness = 2.0
    alpha_float = np.clip((radius - dist) / edge_softness, 0, 1) * 255
    img[:, :, 3] = np.maximum(img[:, :, 3], alpha_float.astype(np.uint8))


def _draw_inner_glow(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    color: Tuple[int, int, int],
    glow_width: int = 8,
):
    """Draw an inner glow effect around the circle edge."""
    h, w = img.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2).astype(np.float32)

    # Inner glow band: from (radius - glow_width) to radius
    inner_edge = radius - glow_width
    band = np.clip((dist - inner_edge) / max(glow_width, 1), 0, 1)
    in_circle = (dist <= radius).astype(np.float32)

    # Apply glow only inside circle
    glow_alpha = band * in_circle * 0.4  # 40% max opacity

    for c in range(3):
        current = img[:, :, c].astype(np.float32)
        glowed = current * (1 - glow_alpha) + color[c] * glow_alpha
        img[:, :, c] = np.clip(glowed, 0, 255).astype(np.uint8)


def _draw_head_shape(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
):
    """Draw a head/face oval with skin tone that shifts with yaw."""
    center = (cx + yaw_offset, cy)
    # Face gets narrower with yaw due to foreshortening
    face_w = radius - abs(yaw_offset) // 3
    face_h = int(radius * 1.1)
    cv2.ellipse(img, center, (face_w, face_h), 0, 0, 360, SKIN_COLOR, -1, cv2.LINE_AA)

    # Shadow on the far side of the turn
    if yaw_offset != 0:
        shadow_offset = -yaw_offset // 2
        shadow_center = (cx + yaw_offset + shadow_offset, cy)
        shadow_w = face_w // 2
        cv2.ellipse(img, shadow_center, (shadow_w, face_h), 0, 0, 360, SKIN_SHADOW, -1, cv2.LINE_AA)
        # Redraw main face on top to blend
        cv2.ellipse(img, center, (face_w - 2, face_h - 2), 0, 0, 360, SKIN_COLOR, -1, cv2.LINE_AA)


def _draw_hair(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
):
    """Draw hair that rotates with the head."""
    center_x = cx + yaw_offset
    hair_top = cy - int(radius * 1.0)
    hair_w = radius - abs(yaw_offset) // 3 + 5

    # Main hair arc on top of head
    cv2.ellipse(img, (center_x, hair_top + radius // 3),
                (hair_w, int(radius * 0.6)), 0, 180, 360, HAIR_COLOR, -1, cv2.LINE_AA)

    # Side hair (only visible on the side facing viewer)
    if yaw_offset <= 0:
        # Show left side hair
        side_x = center_x - hair_w + 5
        cv2.ellipse(img, (side_x, cy - radius // 4),
                    (8, radius // 3), 0, 0, 360, HAIR_COLOR, -1, cv2.LINE_AA)
    if yaw_offset >= 0:
        # Show right side hair
        side_x = center_x + hair_w - 5
        cv2.ellipse(img, (side_x, cy - radius // 4),
                    (8, radius // 3), 0, 0, 360, HAIR_COLOR, -1, cv2.LINE_AA)


def _draw_ears(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
    yaw: str,
):
    """Draw ears that show/hide based on direction."""
    ear_y = cy - radius // 8
    face_w = radius - abs(yaw_offset) // 3
    ear_w = 10
    ear_h = 15

    # Left ear (hidden when turning right)
    if yaw != "right":
        left_ear_x = cx + yaw_offset - face_w - 2
        cv2.ellipse(img, (left_ear_x, ear_y), (ear_w, ear_h), 0, 0, 360, SKIN_COLOR, -1, cv2.LINE_AA)
        cv2.ellipse(img, (left_ear_x, ear_y), (ear_w - 3, ear_h - 4), 0, 0, 360, SKIN_SHADOW, -1, cv2.LINE_AA)

    # Right ear (hidden when turning left)
    if yaw != "left":
        right_ear_x = cx + yaw_offset + face_w + 2
        cv2.ellipse(img, (right_ear_x, ear_y), (ear_w, ear_h), 0, 0, 360, SKIN_COLOR, -1, cv2.LINE_AA)
        cv2.ellipse(img, (right_ear_x, ear_y), (ear_w - 3, ear_h - 4), 0, 0, 360, SKIN_SHADOW, -1, cv2.LINE_AA)


def _draw_nose(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
    color: Tuple[int, int, int],
):
    """Draw a nose that rotates with yaw."""
    nose_cx = cx + yaw_offset + yaw_offset // 3
    nose_cy = cy + radius // 8
    nose_w = max(4, radius // 8 - abs(yaw_offset) // 8)

    # Simple triangular nose
    pts = np.array([
        [nose_cx, nose_cy - nose_w],
        [nose_cx - nose_w, nose_cy + nose_w],
        [nose_cx + nose_w, nose_cy + nose_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], SKIN_SHADOW, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)



def _draw_eyes(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
    expression: str,
    color: Tuple[int, int, int],
):
    """Draw eyes based on expression and yaw."""
    eye_y = cy - radius // 4
    eye_spacing = radius // 2
    left_eye_x = cx + yaw_offset - eye_spacing + yaw_offset // 4
    right_eye_x = cx + yaw_offset + eye_spacing + yaw_offset // 4

    eye_w = radius // 5
    eye_h = radius // 6

    # White of eye (sclera)
    sclera_color = (240, 240, 245)

    if expression == "surprised":
        # Wide open eyes (large circles)
        cv2.circle(img, (left_eye_x, eye_y), eye_w + 2, sclera_color, -1, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x, eye_y), eye_w + 2, sclera_color, -1, cv2.LINE_AA)
        cv2.circle(img, (left_eye_x, eye_y), eye_w + 2, color, 2, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x, eye_y), eye_w + 2, color, 2, cv2.LINE_AA)
        # Large pupils
        cv2.circle(img, (left_eye_x, eye_y), eye_w // 2 + 1, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x, eye_y), eye_w // 2 + 1, (30, 30, 30), -1, cv2.LINE_AA)
        # Eye highlights
        cv2.circle(img, (left_eye_x + 2, eye_y - 2), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x + 2, eye_y - 2), 2, (255, 255, 255), -1, cv2.LINE_AA)
    elif expression == "angry":
        # Squinted/narrow eyes
        cv2.ellipse(img, (left_eye_x, eye_y), (eye_w, eye_h // 2), 0, 0, 360, sclera_color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, eye_y), (eye_w, eye_h // 2), 0, 0, 360, sclera_color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (left_eye_x, eye_y), (eye_w, eye_h // 2), 0, 0, 360, color, 2, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, eye_y), (eye_w, eye_h // 2), 0, 0, 360, color, 2, cv2.LINE_AA)
        # Small angry pupils
        cv2.circle(img, (left_eye_x, eye_y), 3, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x, eye_y), 3, (30, 30, 30), -1, cv2.LINE_AA)
    elif expression == "happy":
        # Happy eyes (arched, slightly closed - ^_^ style)
        cv2.ellipse(img, (left_eye_x, eye_y + 2), (eye_w, eye_h), 0, 200, 340, color, 2, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, eye_y + 2), (eye_w, eye_h), 0, 200, 340, color, 2, cv2.LINE_AA)
    else:
        # Neutral eyes (ovals with pupils)
        cv2.ellipse(img, (left_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, sclera_color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, sclera_color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (left_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, color, 2, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, color, 2, cv2.LINE_AA)
        # Pupils
        cv2.circle(img, (left_eye_x, eye_y), eye_w // 3, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x, eye_y), eye_w // 3, (30, 30, 30), -1, cv2.LINE_AA)
        # Highlight
        cv2.circle(img, (left_eye_x + 1, eye_y - 1), 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (right_eye_x + 1, eye_y - 1), 1, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_brows(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
    expression: str,
    color: Tuple[int, int, int],
):
    """Draw eyebrows based on expression. V-shaped for angry, raised for surprised."""
    eye_y = cy - radius // 4
    eye_spacing = radius // 2
    left_eye_x = cx + yaw_offset - eye_spacing + yaw_offset // 4
    right_eye_x = cx + yaw_offset + eye_spacing + yaw_offset // 4

    eye_w = radius // 5
    brow_y = eye_y - radius // 5 - 5

    if expression == "angry":
        # V-shaped angry brows (angled down toward center)
        # Left brow: higher on outside, lower toward center
        cv2.line(img, (left_eye_x - eye_w - 2, brow_y - 3),
                (left_eye_x + eye_w + 2, brow_y + 6), color, 3, cv2.LINE_AA)
        # Right brow: higher on outside, lower toward center
        cv2.line(img, (right_eye_x - eye_w - 2, brow_y + 6),
                (right_eye_x + eye_w + 2, brow_y - 3), color, 3, cv2.LINE_AA)
    elif expression == "surprised":
        # Raised brows (high arcs)
        raised_y = brow_y - 5
        cv2.ellipse(img, (left_eye_x, raised_y), (eye_w + 3, 5), 0, 180, 360, color, 2, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, raised_y), (eye_w + 3, 5), 0, 180, 360, color, 2, cv2.LINE_AA)
    elif expression == "happy":
        # Slightly raised, gentle arcs
        cv2.ellipse(img, (left_eye_x, brow_y - 2), (eye_w + 2, 4), 0, 190, 350, color, 2, cv2.LINE_AA)
        cv2.ellipse(img, (right_eye_x, brow_y - 2), (eye_w + 2, 4), 0, 190, 350, color, 2, cv2.LINE_AA)
    else:
        # Neutral flat brows
        cv2.line(img, (left_eye_x - eye_w, brow_y),
                (left_eye_x + eye_w, brow_y), color, 2, cv2.LINE_AA)
        cv2.line(img, (right_eye_x - eye_w, brow_y),
                (right_eye_x + eye_w, brow_y), color, 2, cv2.LINE_AA)


def _draw_mouth(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
    expression: str,
    color: Tuple[int, int, int],
):
    """Draw mouth based on expression with enhanced detail."""
    mouth_cx = cx + yaw_offset + yaw_offset // 4
    mouth_cy = cy + radius // 3

    if expression == "happy":
        # Big smile with teeth hint
        mouth_w = radius // 3
        mouth_h = radius // 5
        cv2.ellipse(img, (mouth_cx, mouth_cy), (mouth_w, mouth_h), 0, 0, 180, color, 2, cv2.LINE_AA)
        # Teeth area (white arc inside mouth)
        cv2.ellipse(img, (mouth_cx, mouth_cy + 2), (mouth_w - 4, mouth_h - 4), 0, 0, 180,
                   (240, 240, 240), -1, cv2.LINE_AA)
        # Lip line
        cv2.ellipse(img, (mouth_cx, mouth_cy), (mouth_w, mouth_h), 0, 0, 180, color, 2, cv2.LINE_AA)
    elif expression == "surprised":
        # Open O with teeth showing
        mouth_w = radius // 5
        mouth_h = radius // 3
        # Outer mouth
        cv2.ellipse(img, (mouth_cx, mouth_cy + 5), (mouth_w, mouth_h), 0, 0, 360, color, 2, cv2.LINE_AA)
        # Dark interior
        cv2.ellipse(img, (mouth_cx, mouth_cy + 5), (mouth_w - 3, mouth_h - 3), 0, 0, 360,
                   (30, 30, 50), -1, cv2.LINE_AA)
        # Teeth (white rectangles at top of mouth opening)
        teeth_y = mouth_cy + 5 - mouth_h + 5
        teeth_w = mouth_w - 5
        if teeth_w > 2:
            cv2.rectangle(img,
                         (mouth_cx - teeth_w, teeth_y),
                         (mouth_cx + teeth_w, teeth_y + 6),
                         (230, 230, 240), -1, cv2.LINE_AA)
            # Tooth divider lines
            cv2.line(img, (mouth_cx, teeth_y), (mouth_cx, teeth_y + 6), (180, 180, 190), 1, cv2.LINE_AA)
    elif expression == "angry":
        # Frown / grimace
        mouth_w = radius // 3
        mouth_h = radius // 6
        cv2.ellipse(img, (mouth_cx, mouth_cy + mouth_h), (mouth_w, mouth_h), 0, 180, 360, color, 2, cv2.LINE_AA)
        # Teeth gritting
        cv2.line(img, (mouth_cx - mouth_w + 5, mouth_cy + mouth_h),
                (mouth_cx + mouth_w - 5, mouth_cy + mouth_h), (220, 220, 230), 2, cv2.LINE_AA)
    else:
        # Neutral - slight curve
        mouth_w = radius // 4
        cv2.line(img, (mouth_cx - mouth_w, mouth_cy),
                (mouth_cx + mouth_w, mouth_cy), color, 2, cv2.LINE_AA)


def _draw_blush(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    yaw_offset: int,
):
    """Draw blush marks on cheeks (for happy expression)."""
    cheek_y = cy + radius // 8
    cheek_offset = radius // 3

    left_cheek_x = cx + yaw_offset - cheek_offset
    right_cheek_x = cx + yaw_offset + cheek_offset

    blush_color = (160, 160, 255)  # Pinkish in BGR
    blush_w = 10
    blush_h = 6

    # Draw small elliptical blush marks
    cv2.ellipse(img, (left_cheek_x, cheek_y), (blush_w, blush_h), 0, 0, 360, blush_color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (right_cheek_x, cheek_y), (blush_w, blush_h), 0, 0, 360, blush_color, -1, cv2.LINE_AA)


def _draw_direction_arrow(
    img: np.ndarray, cx: int, cy: int, size: int, yaw: str, color: Tuple[int, int, int]
):
    """Draw a direction indicator arrow below the face."""
    arrow_y = cy + size // 2 + 18
    arrow_len = size // 4

    if yaw == "left":
        cv2.arrowedLine(img, (cx + arrow_len, arrow_y), (cx - arrow_len, arrow_y), color, 2, cv2.LINE_AA, tipLength=0.3)
    elif yaw == "right":
        cv2.arrowedLine(img, (cx - arrow_len, arrow_y), (cx + arrow_len, arrow_y), color, 2, cv2.LINE_AA, tipLength=0.3)
    else:
        # Forward - small dot
        cv2.circle(img, (cx, arrow_y), 5, color, -1, cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    cx: int,
    size: int,
    yaw: str,
    expression: str,
    color: Tuple[int, int, int],
):
    """Draw a label below the sprite showing the target pose."""
    label = f"{yaw} + {expression}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = cx - text_w // 2
    text_y = size - 8

    # Shadow
    cv2.putText(img, label, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0, 255), thickness, cv2.LINE_AA)
    # Text
    cv2.putText(img, label, (text_x, text_y), font, font_scale, color + (255,) if len(color) == 3 else color, thickness, cv2.LINE_AA)



def generate_sprite(yaw: str, expression: str, size: int = SPRITE_SIZE) -> np.ndarray:
    """Generate a single sprite as a BGRA numpy array.

    Args:
        yaw: "left", "forward", or "right"
        expression: "neutral", "happy", "surprised", "angry"
        size: Sprite dimensions (square)

    Returns:
        BGRA image with alpha channel (transparent background)
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)

    cx, cy = size // 2, size // 2 - 15
    radius = size // 3

    # Yaw offset for visual rotation
    yaw_offsets = {"left": -radius // 3, "forward": 0, "right": radius // 3}
    yaw_offset = yaw_offsets.get(yaw, 0)

    color = EXPRESSION_COLORS.get(expression, (200, 200, 210))
    bg_inner, bg_outer = BG_GRADIENT.get(expression, ((70, 70, 80), (30, 30, 40)))

    # 1. Draw gradient background circle
    _draw_gradient_circle(img, cx, cy, radius + 18, bg_inner, bg_outer)

    # 2. Inner glow effect
    _draw_inner_glow(img, cx, cy, radius + 18, color, glow_width=10)

    # 3. Draw ears (behind head)
    _draw_ears(img, cx, cy, radius, yaw_offset, yaw)

    # 4. Draw hair (behind face for top portion)
    _draw_hair(img, cx, cy, radius, yaw_offset)

    # 5. Draw head/face shape
    _draw_head_shape(img, cx, cy, radius, yaw_offset)

    # 6. Draw nose
    _draw_nose(img, cx, cy, radius, yaw_offset, color)

    # 7. Draw eyes
    _draw_eyes(img, cx, cy, radius, yaw_offset, expression, color)

    # 8. Draw brows
    _draw_brows(img, cx, cy, radius, yaw_offset, expression, color)

    # 9. Draw mouth
    _draw_mouth(img, cx, cy, radius, yaw_offset, expression, color)

    # 10. Expression-specific extras
    if expression == "happy":
        _draw_blush(img, cx, cy, radius, yaw_offset)

    # 11. Direction arrow
    _draw_direction_arrow(img, cx, cy, radius, yaw, color)

    # 12. Label
    _draw_label(img, cx, size, yaw, expression, color)

    # Ensure all drawn pixels are fully opaque
    drawn_mask = (img[:, :, 0] > 0) | (img[:, :, 1] > 0) | (img[:, :, 2] > 0)
    img[:, :, 3] = np.where(drawn_mask, np.maximum(img[:, :, 3], np.uint8(255)), img[:, :, 3])

    return img


def generate_all_sprites(output_dir: str = SPRITE_DIR, size: int = SPRITE_SIZE) -> list:
    """Generate all 12 placeholder sprites and save as PNGs.

    Returns list of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for yaw in YAW_LABELS:
        for expression in EXPRESSION_LABELS:
            filename = f"{yaw}_{expression}.png"
            filepath = os.path.join(output_dir, filename)
            sprite = generate_sprite(yaw, expression, size)
            cv2.imwrite(filepath, sprite)
            generated.append(filepath)

    return generated


def ensure_sprites_exist(sprite_dir: str = SPRITE_DIR, size: int = SPRITE_SIZE) -> str:
    """Ensure all sprites exist, generating missing ones.

    Returns the sprite directory path.
    """
    os.makedirs(sprite_dir, exist_ok=True)

    for yaw in YAW_LABELS:
        for expression in EXPRESSION_LABELS:
            filename = f"{yaw}_{expression}.png"
            filepath = os.path.join(sprite_dir, filename)
            if not os.path.exists(filepath):
                sprite = generate_sprite(yaw, expression, size)
                cv2.imwrite(filepath, sprite)

    return sprite_dir


def load_sprite(yaw: str, expression: str, sprite_dir: str = SPRITE_DIR) -> np.ndarray:
    """Load a sprite from disk, generating it if missing.

    Returns BGRA image.
    """
    filename = f"{yaw}_{expression}.png"
    filepath = os.path.join(sprite_dir, filename)

    if os.path.exists(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Ensure 4 channels
            if img.shape[2] == 3:
                alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
                img = np.dstack((img, alpha))
            return img

    # Generate if missing or failed to load
    sprite = generate_sprite(yaw, expression)
    os.makedirs(sprite_dir, exist_ok=True)
    cv2.imwrite(filepath, sprite)
    return sprite


if __name__ == "__main__":
    generated = generate_all_sprites()
    print(f"Generated {len(generated)} sprites in {SPRITE_DIR}")
    for path in generated:
        print(f"  {os.path.basename(path)}")
