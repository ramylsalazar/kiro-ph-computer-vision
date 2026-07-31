"""High score persistence - save/load per difficulty to JSON.

Handles missing or invalid files without crashing.
Only writes when a high score changes.
"""

import json
import os
from typing import Dict

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscores.json")


def load_high_scores(filepath: str = DEFAULT_FILE) -> Dict[str, int]:
    """Load high scores from JSON file.

    Returns a dict mapping difficulty name to high score.
    Handles missing or corrupt files gracefully.
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return {}

    # Validate structure
    if not isinstance(data, dict):
        return {}

    result = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            result[key] = int(value)

    return result


def save_high_scores(scores: Dict[str, int], filepath: str = DEFAULT_FILE) -> bool:
    """Save high scores to JSON file.

    Returns True if write succeeded, False otherwise.
    """
    # Validate before writing
    validated = {}
    for key, value in scores.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            validated[key] = int(value)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2)
        return True
    except OSError:
        return False


class HighScoreManager:
    """Manages high scores per difficulty. Only writes on change."""

    def __init__(self, filepath: str = DEFAULT_FILE):
        self._filepath = filepath
        self._scores = load_high_scores(filepath)
        self._dirty = False

    def get(self, difficulty_name: str) -> int:
        """Get high score for a difficulty. Returns 0 if not set."""
        return self._scores.get(difficulty_name, 0)

    def update(self, difficulty_name: str, score: int) -> bool:
        """Update high score if new score is higher.

        Returns True if high score was beaten.
        Only marks for save if the score actually changed.
        """
        current = self._scores.get(difficulty_name, 0)
        if score > current:
            self._scores[difficulty_name] = score
            self._dirty = True
            self.save()  # Write immediately on change
            return True
        return False

    def save(self) -> bool:
        """Save to file if dirty. Returns True if saved successfully."""
        if not self._dirty:
            return True
        success = save_high_scores(self._scores, self._filepath)
        if success:
            self._dirty = False
        return success

    def reload(self):
        """Reload scores from file."""
        self._scores = load_high_scores(self._filepath)
        self._dirty = False

    @property
    def all_scores(self) -> Dict[str, int]:
        """Return copy of all scores."""
        return dict(self._scores)
