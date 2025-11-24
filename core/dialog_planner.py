from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from core.helpers import DialogSession


class DialogPlanner(ABC):
    """Abstract interface for planner models consumed by the MCTS loop."""

    @abstractmethod
    def get_valid_moves(self, state: DialogSession) -> np.ndarray:
        """Return a binary mask indicating which dialog acts are valid in the current state."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: DialogSession) -> Tuple[np.ndarray, float]:
        """Return (policy, value) predictions for the provided dialog state."""
        raise NotImplementedError


__all__ = ["DialogPlanner"]

