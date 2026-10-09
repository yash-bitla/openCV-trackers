from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from trackcount.boxes import Rect

Frame = npt.NDArray[np.uint8]


class SingleObjectTracker(Protocol):
    """The part of the OpenCV tracker interface that the counter uses."""

    def init(self, frame: Frame, rect: Rect) -> None: ...

    def update(self, frame: Frame) -> tuple[bool, tuple[float, float, float, float]]: ...


TrackerFactory = Callable[[], SingleObjectTracker]

# Name -> (module, factory name). The legacy trackers need opencv-contrib.
_FACTORIES = {
    "csrt": ("", "TrackerCSRT_create"),
    "kcf": ("", "TrackerKCF_create"),
    "mil": ("", "TrackerMIL_create"),
    "boosting": ("legacy", "TrackerBoosting_create"),
    "medianflow": ("legacy", "TrackerMedianFlow_create"),
    "mosse": ("legacy", "TrackerMOSSE_create"),
}

TRACKER_NAMES = sorted(_FACTORIES)


def tracker_factory(name: str) -> TrackerFactory:
    """A function that makes a new OpenCV tracker of the named type."""
    if name not in _FACTORIES:
        raise ValueError(f"unknown tracker '{name}'. Choose one of: {', '.join(TRACKER_NAMES)}")
    module_name, factory_name = _FACTORIES[name]
    module = getattr(cv2, module_name, None) if module_name else cv2
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise RuntimeError(
            f"this OpenCV build has no '{name}' tracker. Install opencv-contrib-python-headless."
        )
    return factory  # type: ignore[no-any-return]
