from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from trackcount.trackers import Frame


class FrameSource(Protocol):
    """Anything that can give the frame size and replay frames from a frame number."""

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        ...

    def frames(self, start: int = 0) -> Iterator[tuple[int, Frame]]:
        """Yield (frame number, frame), beginning at frame `start`."""
        ...


class VideoFile:
    """Frames of a video file, resized to a fixed width.

    The detections were made on frames of this width, so the counter must see the same size.
    """

    def __init__(self, path: Path, width: int = 1500) -> None:
        self.path = path
        self.width = width
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"cannot read video: {path}")
        self._size = self._resize(np.asarray(frame, dtype=np.uint8)).shape[1::-1]

    def _resize(self, frame: Frame) -> Frame:
        height, width = frame.shape[:2]
        if width == self.width:
            return frame
        scale = self.width / width
        resized = cv2.resize(
            frame, (self.width, round(height * scale)), interpolation=cv2.INTER_AREA
        )
        return np.asarray(resized, dtype=np.uint8)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (int(self._size[0]), int(self._size[1]))

    def frames(self, start: int = 0) -> Iterator[tuple[int, Frame]]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            # Read and drop frames up to `start`. Seeking by frame number is not exact for
            # every codec, and the frame numbers must match the detection files.
            for _ in range(start):
                if not capture.grab():
                    return
            number = start
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield number, self._resize(np.asarray(frame, dtype=np.uint8))
                number += 1
        finally:
            capture.release()


class FrameList:
    """Frames held in memory. Used by the tests."""

    def __init__(self, frames: list[Frame]) -> None:
        if not frames:
            raise ValueError("no frames")
        self._frames = frames

    @property
    def frame_size(self) -> tuple[int, int]:
        height, width = self._frames[0].shape[:2]
        return (width, height)

    def frames(self, start: int = 0) -> Iterator[tuple[int, Frame]]:
        for number in range(start, len(self._frames)):
            yield number, self._frames[number]
