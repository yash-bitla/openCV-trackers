import pytest

from trackcount.detections import Detections
from trackcount.direction import estimate_exit_side


def moving_object(start_x: int, step: int, frames: int) -> Detections:
    return {f: [(start_x + step * f, 50, start_x + step * f + 40, 90)] for f in range(frames)}


def test_object_that_moves_left_exits_left() -> None:
    assert estimate_exit_side(moving_object(start_x=300, step=-5, frames=15)) == "left"


def test_object_that_moves_right_exits_right() -> None:
    assert estimate_exit_side(moving_object(start_x=10, step=5, frames=15)) == "right"


def test_short_chain_is_used_when_no_long_one_exists() -> None:
    assert estimate_exit_side(moving_object(start_x=300, step=-5, frames=3)) == "left"


def test_other_objects_in_the_frame_do_not_break_the_chain() -> None:
    detections = moving_object(start_x=300, step=-5, frames=15)
    for frame in detections:
        detections[frame].append((600, 200, 640, 240))  # a second object far away
    assert estimate_exit_side(detections) == "left"


def test_no_consecutive_frames_is_an_error() -> None:
    with pytest.raises(ValueError):
        estimate_exit_side({0: [(0, 0, 10, 10)], 5: [(100, 100, 110, 110)]})
