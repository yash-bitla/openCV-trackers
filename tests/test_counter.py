import numpy as np
import pytest

from trackcount import Config, FrameList, count_objects, tracker_factory
from trackcount.boxes import Box, Rect, box_to_rect
from trackcount.detections import Detections
from trackcount.trackers import Frame

WIDTH, HEIGHT = 320, 180
SIZE = 40
SPEED = 4  # pixels per frame, toward the left


def object_box(enter_frame: int, y: int, frame: int) -> Box:
    """An object that enters at the right edge on `enter_frame` and moves left."""
    x = WIDTH - SPEED * (frame - enter_frame)
    return (x, y, x + SIZE, y + SIZE)


def visible(box: Box) -> bool:
    """A detector reports an object while at least half of it is inside the frame."""
    inside = min(box[2], WIDTH) - max(box[0], 0)
    return inside >= SIZE / 2


def scene(objects: list[tuple[int, int]], num_frames: int) -> Detections:
    detections: Detections = {}
    for frame in range(num_frames):
        boxes = [object_box(enter, y, frame) for enter, y in objects]
        boxes = sorted((b for b in boxes if visible(b)), key=lambda b: b[0])
        if boxes:
            detections[frame] = boxes
    return detections


# --- The counting logic, with a tracker that never makes a mistake -------------------------


def numbered_frames(num_frames: int) -> FrameList:
    """Blank frames that carry their own frame number, so the fake tracker can read it."""
    frames = []
    for number in range(num_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[0, 0, 0] = number
        frames.append(frame)
    return FrameList(frames)


class PerfectTracker:
    """Follows the true path of the object it was started on."""

    def __init__(self, objects: list[tuple[int, int]]) -> None:
        self.objects = objects
        self.target: tuple[int, int] | None = None

    def init(self, frame: Frame, rect: Rect) -> None:
        number = int(frame[0, 0, 0])
        for enter, y in self.objects:
            if box_to_rect(object_box(enter, y, number)) == rect:
                self.target = (enter, y)
                return
        raise AssertionError("the tracker was started on a box that is not an object")

    def update(self, frame: Frame) -> tuple[bool, tuple[float, float, float, float]]:
        assert self.target is not None
        x, y, w, h = box_to_rect(object_box(*self.target, int(frame[0, 0, 0])))
        return True, (x, y, w, h)


@pytest.mark.parametrize(
    "objects",
    [
        [(0, 20)],
        [(0, 20), (30, 100)],
        [(0, 20), (5, 100), (60, 60), (90, 20)],
        [(0, 20), (0, 110)],  # two objects in the frame at the same time
    ],
)
def test_each_object_is_counted_once(objects: list[tuple[int, int]]) -> None:
    num_frames = 200
    detections = scene(objects, num_frames)
    result = count_objects(
        numbered_frames(num_frames), detections, lambda: PerfectTracker(objects), Config(grow_px=0)
    )
    assert result.count == len(objects)
    assert result.exit_side == "left"
    assert result.unassigned == {}


def test_input_detections_are_not_changed() -> None:
    objects = [(0, 20), (30, 100)]
    detections = scene(objects, 150)
    before = {frame: list(boxes) for frame, boxes in detections.items()}
    count_objects(numbered_frames(150), detections, lambda: PerfectTracker(objects))
    assert detections == before


def test_track_stops_soon_after_the_object_leaves() -> None:
    objects = [(0, 20)]
    config = Config(grow_px=0, frames_after_exit=10)
    result = count_objects(
        numbered_frames(200), scene(objects, 200), lambda: PerfectTracker(objects), config
    )
    track = result.tracks[0]
    assert track.exited
    # The left edge is at x = 320 - 4 * frame, so it is below 0 from frame 81.
    # The check uses the box of the frame before, then the track gets 10 more frames.
    assert max(track.rects) == 81 + 10


def test_output_lines_are_in_frame_order() -> None:
    objects = [(0, 20), (30, 100)]
    result = count_objects(
        numbered_frames(150),
        scene(objects, 150),
        lambda: PerfectTracker(objects),
        Config(grow_px=0),
    )
    lines = result.to_lines()
    frames = [int(line.split()[0]) for line in lines]
    assert frames == sorted(frames)
    assert all(len(line.split()) == 6 for line in lines)
    assert {line.split()[1] for line in lines} == {"1", "2"}


def test_no_detections_gives_zero() -> None:
    result = count_objects(numbered_frames(5), {}, lambda: PerfectTracker([]))
    assert result.count == 0


def test_detections_past_the_end_of_the_video_do_not_loop_forever() -> None:
    objects = [(0, 20)]
    detections = scene(objects, 60)
    detections[500] = [(10, 10, 50, 50)]  # no such frame in a 60-frame video
    result = count_objects(
        numbered_frames(60),
        detections,
        lambda: PerfectTracker(objects),
        Config(grow_px=0),
        exit_side="left",
    )
    assert result.count == 1
    assert 500 in result.unassigned


# --- End to end, with a real OpenCV tracker on a synthetic video --------------------------


def textured_frames(objects: list[tuple[int, int]], num_frames: int) -> FrameList:
    """A gray background with one random-texture patch per object, moving left."""
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 255, size=(SIZE, SIZE, 3), dtype=np.uint8) for _ in objects]
    frames = []
    for number in range(num_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 110, dtype=np.uint8)
        for (enter, y), patch in zip(objects, patches, strict=True):
            x1, y1, x2, y2 = (int(v) for v in object_box(enter, y, number))
            left, right = max(x1, 0), min(x2, WIDTH)
            if right > left:
                frame[y1:y2, left:right] = patch[:, left - x1 : right - x1]
        frames.append(frame)
    return FrameList(frames)


def test_csrt_counts_objects_in_a_synthetic_video() -> None:
    # CSRT only. KCF loses an object that starts half outside the frame, which is how
    # every object enters this scene, so it counts the same object several times.
    objects = [(0, 20), (40, 110), (100, 60)]
    num_frames = 220
    result = count_objects(
        textured_frames(objects, num_frames),
        scene(objects, num_frames),
        tracker_factory("csrt"),
    )
    assert result.count == len(objects)
    assert result.unassigned == {}


def test_box_growth_splits_a_track_that_never_leaves_the_frame() -> None:
    # A known property of the method, kept from the original code. The tracked box grows
    # by `grow_px` each frame. For an object of constant size that is still in the frame
    # when the video ends, the grown box stops matching the detections (IoU falls under
    # 0.3), and the object is counted twice. With no growth it is counted once.
    objects = [(0, 20)]
    frames, detections = numbered_frames(60), scene(objects, 60)
    grown = count_objects(frames, detections, lambda: PerfectTracker(objects), Config(grow_px=1))
    fixed = count_objects(frames, detections, lambda: PerfectTracker(objects), Config(grow_px=0))
    assert grown.count == 2
    assert fixed.count == 1


def test_unknown_tracker_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown tracker"):
        tracker_factory("nope")
