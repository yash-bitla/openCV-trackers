import copy
import logging
from dataclasses import dataclass, field

from trackcount.boxes import Rect, box_to_rect, iou, rect_to_box
from trackcount.detections import Detections
from trackcount.direction import ExitSide, estimate_exit_side
from trackcount.trackers import TrackerFactory
from trackcount.video import FrameSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    # A detection joins a track when it overlaps the tracker box by more than this.
    assign_iou: float = 0.3
    # After a track has left the frame, any overlap is enough.
    assign_iou_after_exit: float = 0.0
    # Keep a track alive for this many frames after its box crosses the frame edge.
    frames_after_exit: int = 10
    # Grow the tracked box by this many pixels per frame. Objects get larger as the camera
    # comes closer, and OpenCV trackers keep the box size they started with.
    grow_px: int = 1
    # Direction estimate: links needed, and the overlap that links two detections.
    direction_min_links: int = 10
    direction_link_iou: float = 0.5


@dataclass
class Track:
    object_id: int
    # frame number -> tracked box as (x, y, w, h)
    rects: dict[int, Rect] = field(default_factory=dict)
    exited: bool = False


@dataclass
class Result:
    tracks: list[Track]
    exit_side: ExitSide
    # Detections that no track explained. Not empty only when the loop could not go on.
    unassigned: Detections

    @property
    def count(self) -> int:
        return len(self.tracks)

    def to_lines(self) -> list[str]:
        """One line per tracked box, in frame order: `frame object_id x y w h`."""
        rows = [
            (frame, track.object_id, rect)
            for track in self.tracks
            for frame, rect in track.rects.items()
        ]
        rows.sort(key=lambda row: (row[0], row[1]))
        return [
            f"{frame} {object_id} {' '.join(map(str, rect))}" for frame, object_id, rect in rows
        ]


def _follow(
    source: FrameSource,
    start_frame: int,
    start_rect: Rect,
    object_id: int,
    exit_side: ExitSide,
    make_tracker: TrackerFactory,
    config: Config,
) -> Track:
    """Start a tracker on one detection and follow the object until it leaves the frame."""
    frame_width = source.frame_size[0]
    track = Track(object_id=object_id)
    tracker = make_tracker()
    rect = start_rect
    countdown = config.frames_after_exit
    updates = 0

    for number, frame in source.frames(start_frame):
        if number == start_frame:
            tracker.init(frame, rect)
            track.rects[number] = rect
            continue

        x, _, w, _ = rect
        outside = x < 0 if exit_side == "left" else x + w > frame_width
        if outside:
            track.exited = True
        if track.exited:
            if countdown == 0:
                break
            countdown -= 1

        ok, raw = tracker.update(frame)
        updates += 1
        if not ok:
            continue
        nx, ny, nw, nh = (int(v) for v in raw)
        if updates > 1 and config.grow_px:
            nw, nh = rect[2] + config.grow_px, rect[3] + config.grow_px
        rect = (nx, ny, nw, nh)
        track.rects[number] = rect

    return track


def _assign(detections: Detections, track: Track, config: Config) -> int:
    """Remove the detections that this track explains. Returns how many were removed."""
    threshold = config.assign_iou_after_exit if track.exited else config.assign_iou
    removed = 0
    for frame, rect in track.rects.items():
        if frame not in detections:
            continue
        tracked = rect_to_box(rect)
        kept = [box for box in detections[frame] if iou(box, tracked) <= threshold]
        removed += len(detections[frame]) - len(kept)
        if kept:
            detections[frame] = kept
        else:
            del detections[frame]
    return removed


def count_objects(
    source: FrameSource,
    detections: Detections,
    make_tracker: TrackerFactory,
    config: Config | None = None,
    exit_side: ExitSide | None = None,
) -> Result:
    """Count the distinct objects behind per-frame detections from a moving camera.

    Repeat until no detection is left: start a tracker on the earliest unassigned
    detection, follow it until it leaves the frame, and remove every detection that
    overlaps the tracked box. Each tracker is one object.
    """
    config = config or Config()
    remaining = copy.deepcopy(detections)
    if not remaining:
        return Result(tracks=[], exit_side=exit_side or "left", unassigned={})
    if exit_side is None:
        exit_side = estimate_exit_side(
            remaining, config.direction_min_links, config.direction_link_iou
        )

    tracks: list[Track] = []
    while remaining:
        start_frame = min(remaining)
        seed = remaining[start_frame][0]
        track = _follow(
            source, start_frame, box_to_rect(seed), len(tracks) + 1, exit_side, make_tracker, config
        )
        if _assign(remaining, track, config) == 0:
            # The seed detection always overlaps its own track, so this means the frame
            # was not in the video. Stop instead of looping forever.
            log.warning("frame %d has detections but is not in the video; stopping", start_frame)
            break
        tracks.append(track)
        log.info("object %d: frames %d to %d", track.object_id, start_frame, max(track.rects))

    return Result(tracks=tracks, exit_side=exit_side, unassigned=remaining)
