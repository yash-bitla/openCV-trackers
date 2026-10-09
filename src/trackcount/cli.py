import argparse
import logging
import sys
from pathlib import Path

from trackcount.counter import Config, count_objects
from trackcount.detections import load_detections
from trackcount.trackers import TRACKER_NAMES, tracker_factory
from trackcount.video import VideoFile


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trackcount",
        description="Count distinct objects in a moving-camera video, from per-frame detections.",
    )
    p.add_argument("video", type=Path, help="video file")
    p.add_argument("detections", type=Path, help="folder with one YOLO text file per frame")
    p.add_argument("--out", type=Path, help="write one line per tracked box: frame id x y w h")
    p.add_argument("--tracker", choices=TRACKER_NAMES, default="csrt")
    p.add_argument(
        "--width", type=int, default=1500, help="frame width the detections were made at"
    )
    p.add_argument("--exit-side", choices=["left", "right"], help="skip the direction estimate")
    p.add_argument("--assign-iou", type=float, default=Config.assign_iou)
    p.add_argument("--frames-after-exit", type=int, default=Config.frames_after_exit)
    p.add_argument("--grow-px", type=int, default=Config.grow_px)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    if not args.video.is_file():
        print(f"error: no such video: {args.video}", file=sys.stderr)
        return 2
    if not args.detections.is_dir():
        print(f"error: no such folder: {args.detections}", file=sys.stderr)
        return 2

    source = VideoFile(args.video, width=args.width)
    detections = load_detections(args.detections, source.frame_size)
    config = Config(
        assign_iou=args.assign_iou,
        frames_after_exit=args.frames_after_exit,
        grow_px=args.grow_px,
    )
    result = count_objects(
        source, detections, tracker_factory(args.tracker), config, exit_side=args.exit_side
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(result.to_lines()) + "\n")
    print(f"objects: {result.count}")
    print(f"exit side: {result.exit_side}")
    if result.unassigned:
        left = sum(len(boxes) for boxes in result.unassigned.values())
        print(f"warning: {left} detections were not assigned to any object", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
