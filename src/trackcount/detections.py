import re
from pathlib import Path

from trackcount.boxes import Box, yolo_to_box

# frame number -> boxes in that frame, sorted left to right
Detections = dict[int, list[Box]]

_FRAME_NUMBER = re.compile(r"(\d+)$")


def frame_number(path: Path) -> int:
    """The frame number is the last run of digits in the file name: `IMG_0108_42.txt` -> 42."""
    match = _FRAME_NUMBER.search(path.stem)
    if match is None:
        raise ValueError(f"no frame number in file name: {path.name}")
    return int(match.group(1))


def load_detections(directory: Path, frame_size: tuple[int, int]) -> Detections:
    """Read one YOLO text file per frame: `class x_center y_center width height`, normalized.

    Frames with no boxes are left out.
    """
    detections: Detections = {}
    for path in sorted(directory.glob("*.txt")):
        boxes: list[Box] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"{path.name}:{line_number}: expected 5 values, got {len(parts)}")
            cx, cy, w, h = (float(v) for v in parts[1:5])
            boxes.append(yolo_to_box((cx, cy, w, h), frame_size))
        if boxes:
            detections[frame_number(path)] = sorted(boxes, key=lambda box: box[0])
    return detections
