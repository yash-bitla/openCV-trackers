"""Box formats and overlap.

Two formats are used:
- `Box`: corners in pixels, (x1, y1, x2, y2). Detections and overlaps use this.
- `Rect`: top-left corner and size in pixels, (x, y, w, h). OpenCV trackers use this.
"""

Box = tuple[float, float, float, float]
Rect = tuple[int, int, int, int]


def yolo_to_box(yolo: tuple[float, float, float, float], frame_size: tuple[int, int]) -> Box:
    """Convert a normalized YOLO box (x_center, y_center, width, height) to pixel corners."""
    width, height = frame_size
    cx, cy, w, h = yolo
    return (
        round((cx - w / 2) * width),
        round((cy - h / 2) * height),
        round((cx + w / 2) * width),
        round((cy + h / 2) * height),
    )


def box_to_rect(box: Box) -> Rect:
    x1, y1, x2, y2 = box
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def rect_to_box(rect: Rect) -> Box:
    x, y, w, h = rect
    return (x, y, x + w, y + h)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes. 0 when they do not overlap."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
