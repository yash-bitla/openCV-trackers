from typing import Literal

from trackcount.boxes import Box, iou
from trackcount.detections import Detections

# The side of the frame where objects leave. A camera that moves right makes objects exit left.
ExitSide = Literal["left", "right"]


def _follow(detections: Detections, frame: int, box: Box, link_iou: float) -> list[Box]:
    """Follow one object through consecutive frames, by overlap with its previous box."""
    chain = [box]
    while True:
        frame += 1
        candidates = [b for b in detections.get(frame, []) if iou(chain[-1], b) > link_iou]
        if not candidates:
            return chain
        chain.append(max(candidates, key=lambda b: iou(chain[-1], b)))


def estimate_exit_side(
    detections: Detections, min_links: int = 10, link_iou: float = 0.5
) -> ExitSide:
    """Find where objects leave the frame, from the motion of one object's detections.

    The first object that can be followed for `min_links` consecutive frames decides.
    If no object lasts that long, the longest chain decides.
    """
    best: list[Box] = []
    for frame in sorted(detections):
        for box in detections[frame]:
            chain = _follow(detections, frame, box, link_iou)
            if len(chain) > len(best):
                best = chain
            if len(best) > min_links:
                break
        if len(best) > min_links:
            break
    if len(best) < 2:
        raise ValueError(
            "cannot estimate the direction: no object appears in two consecutive frames"
        )

    steps = len(best) - 1
    left_edge = (best[-1][0] - best[0][0]) / steps
    right_edge = (best[-1][2] - best[0][2]) / steps
    # The edge that moves more is the more reliable one when the object is partly outside.
    rate = left_edge if abs(left_edge) > abs(right_edge) else right_edge
    return "left" if rate < 0 else "right"
