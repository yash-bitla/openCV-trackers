import pytest

from trackcount.boxes import box_to_rect, iou, rect_to_box, yolo_to_box


def test_yolo_to_box() -> None:
    # Center (0.5, 0.5), size (0.2, 0.4) on a 1000 x 500 frame:
    # x: 500 - 100 = 400 to 500 + 100 = 600.  y: 250 - 100 = 150 to 250 + 100 = 350.
    assert yolo_to_box((0.5, 0.5, 0.2, 0.4), (1000, 500)) == (400, 150, 600, 350)


def test_rect_and_box_are_inverse() -> None:
    assert box_to_rect((400, 150, 600, 350)) == (400, 150, 200, 200)
    assert rect_to_box((400, 150, 200, 200)) == (400, 150, 600, 350)


def test_iou_hand_computed() -> None:
    # Two 10 x 10 boxes shifted by 5 in x: intersection 5 * 10 = 50, union 100 + 100 - 50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_edge_cases() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0  # touching edges do not overlap
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
