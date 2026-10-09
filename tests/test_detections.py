from pathlib import Path

import pytest

from trackcount.detections import frame_number, load_detections


def test_frame_number_is_the_last_digits_of_the_name() -> None:
    assert frame_number(Path("IMG_0108_42.txt")) == 42
    assert frame_number(Path("frame7.txt")) == 7
    with pytest.raises(ValueError):
        frame_number(Path("notes.txt"))


def test_load_detections_converts_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "IMG_0108_3.txt").write_text("0 0.75 0.5 0.1 0.2\n0 0.25 0.5 0.1 0.2\n")
    (tmp_path / "IMG_0108_4.txt").write_text("")
    detections = load_detections(tmp_path, (1000, 500))
    # Frame 4 has no boxes, so it is left out. Frame 3 is sorted left to right.
    assert detections == {3: [(200, 200, 300, 300), (700, 200, 800, 300)]}


def test_load_detections_reports_a_bad_line(tmp_path: Path) -> None:
    (tmp_path / "f_1.txt").write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError, match="f_1.txt:1"):
        load_detections(tmp_path, (100, 100))
