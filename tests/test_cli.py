from pathlib import Path

import cv2
import pytest

from test_counter import HEIGHT, WIDTH, scene, textured_frames
from trackcount.cli import main

OBJECTS = [(0, 20), (40, 110), (100, 60)]
NUM_FRAMES = 220


def write_inputs(root: Path) -> tuple[Path, Path]:
    """A real video file and one YOLO text file per frame, like the inputs of the paper."""
    video = root / "street.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"MJPG"), 30, (WIDTH, HEIGHT))
    for _, frame in textured_frames(OBJECTS, NUM_FRAMES).frames():
        writer.write(frame)
    writer.release()

    folder = root / "detections"
    folder.mkdir()
    for number, boxes in scene(OBJECTS, NUM_FRAMES).items():
        lines = []
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / WIDTH, (y1 + y2) / 2 / HEIGHT
            w, h = (x2 - x1) / WIDTH, (y2 - y1) / HEIGHT
            lines.append(f"0 {cx} {cy} {w} {h}")
        (folder / f"street_{number}.txt").write_text("\n".join(lines) + "\n")
    return video, folder


def test_cli_counts_objects_and_writes_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video, folder = write_inputs(tmp_path)
    out = tmp_path / "results" / "street.txt"

    code = main([str(video), str(folder), "--width", str(WIDTH), "--out", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "objects: 3" in printed
    assert "exit side: left" in printed
    lines = out.read_text().splitlines()
    assert {line.split()[1] for line in lines} == {"1", "2", "3"}
    assert all(len(line.split()) == 6 for line in lines)


def test_cli_reports_a_missing_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "detections").mkdir()
    code = main([str(tmp_path / "nope.mp4"), str(tmp_path / "detections")])
    assert code == 2
    assert "no such video" in capsys.readouterr().err
