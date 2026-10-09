# OpenCV-Trackers

Tracking code that counts objects in street video from a moving camera. I wrote it for the paper:

> **Object Detection and Counting Challenges in Real Street Monitoring: Case Study of Homeless Encampments**
> Yash Bitla, Abdullah Alfarrarjeh, Seon Ho Kim, Utkarsh Baranwal. IEEE International Conference on Image Processing (ICIP), 2023.
> [Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10222473)

The research was done at the [Integrated Media Systems Center (IMSC)](https://imsc.usc.edu/) at the University of Southern California.

The original 2023 scripts are in [`legacy/`](legacy), not changed. The same method is also packaged as `trackcount`, with tests. For my current work on tracking, with benchmarks, see [leantrack](https://github.com/yash-bitla/leantrack).

## The problem

A detector finds tents in each frame of a video that is recorded from a moving vehicle. The same tent appears in many frames, so the number of detections is much larger than the number of tents. To count tents, each one must be followed across frames and counted once.

Most OpenCV tracking examples assume a camera that does not move, and they follow one object that a person selects by hand. Here the camera moves along the street, many objects enter and leave the frame, and the detector output starts each track.

## How it works

1. **Read the detections.** Each frame has a text file of boxes in YOLO format (`class x_center y_center width height`, normalized).
2. **Estimate the camera direction.** Follow one object through consecutive detections and take the sign of its mean horizontal motion. This tells the tracker on which side objects leave the frame.
3. **Start a tracker on the first unassigned detection.** An OpenCV tracker follows that object through the video until its box leaves the frame on the exit side.
4. **Assign detections to the track.** In each frame, a detection that overlaps the tracker box (IoU above a threshold) belongs to this object and is removed from the list.
5. **Repeat** until no detection is left. Each tracker that was started is one object, so the number of trackers is the count.

The output has one line for each tracked box: `frame object_id x y width height`.

## Use it

The method is packaged as `trackcount`, with a command-line tool.

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
trackcount video.mp4 detections/ --out result.txt
```

`detections/` has one text file per frame. The frame number is the last number in the file name (`IMG_0108_42.txt` is frame 42).

```text
objects: 7
exit side: left
```

| Option | Effect |
|---|---|
| `--tracker` | `csrt` (default), `kcf`, `mil`, `boosting`, `medianflow` or `mosse` |
| `--width` | The frame width that the detections were made at. Default 1500. |
| `--exit-side` | `left` or `right`. Skips the direction estimate. |
| `--assign-iou` | Overlap above which a detection joins a track. Default 0.3. |
| `--frames-after-exit` | Frames to keep a track after its box crosses the frame edge. Default 10. |
| `--grow-px` | Pixels to grow the tracked box each frame. Default 1. |

From Python:

```python
from pathlib import Path
from trackcount import VideoFile, count_objects, load_detections, tracker_factory

video = VideoFile(Path("video.mp4"))
detections = load_detections(Path("detections"), video.frame_size)
result = count_objects(video, detections, tracker_factory("csrt"))
print(result.count)
```

## Repository layout

| Path | Content |
|---|---|
| `src/trackcount/counter.py` | The counting loop: start a tracker, follow it, assign detections |
| `src/trackcount/direction.py` | The estimate of the side where objects leave the frame |
| `src/trackcount/detections.py` | Reader for the YOLO text files |
| `src/trackcount/trackers.py` | The OpenCV trackers by name |
| `src/trackcount/video.py` | Frames from a video file, resized to the detection width |
| `src/trackcount/boxes.py` | Box formats and IoU |
| `src/trackcount/cli.py` | The `trackcount` command |
| `tests/` | 26 tests. They need no data: they build a synthetic moving-camera scene. |
| `legacy/` | The four original research scripts from 2023, not changed |

### The original scripts

The scripts in `legacy/` are steps of one experiment, from the first test to the batch version. They have paths from my own machine in the code.

| File | What it does |
|---|---|
| `tracker.py` | A single-object test. Starts one CSRT tracker from a fixed box at a fixed frame. |
| `homeless_encampment.py` | The first counting version, for one video, with a KCF tracker. |
| `modified.py` | Adds the camera direction estimate. |
| `openCV_tracker.py` | The batch version, with a Boosting tracker. The package follows this one. |

### What the package does differently

`trackcount` is a re-implementation of the method in `legacy/openCV_tracker.py`. I did not run it on the data of the paper, so I cannot say that it gives the same numbers.

- It reads the video from the start frame of each track. The script reads from frame 0 each time.
- The direction estimate uses the longest chain of linked detections when no object lasts 10 frames. The script has no such fallback.
- The count is the number of tracks. The script prints that number minus 1.
- The default tracker is CSRT. The batch script uses Boosting, which is available as `--tracker boosting`.
- It has no PyTorch dependency. IoU is computed directly.

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check . && mypy
```

The tests cover the counting loop with a tracker that makes no errors, and the full pipeline with a real CSRT tracker on a synthetic video file.

## Known behavior

- **Box growth can split a track.** The tracked box grows by `--grow-px` each frame, because objects get larger as the camera comes nearer. For an object of constant size that is still in the frame when the video ends, the grown box stops matching the detections, and the object is counted two times. A test documents this. Use `--grow-px 0` for objects that do not change size.
- **KCF loses objects that start at the frame edge.** In the synthetic scene, it counts one object several times. CSRT does not.

## Limits

- The repository has no videos, no detection files and no model weights. The dataset of the paper is not included.
- The package is tested on synthetic scenes only. For the evaluation on real street video, see the paper.
- It processes one video at a time, and each track reads the video again from its start frame. A long video with many objects is slow.
