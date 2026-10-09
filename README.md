# OpenCV-Trackers

Tracking code that counts objects in street video from a moving camera. I wrote it for the paper:

> **Object Detection and Counting Challenges in Real Street Monitoring: Case Study of Homeless Encampments**
> Yash Bitla, Abdullah Alfarrarjeh, Seon Ho Kim, Utkarsh Baranwal. IEEE International Conference on Image Processing (ICIP), 2023.
> [Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10222473)

The research was done at the [Integrated Media Systems Center (IMSC)](https://imsc.usc.edu/) at the University of Southern California.

This is research code from 2023, kept as it was. For my current work on tracking, with tests and benchmarks, see [leantrack](https://github.com/yash-bitla/leantrack).

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

## Files

The four scripts are steps of one experiment, from the first test to the batch version.

| File | What it does |
|---|---|
| `tracker.py` | A single-object test. Starts one CSRT tracker from a fixed box at a fixed frame and writes its boxes to `tracker.json`. |
| `homeless_encampment.py` | The first counting version, for one video. Uses a KCF tracker and removes the detections that each track explains. |
| `modified.py` | Adds the camera direction estimate, so a track stops when its object leaves the frame on the correct side. |
| `openCV_tracker.py` | The batch version. Processes every video in a folder with a Boosting tracker and writes one result file for each video. |

## Run it

The scripts need Python with `opencv-contrib-python` (for the legacy trackers), `imutils`, `numpy`, `torch` and `torchvision`.

```bash
python tracker.py --video path/to/video.mp4
```

`openCV_tracker.py` reads videos from `RESULT/`, detection files from `not_noisy/`, and writes to `CVoutput/`. `homeless_encampment.py` and `modified.py` have paths from my own machine in the code, so you must change them before a run.

## Limits

- The repository has no videos, no detection files and no model weights. The dataset of the paper is not included.
- The scripts are not a library. Paths and thresholds are in the code.
- There are no tests. For the evaluation, see the paper.
