# Dataset Structure

This folder contains a consolidated dataset for Task 1 and Task 2, with one scenario per experiment and a single CSV index for loading. The dataset for Task 3 uses the same data of Task 1.

## Layout

```
dataset/
  task1/exp1..exp5/
    video/frames.csv
    video/color/
    video/disparity/
    radio/E2.csv
    radio/SRS.json
    annotations/*.csv
  task2/exp6..exp8/
    video/frames.csv
    video/color/
    video/disparity/
    radio/E2.csv
    radio/SRS.json
    annotations/*.csv
  calibration/
    nerian_gnb_1_calib.yaml
  index.csv
```

## Index

`dataset/index.csv` has one row per scenario with dataset-relative paths.

Columns:

- `task`: task1 or task2
- `scenario_id`: exp1..exp8
- `scenario_name`: short label
- `video_frames_csv`: path to frames CSV
- `video_color_dir`: path to color frames directory
- `video_disparity_dir`: path to disparity frames directory
- `radio_e2`: path to E2 CSV
- `radio_srs`: path to SRS JSON
- `annotation`: path to annotation CSV
- `notes`: short scenario description

## Timestamps

- Video timestamps are seconds (float).
- Radio timestamps are seconds (float) and keep absolute time. Radio files are provided as `E2.csv` and `SRS.json`.
- Annotation timestamps are seconds (float).

## Calibration

Stereo calibration is provided in `dataset/calibration/nerian_gnb_1_calib.yaml`
for depth reconstruction from disparity (intrinsics, extrinsics, and Q matrix).

## Task 1

### Annotations

Annotation CSV contains per-frame labels:
`timestamp,state` with `state` in `{no, partial, full}`.
The file name varies per experiment; use `index.csv` to locate it.

### Proposal

Predict the blockage state at `t + dt` using video and radio inputs, with fixed
`dt = 142 ms` (aligned to the video sampling interval).

## Task 2

### Annotations

Annotation CSV contains per-sample translation of Quectel w.r.t. liteon:
`timestamp,x,y,z` (mm), expressed in the liteon frame.
The file name varies per experiment; use `index.csv` to locate it.

### Proposal

Predict the UE position at time `t` using video and radio inputs.

## Task 3

### Proposal

Task 3 uses the same video and radio inputs as Task 1, but with a
different objective: predict the future SRS channel measurement. For each E2
sample at time `t`, predict `srs_ch` at `t + dt`, with fixed `dt = 50 ms`
(aligned to the E2 sampling interval). Models may use past SRS values as
inputs, in addition to video and E2.

## Notes on `frames.csv`

`video/frames.csv` uses paths relative to the `video/` directory:
`color/img_XXXX.png` and `disparity/img_XXXX.png`.