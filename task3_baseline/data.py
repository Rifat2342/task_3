import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from task3_baseline.utils import parse_scenarios, parse_tasks


def load_index(index_csv, tasks=None):
    index_df = pd.read_csv(index_csv)
    if tasks:
        tasks = set(parse_tasks(tasks))
        index_df = index_df[index_df["task"].isin(tasks)]
    return index_df.reset_index(drop=True)


def _resample_mode():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def _load_color(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), _resample_mode())
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_disparity(path, image_size):
    img = Image.open(path).convert("L")
    img = img.resize((image_size, image_size), _resample_mode())
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _flatten_srs(srs_ch):
    arr = np.asarray(srs_ch, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("srs_ch must have shape (N, 2)")
    return arr.reshape(-1)


def _load_srs(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        raise ValueError("SRS JSON must be a list of entries.")

    times = np.array([entry["timestamp"] for entry in entries], dtype=np.float32)
    order = np.argsort(times)
    times = times[order]

    values = []
    expected_dim = None
    for idx in order:
        entry = entries[int(idx)]
        flattened = _flatten_srs(entry["srs_ch"])
        if expected_dim is None:
            expected_dim = flattened.shape[0]
        elif flattened.shape[0] != expected_dim:
            raise ValueError("Inconsistent SRS dimensions in {}".format(path))
        values.append(flattened)

    return times, np.stack(values, axis=0)


def _nearest_index(times, query_time, tolerance):
    idx = int(np.searchsorted(times, query_time))
    candidates = []
    if idx < len(times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best_idx = min(candidates, key=lambda i: abs(times[i] - query_time))
    if abs(times[best_idx] - query_time) <= tolerance:
        return best_idx
    return None


def _validate_e2_columns(columns, reference_columns):
    missing = [col for col in reference_columns if col not in columns]
    if missing:
        raise ValueError(
            "E2 feature columns mismatch. Missing columns: {}".format(", ".join(missing)
            )
        )


def build_samples(
    index_df,
    dataset_root,
    scenarios,
    dt=0.05,
    video_tolerance=0.2,
    srs_tolerance=0.02,
    e2_columns=None,
    use_srs_input=False,
    require_target=True,
    srs_dim=None,
):
    dataset_root = Path(dataset_root)
    scenarios = set(parse_scenarios(scenarios))
    if scenarios:
        scenario_df = index_df[index_df["scenario_id"].isin(scenarios)]
    else:
        scenario_df = index_df

    if scenario_df.empty:
        raise ValueError("No scenarios matched the provided list.")

    samples = []
    reference_e2_columns = list(e2_columns) if e2_columns else None
    inferred_srs_dim = srs_dim

    for row in scenario_df.itertuples(index=False):
        frames_path = dataset_root / row.video_frames_csv
        video_root = frames_path.parent
        frames_df = pd.read_csv(frames_path)
        frames_df = frames_df.sort_values("timestamp")
        frame_times = frames_df["timestamp"].to_numpy(dtype=np.float32)
        color_paths = frames_df["color"].tolist()
        disp_paths = frames_df["disparity"].tolist()

        e2_path = dataset_root / row.radio_e2
        e2_df = pd.read_csv(e2_path)
        e2_df["timestamp"] = e2_df["timestamp"].astype(float)
        e2_df = e2_df.sort_values("timestamp")
        numeric_cols = e2_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "timestamp"]

        if reference_e2_columns is None:
            reference_e2_columns = numeric_cols
        else:
            _validate_e2_columns(numeric_cols, reference_e2_columns)

        e2_times = e2_df["timestamp"].to_numpy(dtype=np.float32)
        e2_values = e2_df[reference_e2_columns].to_numpy(dtype=np.float32)

        srs_times = None
        srs_values = None
        if require_target or use_srs_input:
            srs_path = dataset_root / row.radio_srs
            srs_times, srs_values = _load_srs(srs_path)
            if inferred_srs_dim is None:
                inferred_srs_dim = srs_values.shape[1]
            elif srs_values.shape[1] != inferred_srs_dim:
                raise ValueError("SRS dimension mismatch across scenarios.")
        elif inferred_srs_dim is None:
            raise ValueError(
                "srs_dim must be provided when building samples without targets."
            )

        for idx, time_t in enumerate(e2_times):
            frame_idx = _nearest_index(frame_times, time_t, video_tolerance)
            if frame_idx is None:
                continue
            if require_target:
                target_time = float(time_t + dt)
                target_idx = _nearest_index(srs_times, target_time, srs_tolerance)
                if target_idx is None:
                    continue
                target = srs_values[target_idx]
            else:
                target = np.zeros(inferred_srs_dim, dtype=np.float32)

            srs_input = None
            if use_srs_input:
                input_idx = _nearest_index(srs_times, time_t, srs_tolerance)
                if input_idx is None:
                    if require_target:
                        continue
                    srs_input = np.zeros(inferred_srs_dim, dtype=np.float32)
                else:
                    srs_input = srs_values[input_idx]

            samples.append(
                {
                    "color_path": str(video_root / color_paths[frame_idx]),
                    "disparity_path": str(video_root / disp_paths[frame_idx]),
                    "e2": e2_values[idx],
                    "srs_input": srs_input,
                    "target": target,
                    "timestamp": float(time_t),
                    "scenario_id": row.scenario_id,
                }
            )

    return samples, reference_e2_columns, inferred_srs_dim


def compute_e2_stats(samples):
    features = np.stack([sample["e2"] for sample in samples], axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_srs_stats(samples, key):
    values = np.stack([sample[key] for sample in samples], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class Task3Dataset(Dataset):
    def __init__(
        self,
        samples,
        image_size=128,
        video_mode="rgbd",
        e2_mean=None,
        e2_std=None,
        srs_input_mean=None,
        srs_input_std=None,
        target_mean=None,
        target_std=None,
        normalize_target=True,
        normalize_srs_input=True,
        use_srs_input=False,
    ):
        self.samples = samples
        self.image_size = image_size
        self.video_mode = video_mode
        self.use_video = video_mode != "none"
        self.use_color = video_mode in ("rgb", "rgbd")
        self.use_disparity = video_mode in ("disparity", "rgbd")
        self.e2_mean = e2_mean
        self.e2_std = e2_std
        self.srs_input_mean = srs_input_mean
        self.srs_input_std = srs_input_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.normalize_target = normalize_target
        self.normalize_srs_input = normalize_srs_input
        self.use_srs_input = use_srs_input

        if video_mode == "rgbd":
            self.video_channels = 4
        elif video_mode == "rgb":
            self.video_channels = 3
        elif video_mode == "disparity":
            self.video_channels = 1
        else:
            self.video_channels = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        e2 = sample["e2"].astype(np.float32)
        if self.e2_mean is not None and self.e2_std is not None:
            e2 = (e2 - self.e2_mean) / self.e2_std
        e2_tensor = torch.from_numpy(e2)

        target = sample["target"].astype(np.float32)
        if self.normalize_target and self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std
        target_tensor = torch.from_numpy(target)

        if self.use_srs_input:
            srs_input = sample["srs_input"].astype(np.float32)
            if (
                self.normalize_srs_input
                and self.srs_input_mean is not None
                and self.srs_input_std is not None
            ):
                srs_input = (srs_input - self.srs_input_mean) / self.srs_input_std
            srs_input_tensor = torch.from_numpy(srs_input)
        else:
            srs_input_tensor = torch.zeros(1, dtype=torch.float32)

        if self.use_video:
            channels = []
            if self.use_color:
                channels.append(_load_color(sample["color_path"], self.image_size))
            if self.use_disparity:
                channels.append(
                    _load_disparity(sample["disparity_path"], self.image_size)
                )
            if channels:
                video = torch.cat(channels, dim=0)
            else:
                video = torch.zeros(
                    (self.video_channels, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
        else:
            video = torch.zeros(
                (self.video_channels, self.image_size, self.image_size),
                dtype=torch.float32,
            )

        return {
            "video": video,
            "e2": e2_tensor,
            "srs_input": srs_input_tensor,
            "target": target_tensor,
            "timestamp": sample["timestamp"],
            "scenario_id": sample["scenario_id"],
        }
