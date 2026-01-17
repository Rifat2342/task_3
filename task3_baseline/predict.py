import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from task3_baseline.data import Task3Dataset, build_samples, load_index
from task3_baseline.model import MultiModalSRSNet
from task3_baseline.utils import denormalize_values, parse_scenarios, parse_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Task 3 SRS channels")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument(
        "--tasks",
        default="task1,task2",
        help="Comma-separated task list to include (task1, task2)",
    )
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.scenarios = parse_scenarios(args.scenarios)

    output_path = Path(args.output)
    if output_path.suffix != ".npz":
        raise ValueError("Output file must have .npz extension for Task 3 predictions.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    e2_columns = checkpoint["e2_columns"]
    e2_mean = checkpoint["e2_mean"]
    e2_std = checkpoint["e2_std"]
    srs_input_mean = checkpoint.get("srs_input_mean")
    srs_input_std = checkpoint.get("srs_input_std")
    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")
    target_normalize = checkpoint.get("target_normalize", True)
    srs_input_normalize = checkpoint.get("srs_input_normalize", True)
    use_srs_input = checkpoint.get("use_srs_input", False)
    video_mode = checkpoint.get("video_mode", "rgbd")
    image_size = checkpoint.get("image_size", 128)
    backbone = checkpoint.get(
        "backbone", checkpoint.get("config", {}).get("backbone", "simple")
    )
    srs_dim = checkpoint.get("srs_dim")

    index_csv = args.index_csv
    if index_csv is None:
        index_csv = Path(args.dataset_root) / "index.csv"

    index_df = load_index(index_csv, tasks=parse_tasks(args.tasks))
    dt = checkpoint.get("dt", 0.05)
    samples, _, _ = build_samples(
        index_df,
        args.dataset_root,
        args.scenarios,
        dt=dt,
        video_tolerance=checkpoint.get("video_tolerance", 0.2),
        srs_tolerance=checkpoint.get("srs_tolerance", 0.02),
        e2_columns=e2_columns,
        use_srs_input=use_srs_input,
        require_target=False,
        srs_dim=srs_dim,
    )

    dataset = Task3Dataset(
        samples,
        image_size=image_size,
        video_mode=video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
        srs_input_mean=srs_input_mean,
        srs_input_std=srs_input_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=target_normalize,
        normalize_srs_input=srs_input_normalize,
        use_srs_input=use_srs_input,
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiModalSRSNet(
        e2_dim=len(e2_columns),
        srs_dim=srs_dim,
        video_mode=video_mode,
        use_srs_input=use_srs_input,
        backbone=backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    preds_all = []
    timestamps = []
    target_timestamps = []
    scenario_ids = []

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            e2 = batch["e2"].to(device)
            srs_input = batch["srs_input"].to(device) if use_srs_input else None
            preds = model(video, e2, srs_input=srs_input).cpu()

            if target_normalize:
                mean = torch.tensor(target_mean, dtype=preds.dtype)
                std = torch.tensor(target_std, dtype=preds.dtype)
                preds = denormalize_values(preds, mean, std)

            preds_all.append(preds.numpy())
            batch_times = [float(ts) for ts in batch["timestamp"]]
            timestamps.extend(batch_times)
            target_timestamps.extend([ts + dt for ts in batch_times])
            scenario_ids.extend(batch["scenario_id"])

    preds_all = np.concatenate(preds_all, axis=0)
    np.savez(
        output_path,
        preds=preds_all,
        timestamps=np.asarray(timestamps, dtype=np.float32),
        target_timestamps=np.asarray(target_timestamps, dtype=np.float32),
        scenario_ids=np.asarray(scenario_ids, dtype=object),
        srs_dim=np.asarray([srs_dim], dtype=np.int32),
        dt=np.asarray([dt], dtype=np.float32),
    )
    print("Wrote {} predictions to {}".format(len(preds_all), output_path))


if __name__ == "__main__":
    main()
