import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from task3_baseline.data import Task3Dataset, build_samples, load_index
from task3_baseline.model import MultiModalSRSNet
from task3_baseline.utils import compute_srs_metrics, parse_scenarios, parse_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Task 3 baseline")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument(
        "--tasks",
        default="task1,task2",
        help="Comma-separated task list to include (task1, task2)",
    )
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.scenarios = parse_scenarios(args.scenarios)

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
    samples, _, _ = build_samples(
        index_df,
        args.dataset_root,
        args.scenarios,
        dt=checkpoint.get("dt", 0.05),
        video_tolerance=checkpoint.get("video_tolerance", 0.2),
        srs_tolerance=checkpoint.get("srs_tolerance", 0.02),
        e2_columns=e2_columns,
        use_srs_input=use_srs_input,
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

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            e2 = batch["e2"].to(device)
            srs_input = batch["srs_input"].to(device) if use_srs_input else None
            targets = batch["target"].to(device)
            preds = model(video, e2, srs_input=srs_input)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metric_mean = target_mean if target_normalize else None
    metric_std = target_std if target_normalize else None
    metrics = compute_srs_metrics(preds, targets, metric_mean, metric_std)

    print("MAE (I/Q units): {:.4f}".format(metrics["mae"]))
    print("RMSE (I/Q units): {:.4f}".format(metrics["rmse"]))


if __name__ == "__main__":
    main()
