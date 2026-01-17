import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task3_baseline.data import (
    Task3Dataset,
    build_samples,
    compute_e2_stats,
    compute_srs_stats,
    load_index,
)
from task3_baseline.model import MultiModalSRSNet
from task3_baseline.utils import compute_srs_metrics, parse_scenarios, parse_tasks, set_seed


def _resolve_scenarios(args, index_df):
    train_scenarios = parse_scenarios(args.train_scenarios)
    val_scenarios = parse_scenarios(args.val_scenarios)

    if args.preset == "loso":
        holdout = args.holdout_scenario
        if holdout is None:
            if len(val_scenarios) == 1:
                holdout = val_scenarios[0]
            else:
                raise ValueError(
                    "LOSO preset requires --holdout-scenario or a single --val-scenarios entry."
                )

        all_scenarios = index_df["scenario_id"].tolist()
        if holdout not in all_scenarios:
            raise ValueError(
                "Holdout scenario {} not found in dataset index.".format(holdout)
            )

        train_scenarios = [sid for sid in all_scenarios if sid != holdout]
        val_scenarios = [holdout]

    return train_scenarios, val_scenarios


def _build_loaders(args, index_df, train_scenarios, val_scenarios):
    train_samples, e2_columns, srs_dim = build_samples(
        index_df,
        args.dataset_root,
        train_scenarios,
        dt=args.dt,
        video_tolerance=args.video_tolerance,
        srs_tolerance=args.srs_tolerance,
        use_srs_input=args.use_srs_input,
    )

    val_samples, _, _ = build_samples(
        index_df,
        args.dataset_root,
        val_scenarios,
        dt=args.dt,
        video_tolerance=args.video_tolerance,
        srs_tolerance=args.srs_tolerance,
        e2_columns=e2_columns,
        use_srs_input=args.use_srs_input,
    )

    e2_mean, e2_std = compute_e2_stats(train_samples)
    target_mean, target_std = compute_srs_stats(train_samples, "target")

    if args.use_srs_input:
        srs_input_mean, srs_input_std = compute_srs_stats(train_samples, "srs_input")
    else:
        srs_input_mean, srs_input_std = None, None

    train_dataset = Task3Dataset(
        train_samples,
        image_size=args.image_size,
        video_mode=args.video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
        srs_input_mean=srs_input_mean,
        srs_input_std=srs_input_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=args.target_normalize,
        normalize_srs_input=args.srs_input_normalize,
        use_srs_input=args.use_srs_input,
    )

    val_dataset = Task3Dataset(
        val_samples,
        image_size=args.image_size,
        video_mode=args.video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
        srs_input_mean=srs_input_mean,
        srs_input_std=srs_input_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=args.target_normalize,
        normalize_srs_input=args.srs_input_normalize,
        use_srs_input=args.use_srs_input,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        e2_columns,
        srs_dim,
        e2_mean,
        e2_std,
        srs_input_mean,
        srs_input_std,
        target_mean,
        target_std,
    )


def _run_epoch(
    model,
    loader,
    criterion,
    target_mean,
    target_std,
    optimizer=None,
    device="cpu",
    normalize_target=True,
    use_srs_input=False,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, leave=False):
        video = batch["video"].to(device)
        e2 = batch["e2"].to(device)
        targets = batch["target"].to(device)
        srs_input = batch["srs_input"].to(device) if use_srs_input else None

        if is_train:
            optimizer.zero_grad()

        preds = model(video, e2, srs_input=srs_input)
        loss = criterion(preds, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if not all_targets:
        return {"loss": float("nan"), "mae": 0.0, "rmse": 0.0}

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metric_mean = target_mean if normalize_target else None
    metric_std = target_std if normalize_target else None
    metrics = compute_srs_metrics(preds, targets, metric_mean, metric_std)
    metrics["loss"] = total_loss / targets.size(0)
    return metrics


def _build_loss(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    if loss_name == "huber":
        return torch.nn.SmoothL1Loss()
    raise ValueError("Unknown loss: {}".format(loss_name))


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 PyTorch baseline")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument(
        "--tasks",
        default="task1,task2",
        help="Comma-separated task list to include (task1, task2)",
    )
    parser.add_argument(
        "--preset",
        choices=["none", "loso"],
        default="none",
        help="Preset data splits (loso = leave-one-scenario-out)",
    )
    parser.add_argument(
        "--holdout-scenario",
        default=None,
        help="Scenario ID to hold out when using --preset loso",
    )
    parser.add_argument(
        "--train-scenarios",
        default="exp1,exp2,exp3,exp4,exp5,exp6,exp7",
        help="Comma-separated list of scenario IDs",
    )
    parser.add_argument(
        "--val-scenarios",
        default="exp8",
        help="Comma-separated list of scenario IDs",
    )
    parser.add_argument(
        "--video-mode",
        choices=["rgbd", "rgb", "disparity", "none"],
        default="rgbd",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--video-tolerance", type=float, default=0.2)
    parser.add_argument("--srs-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="runs/task3_baseline")
    parser.add_argument(
        "--backbone",
        choices=["simple", "resnet18"],
        default="simple",
        help="Visual backbone for the video branch",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for the visual backbone",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the visual backbone parameters",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="mse",
        help="Loss function for regression",
    )
    parser.add_argument(
        "--use-srs-input",
        action="store_true",
        help="Include SRS at time t as an input branch",
    )
    parser.add_argument(
        "--no-target-normalize",
        action="store_false",
        dest="target_normalize",
        help="Disable target normalization",
    )
    parser.add_argument(
        "--no-srs-input-normalize",
        action="store_false",
        dest="srs_input_normalize",
        help="Disable SRS input normalization",
    )
    parser.set_defaults(target_normalize=True, srs_input_normalize=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cpu or cuda). Defaults to auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.pretrained and args.backbone == "simple":
        raise ValueError("--pretrained requires --backbone resnet18")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    index_csv = args.index_csv
    if index_csv is None:
        index_csv = Path(args.dataset_root) / "index.csv"
    index_df = load_index(index_csv, tasks=parse_tasks(args.tasks))

    train_scenarios, val_scenarios = _resolve_scenarios(args, index_df)
    (
        train_loader,
        val_loader,
        e2_columns,
        srs_dim,
        e2_mean,
        e2_std,
        srs_input_mean,
        srs_input_std,
        target_mean,
        target_std,
    ) = _build_loaders(args, index_df, train_scenarios, val_scenarios)

    model = MultiModalSRSNet(
        e2_dim=len(e2_columns),
        srs_dim=srs_dim,
        video_mode=args.video_mode,
        use_srs_input=args.use_srs_input,
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = _build_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            target_mean,
            target_std,
            optimizer=optimizer,
            device=device,
            normalize_target=args.target_normalize,
            use_srs_input=args.use_srs_input,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            target_mean,
            target_std,
            device=device,
            normalize_target=args.target_normalize,
            use_srs_input=args.use_srs_input,
        )

        print(
            "Epoch {:02d} | Train loss {:.4f} mae {:.4f} rmse {:.4f} | "
            "Val loss {:.4f} mae {:.4f} rmse {:.4f}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["mae"],
                train_metrics["rmse"],
                val_metrics["loss"],
                val_metrics["mae"],
                val_metrics["rmse"],
            )
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "e2_mean": e2_mean,
            "e2_std": e2_std,
            "e2_columns": e2_columns,
            "srs_input_mean": srs_input_mean,
            "srs_input_std": srs_input_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_normalize": args.target_normalize,
            "srs_input_normalize": args.srs_input_normalize,
            "srs_dim": srs_dim,
            "video_mode": args.video_mode,
            "image_size": args.image_size,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "use_srs_input": args.use_srs_input,
            "dt": args.dt,
            "video_tolerance": args.video_tolerance,
            "srs_tolerance": args.srs_tolerance,
            "config": vars(args),
            "train_scenarios": train_scenarios,
            "val_scenarios": val_scenarios,
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, out_dir / "best.pt")

    print("Best val loss: {:.4f}".format(best_val_loss))


if __name__ == "__main__":
    main()
