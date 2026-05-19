import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retina_gcn.data import RetinaMultiLabelDataset, detect_label_cols
from retina_gcn.graphs import (
    build_adj_from_train_labels,
    build_identity_adj,
    permute_adj,
)
from retina_gcn.models import BaselineCNN, CNNWithLabelGCN, init_bias_from_prior
from retina_gcn.reporting import (
    mean_std_aggregate,
    save_per_label_table,
    to_summary_row,
)
from retina_gcn.train import (
    ExperimentResult,
    run_baseline,
    run_gcn_full,
    run_gcn_refiner_only,
)
from retina_gcn.utils import seed_everything


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int):
    tfm_train = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tfm_eval = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm_train, tfm_eval


def parse_args():
    ap = argparse.ArgumentParser(
        description="Multi-label retinal classification with a Label-Graph GCN refiner."
    )
    ap.add_argument("--data_dir", required=True)
    ap.add_argument(
        "--backbone", default="densenet121", choices=["resnet18", "densenet121"]
    )
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Baseline epochs; FULL uses warmup + finetune within this budget.",
    )
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument(
        "--refonly_epochs",
        type=int,
        default=8,
        help="Epochs for refiner-only experiments.",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_refiner", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--label_cols", default=None)
    ap.add_argument("--img_exts", default=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp")
    ap.add_argument("--out_dir", default="runs_label_gcn_final")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--use_val_as_test", action="store_true")
    ap.add_argument(
        "--internal_val_frac",
        type=float,
        default=0.20,
        help="Fraction held out from train when no test_data.csv is provided.",
    )
    ap.add_argument("--graph_topk", type=int, default=6)
    ap.add_argument("--perm_seed", type=int, default=123)
    ap.add_argument("--run_full", action="store_true")
    ap.add_argument("--run_refonly", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.run_full and not args.run_refonly:
        args.run_full = True
        args.run_refonly = True

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train_data.csv"
    val_csv = data_dir / "val_data.csv"
    test_csv = data_dir / "test_data.csv"
    img_dir = data_dir / "images" / "images"

    assert train_csv.exists(), f"Missing: {train_csv}"
    assert val_csv.exists(), f"Missing: {val_csv}"
    assert img_dir.exists(), f"Missing: {img_dir}"

    img_exts = [e.strip() for e in args.img_exts.split(",") if e.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    amp = not args.no_amp

    train_df_full = pd.read_csv(train_csv)
    val_df_full = pd.read_csv(val_csv)
    test_df_full = pd.read_csv(test_csv) if test_csv.exists() else None

    if args.label_cols is not None:
        label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
    else:
        label_cols = detect_label_cols(train_df_full)
    if len(label_cols) == 0:
        raise ValueError("No label columns found. Pass --label_cols col1,col2,...")

    stamp = int(time.time())
    root_out = Path(args.out_dir) / f"{args.backbone}_{stamp}"
    root_out.mkdir(parents=True, exist_ok=True)

    tfm_train, tfm_eval = build_transforms(args.img_size)

    aggregate_rows = []

    for seed in seeds:
        print("\n" + "=" * 100)
        print(f"SEED: {seed}")
        seed_everything(seed)

        if test_df_full is not None:
            train_df = train_df_full.copy()
            val_df = val_df_full.copy()
            test_df = test_df_full.copy()
            split_note = "Using provided train/val/test CSVs."
        else:
            if args.use_val_as_test:
                idx = np.arange(len(train_df_full))
                rng = np.random.RandomState(seed)
                rng.shuffle(idx)
                n_val = max(1, int(args.internal_val_frac * len(train_df_full)))
                val_idx = idx[:n_val]
                tr_idx = idx[n_val:]
                train_df = train_df_full.iloc[tr_idx].copy()
                val_df = train_df_full.iloc[val_idx].copy()
                test_df = val_df_full.copy()
                split_note = (
                    "No test_data.csv. Using val_data.csv as TEST; "
                    "internal split from train_data.csv as VAL."
                )
            else:
                train_df = train_df_full.copy()
                val_df = val_df_full.copy()
                test_df = val_df_full.copy()
                split_note = (
                    "No test_data.csv. WARNING: using val_data.csv as both VAL and TEST."
                )

        print(split_note)
        print(f"Sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        Y_train = train_df[label_cols].values.astype(np.float32)
        train_freq = Y_train.mean(axis=0)
        C = len(label_cols)

        A_pmi = build_adj_from_train_labels(Y_train, mode="pmi", topk=args.graph_topk)
        A_cond = build_adj_from_train_labels(
            Y_train, mode="condprob", topk=args.graph_topk
        )
        A_id = build_identity_adj(C)
        A_perm_pmi = permute_adj(A_pmi, seed=args.perm_seed)
        A_perm_cond = permute_adj(A_cond, seed=args.perm_seed)

        seed_out = root_out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "A_pmi": A_pmi,
                "A_condprob": A_cond,
                "A_identity": A_id,
                "A_perm_pmi": A_perm_pmi,
                "A_perm_condprob": A_perm_cond,
            },
            seed_out / "adjacency_matrices.pt",
        )

        train_ds = RetinaMultiLabelDataset(
            train_df, img_dir, label_cols, transform=tfm_train, img_exts=img_exts
        )
        val_ds = RetinaMultiLabelDataset(
            val_df, img_dir, label_cols, transform=tfm_eval, img_exts=img_exts
        )
        test_ds = RetinaMultiLabelDataset(
            test_df, img_dir, label_cols, transform=tfm_eval, img_exts=img_exts
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)

        with open(seed_out / "config.json", "w") as f:
            json.dump(
                {
                    "seed": seed,
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "warmup_epochs": args.warmup_epochs,
                    "refonly_epochs": args.refonly_epochs,
                    "lr": args.lr,
                    "lr_refiner": args.lr_refiner,
                    "weight_decay": args.weight_decay,
                    "internal_val_frac": args.internal_val_frac,
                    "graph_topk": args.graph_topk,
                    "perm_seed": args.perm_seed,
                    "label_cols": label_cols,
                    "split_note": split_note,
                    "run_full": args.run_full,
                    "run_refonly": args.run_refonly,
                },
                f,
                indent=2,
            )

        results: List[ExperimentResult] = []

        print("\n" + "-" * 90)
        print("Experiment: baseline_cnn")
        baseline_dir = seed_out / "baseline_cnn"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        baseline_model = BaselineCNN(C, args.backbone)
        init_bias_from_prior(baseline_model.head, train_freq)

        baseline_res = run_baseline(
            exp_name="baseline_cnn",
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            out_dir=str(baseline_dir),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_freq=train_freq,
            amp=amp,
            selection_metric="macro_ap",
        )
        save_per_label_table(
            str(baseline_dir / "per_label_test.csv"),
            label_cols,
            train_freq,
            baseline_res.test,
        )
        results.append(baseline_res)
        baseline_ckpt = baseline_res.ckpt_path

        if args.run_full:
            full_list = [
                ("gcn_pmi_full", A_pmi),
                ("gcn_condprob_full", A_cond),
                ("gcn_identity_full", A_id),
                ("gcn_perm_pmi_full", A_perm_pmi),
                ("gcn_perm_condprob_full", A_perm_cond),
            ]
            for exp_name, A in full_list:
                print("\n" + "-" * 90)
                print(f"Experiment: {exp_name}")

                exp_dir = seed_out / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)

                model = CNNWithLabelGCN(
                    C, args.backbone, A_norm=A, gcn_hidden=64, gcn_dropout=0.2
                )

                res = run_gcn_full(
                    exp_name=exp_name,
                    gcn_model=model,
                    baseline_ckpt=baseline_ckpt,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    out_dir=str(exp_dir),
                    epochs_total=args.epochs,
                    lr_main=args.lr,
                    weight_decay=args.weight_decay,
                    train_freq=train_freq,
                    amp=amp,
                    warmup_epochs=args.warmup_epochs,
                    lr_refiner=args.lr_refiner,
                )
                save_per_label_table(
                    str(exp_dir / "per_label_test.csv"),
                    label_cols,
                    train_freq,
                    res.test,
                )
                results.append(res)

        if args.run_refonly:
            ref_list = [
                ("gcn_pmi_refonly", A_pmi),
                ("gcn_condprob_refonly", A_cond),
                ("gcn_identity_refonly", A_id),
                ("gcn_perm_pmi_refonly", A_perm_pmi),
                ("gcn_perm_condprob_refonly", A_perm_cond),
            ]
            for exp_name, A in ref_list:
                print("\n" + "-" * 90)
                print(f"Experiment: {exp_name}")

                exp_dir = seed_out / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)

                model = CNNWithLabelGCN(
                    C, args.backbone, A_norm=A, gcn_hidden=64, gcn_dropout=0.2
                )

                res = run_gcn_refiner_only(
                    exp_name=exp_name,
                    gcn_model=model,
                    baseline_ckpt=baseline_ckpt,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    out_dir=str(exp_dir),
                    epochs=args.refonly_epochs,
                    lr_refiner=args.lr_refiner,
                    weight_decay=args.weight_decay,
                    train_freq=train_freq,
                    amp=amp,
                )
                save_per_label_table(
                    str(exp_dir / "per_label_test.csv"),
                    label_cols,
                    train_freq,
                    res.test,
                )
                results.append(res)

        summary_rows = [to_summary_row(r) for r in results]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(seed_out / "summary.csv", index=False)
        with open(seed_out / "summary.json", "w") as f:
            json.dump(summary_rows, f, indent=2)

        print("\n[Seed summary]")
        cols_to_show = [
            "exp",
            "mode",
            "best_epoch",
            "test_macro_ap",
            "test_macro_f1",
            "test_micro_f1",
            "rare_ap",
            "medium_ap",
            "frequent_ap",
        ]
        print(summary_df[cols_to_show])

        for row in summary_rows:
            row2 = dict(row)
            row2["seed"] = seed
            aggregate_rows.append(row2)

    agg_df = pd.DataFrame(aggregate_rows)
    agg_df.to_csv(root_out / "all_seeds_raw.csv", index=False)

    keys = [
        "test_macro_ap",
        "test_macro_auc",
        "test_macro_f1",
        "test_micro_f1",
        "rare_ap",
        "medium_ap",
        "frequent_ap",
    ]

    exp_names = sorted(agg_df["exp"].unique().tolist())
    agg_rows = []
    for exp in exp_names:
        sub = agg_df[agg_df["exp"] == exp].to_dict(orient="records")
        stats = mean_std_aggregate(sub, keys)
        stats["exp"] = exp
        stats["mode"] = sub[0]["mode"] if len(sub) else ""
        stats["n_seeds"] = int(len(sub))
        agg_rows.append(stats)

    agg_summary = pd.DataFrame(agg_rows)
    agg_summary = agg_summary[
        ["exp", "mode", "n_seeds"]
        + [k + "_mean" for k in keys]
        + [k + "_std" for k in keys]
    ]
    agg_summary.to_csv(root_out / "aggregate_summary_mean_std.csv", index=False)

    for mode in ["full", "refonly"]:
        sub_df = agg_df[agg_df["mode"] == mode]
        if len(sub_df) == 0:
            continue
        exp_names_m = sorted(sub_df["exp"].unique().tolist())
        rows_m = []
        for exp in exp_names_m:
            sub = sub_df[sub_df["exp"] == exp].to_dict(orient="records")
            stats = mean_std_aggregate(sub, keys)
            stats["exp"] = exp
            stats["mode"] = mode
            stats["n_seeds"] = int(len(sub))
            rows_m.append(stats)
        out_m = pd.DataFrame(rows_m)
        out_m = out_m[
            ["exp", "mode", "n_seeds"]
            + [k + "_mean" for k in keys]
            + [k + "_std" for k in keys]
        ]
        out_m.to_csv(root_out / f"aggregate_summary_{mode}.csv", index=False)

    print("\n" + "=" * 100)
    print("DONE.")
    print("Outputs saved to:", str(root_out))
    print(
        "Main aggregate table:",
        str(root_out / "aggregate_summary_mean_std.csv"),
    )
    if (root_out / "aggregate_summary_full.csv").exists():
        print("Full table:", str(root_out / "aggregate_summary_full.csv"))
    if (root_out / "aggregate_summary_refonly.csv").exists():
        print(
            "Refiner-only table (ablation):",
            str(root_out / "aggregate_summary_refonly.csv"),
        )


if __name__ == "__main__":
    main()
