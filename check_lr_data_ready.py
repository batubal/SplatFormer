#!/usr/bin/env python3
"""
Check whether LR splatfacto outputs are ready for each setting × category.

Default settings match ``train_lr_splats_splatfacto.slurm``:

    dense_x2     $SCRATCH/lr_data_splatformer
    sparse4_x2   $SCRATCH/lr_data_splatformer_sparse4_x2
    dense_x4     $SCRATCH/lr_data_splatformer_x4
    sparse4_x4   $SCRATCH/lr_data_splatformer_sparse4_x4

For every expected test-split sample (≤ --num_samples per category) reports:

    ply   LR splat at <output_root>/<cat>/<stem>.ply
    ckpt  splatfacto checkpoint + nerf_dataset under .work/<stem>/
    sf    SplatFormer eval export under test-set/customOOD_<setting>/

Examples:
    python check_lr_data_ready.py
    python check_lr_data_ready.py --setting dense_x2 --setting sparse4_x2
    python check_lr_data_ready.py --verbose
    python check_lr_data_ready.py --map dense_x2=/path/to/lr_data_splatformer
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from lr_experiment_settings import (
    apply_output_root_maps,
    default_experiments,
    filter_experiments,
    scratch_dir,
)
from lr_splat_helpers import (
    compute_dataset_split_indices,
    list_category_split_ply_files,
    resolve_split_metadata_path,
)


def _parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    gs_root = os.path.abspath(os.path.join(here, "..", "gaussian_sr"))
    default_cache = os.path.join(gs_root, "dataset_percentile_dynamic_v2_store", "meta.pt")
    if not os.path.isfile(default_cache):
        default_cache = os.path.join(gs_root, "meta.pt")

    scratch = scratch_dir()
    parser = argparse.ArgumentParser(
        description="Check LR PLY / ckpt / SplatFormer readiness per setting and category.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_cache",
        type=str,
        default=default_cache,
        help="gaussian_sr meta.pt used for stratified splits.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Expected max samples per category (same as SLURM NUM_SAMPLES).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.path.join(scratch, "data"),
        help="HR data root (used to build expected stems).",
    )
    parser.add_argument("--scratch", type=str, default=scratch)
    parser.add_argument(
        "--setting",
        action="append",
        default=None,
        help="Setting name to check (repeatable). Default: all four slurm settings.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=None,
        metavar="NAME=OUTPUT_ROOT",
        help="Override a setting's trained OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--output_root",
        action="append",
        default=None,
        help="Legacy: extra output roots to check as unlabeled folders (no sf path).",
    )
    parser.add_argument(
        "--category_id",
        type=str,
        default=None,
        help="If set, only check this category.",
    )
    parser.add_argument(
        "--check_splatformer",
        action="store_true",
        default=True,
        help="Also require SplatFormer export files under each setting's eval root.",
    )
    parser.add_argument(
        "--no_check_splatformer",
        action="store_false",
        dest="check_splatformer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print missing stems per category.",
    )
    return parser.parse_args()


def _expected_stems_by_category(
    dataset_cache: str,
    root_path: str,
    *,
    split: str,
    seed: int,
    num_samples: int,
    offset: int,
    category_id: str | None,
) -> dict[str, list[str]]:
    train_idx, val_idx, test_idx, filenames = compute_dataset_split_indices(
        dataset_cache, seed=seed
    )
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    if category_id:
        categories = [category_id]
    else:
        seen: set[str] = set()
        categories = []
        for i in split_map[split]:
            cat = filenames[i].replace("\\", "/").split("/", 1)[0]
            if cat and not cat.startswith(".") and cat not in seen:
                seen.add(cat)
                categories.append(cat)
        categories = sorted(categories)

    out: dict[str, list[str]] = {}
    for cat in categories:
        ply_files = list_category_split_ply_files(
            root_path,
            cat,
            dataset_cache,
            split=split,
            seed=seed,
        )
        if num_samples > 0:
            sliced = ply_files[offset : offset + num_samples]
        else:
            sliced = ply_files[offset:]
        out[cat] = [os.path.splitext(os.path.basename(p))[0] for p in sliced]
    return out


def _splatformer_ready(splatformer_root: str, stem: str) -> bool:
    ns = os.path.join(splatformer_root, "nerfstudio", stem, "splatfacto")
    colmap = os.path.join(splatformer_root, "colmap", stem)
    ckpt_dir = os.path.join(ns, "nerfstudio_models")
    pkl = os.path.join(ns, "camera_for-3d-denoise.pkl")
    images = os.path.join(colmap, "images")
    if not (os.path.isfile(pkl) and os.path.isdir(ckpt_dir) and os.path.isdir(images)):
        return False
    has_ckpt = any(
        name.startswith("step-") and name.endswith(".ckpt") for name in os.listdir(ckpt_dir)
    )
    names = os.listdir(images)
    has_train = any(name.lower().startswith("train") for name in names)
    has_test = any(
        name.lower().startswith("test") or name.lower().startswith("frame_eval") for name in names
    )
    return has_ckpt and has_train and has_test


def _ckpt_ready(output_root: str, category_id: str, stem: str) -> bool:
    work_dir = os.path.join(output_root, category_id, ".work", stem)
    transforms = os.path.join(work_dir, "nerf_dataset", "transforms_train.json")
    if not os.path.isfile(transforms):
        return False
    ns_out = os.path.join(work_dir, "ns_outputs")
    if not os.path.isdir(ns_out):
        return False
    pattern = os.path.join(ns_out, "**", "nerfstudio_models", "step-*.ckpt")
    return bool(glob.glob(pattern, recursive=True))


def _ply_ready(output_root: str, category_id: str, stem: str) -> bool:
    return os.path.isfile(os.path.join(output_root, category_id, f"{stem}.ply"))


def _fmt_ratio(ready: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{ready}/{total}"


def check_one_setting(
    label: str,
    output_root: str,
    expected: dict[str, list[str]],
    *,
    splatformer_root: str | None,
    check_splatformer: bool,
    verbose: bool,
) -> dict[str, dict[str, int]]:
    """
    Return per-category counts:
      {cat: {n, ply, ckpt, sf}}
    """
    print(f"\n=== {label} ===")
    print(f"output_root: {output_root}")
    if splatformer_root:
        print(f"splatformer_root: {splatformer_root}")

    per_cat: dict[str, dict[str, int]] = {}
    root_ok = os.path.isdir(output_root)
    if not root_ok:
        print("  MISSING output_root directory")

    n_exp = n_ply = n_ckpt = n_sf = 0
    incomplete_cats = 0
    for cat, stems in expected.items():
        missing_ply: list[str] = []
        missing_ckpt: list[str] = []
        missing_sf: list[str] = []
        ply = ckpt = sf = 0
        for stem in stems:
            if root_ok and _ply_ready(output_root, cat, stem):
                ply += 1
            else:
                missing_ply.append(stem)
            if root_ok and _ckpt_ready(output_root, cat, stem):
                ckpt += 1
            else:
                missing_ckpt.append(stem)
            if check_splatformer and splatformer_root:
                if _splatformer_ready(splatformer_root, stem):
                    sf += 1
                else:
                    missing_sf.append(stem)

        n = len(stems)
        n_exp += n
        n_ply += ply
        n_ckpt += ckpt
        n_sf += sf
        per_cat[cat] = {"n": n, "ply": ply, "ckpt": ckpt, "sf": sf}

        ply_ok = ply == n
        ckpt_ok = ckpt == n
        sf_ok = (not check_splatformer) or (sf == n)
        status = "OK" if (root_ok and ply_ok and ckpt_ok and sf_ok) else "INCOMPLETE"
        if status != "OK":
            incomplete_cats += 1

        line = (
            f"  {cat}: ply {_fmt_ratio(ply, n)}  ckpt {_fmt_ratio(ckpt, n)}"
        )
        if check_splatformer:
            line += f"  sf {_fmt_ratio(sf, n)}"
        line += f"  [{status}]"
        print(line)
        if verbose and missing_ply:
            shown = missing_ply[:5]
            extra = " ..." if len(missing_ply) > 5 else ""
            print(f"    missing ply ({len(missing_ply)}): {shown}{extra}")
        if verbose and missing_ckpt:
            shown = missing_ckpt[:5]
            extra = " ..." if len(missing_ckpt) > 5 else ""
            print(f"    missing ckpt ({len(missing_ckpt)}): {shown}{extra}")
        if verbose and check_splatformer and missing_sf:
            shown = missing_sf[:5]
            extra = " ..." if len(missing_sf) > 5 else ""
            print(f"    missing sf ({len(missing_sf)}): {shown}{extra}")

    total_line = (
        f"  TOTAL: ply {_fmt_ratio(n_ply, n_exp)}  ckpt {_fmt_ratio(n_ckpt, n_exp)}"
    )
    if check_splatformer:
        total_line += f"  sf {_fmt_ratio(n_sf, n_exp)}"
    total_line += f"  incomplete_categories={incomplete_cats}/{len(expected)}"
    print(total_line)
    return per_cat


def _print_matrix(
    title: str,
    setting_names: list[str],
    categories: list[str],
    rows: dict[str, dict[str, dict[str, int]]],
    field: str,
) -> None:
    """Print category × setting table of field/n."""
    col_w = max(10, max((len(n) for n in setting_names), default=10))
    cat_w = max(10, max((len(c) for c in categories), default=10))
    header = f"{'category':<{cat_w}}  " + "  ".join(f"{n:>{col_w}}" for n in setting_names)
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for cat in categories:
        cells = []
        for name in setting_names:
            counts = rows.get(name, {}).get(cat)
            if not counts:
                cells.append(f"{'—':>{col_w}}")
            else:
                cells.append(f"{_fmt_ratio(counts[field], counts['n']):>{col_w}}")
        print(f"{cat:<{cat_w}}  " + "  ".join(cells))

    totals = []
    for name in setting_names:
        cat_counts = rows.get(name, {})
        ready = sum(v[field] for v in cat_counts.values())
        total = sum(v["n"] for v in cat_counts.values())
        totals.append(f"{_fmt_ratio(ready, total):>{col_w}}")
    print("-" * len(header))
    print(f"{'TOTAL':<{cat_w}}  " + "  ".join(totals))


def main() -> None:
    args = _parse_args()
    resolve_split_metadata_path(args.dataset_cache)

    expected = _expected_stems_by_category(
        args.dataset_cache,
        args.root_path,
        split=args.split,
        seed=args.seed,
        num_samples=args.num_samples,
        offset=args.offset,
        category_id=args.category_id,
    )
    categories = list(expected)
    n_expected = sum(len(v) for v in expected.values())
    print(
        f"Expecting ≤{args.num_samples} '{args.split}' samples/category "
        f"from {args.dataset_cache}"
    )
    print(f"Categories: {len(expected)}  total expected samples: {n_expected}")

    experiments = default_experiments(scratch=args.scratch)
    experiments = apply_output_root_maps(experiments, args.map)
    experiments = filter_experiments(experiments, args.setting)

    all_ok = True
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    setting_names: list[str] = []

    for exp in experiments:
        setting_names.append(exp.name)
        per_cat = check_one_setting(
            exp.name,
            exp.output_root,
            expected,
            splatformer_root=exp.splatformer_root,
            check_splatformer=args.check_splatformer,
            verbose=args.verbose,
        )
        matrix[exp.name] = per_cat
        for counts in per_cat.values():
            if counts["ply"] < counts["n"] or counts["ckpt"] < counts["n"]:
                all_ok = False
            if args.check_splatformer and counts["sf"] < counts["n"]:
                all_ok = False

    if args.output_root:
        for i, root in enumerate(args.output_root):
            label = os.path.basename(root.rstrip("/")) or f"extra_{i}"
            if label in matrix:
                label = f"{label}_{i}"
            setting_names.append(label)
            per_cat = check_one_setting(
                label,
                root,
                expected,
                splatformer_root=None,
                check_splatformer=False,
                verbose=args.verbose,
            )
            matrix[label] = per_cat
            for counts in per_cat.values():
                if counts["ply"] < counts["n"] or counts["ckpt"] < counts["n"]:
                    all_ok = False

    _print_matrix("PLY ready (setting × category)", setting_names, categories, matrix, "ply")
    _print_matrix("CKPT ready (setting × category)", setting_names, categories, matrix, "ckpt")
    if args.check_splatformer:
        _print_matrix(
            "SplatFormer export ready (setting × category)",
            [e.name for e in experiments],
            categories,
            matrix,
            "sf",
        )

    if all_ok:
        print("\nAll requested LR outputs are ready.")
        sys.exit(0)
    print("\nSome LR outputs are still missing.")
    sys.exit(1)


if __name__ == "__main__":
    main()
