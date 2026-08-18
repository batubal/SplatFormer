#!/usr/bin/env python3
"""
Re-export already-trained LR splatfacto runs into *separate* SplatFormer eval
folders. Does not retrain.

The SLURM job writes LR PLYs / .work dirs under distinct OUTPUT_ROOTs, but every
setting exported to the same ``test-set/customOOD``. This script walks each
trained ``.work/<stem>/`` tree and writes:

    test-set/customOOD_<setting>/{nerfstudio,colmap}/<stem>/

Default settings match ``train_lr_splats_splatfacto.slurm`` comments:

    dense_x2     /arf/scratch/batubal/lr_data_splatformer
    sparse4_x2   .../lr_data_splatformer_sparse4_x2
    dense_x4     .../lr_data_splatformer_x4
    sparse4_x4   .../lr_data_splatformer_sparse4_x4

Examples:
    python reexport_splatformer_by_setting.py --dry_run
    python reexport_splatformer_by_setting.py --setting dense_x2 --setting sparse4_x2
    python reexport_splatformer_by_setting.py --restore_custom_ood dense_x2

Then evaluate one setting:
    bash scripts/train-on-custom-lr_inference.sh dense_x2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Iterator

from lr_experiment_settings import (
    REPO_DIR,
    apply_output_root_maps,
    default_experiments,
    filter_experiments,
    scratch_dir,
)
from lr_splat_helpers import patch_numpy_pickle_aliases

patch_numpy_pickle_aliases()
from train_lr_splats_splatfacto import (
    _find_latest_ckpt,
    _find_latest_config,
    export_splatformer_scene,
)


def custom_lr_gin_text(splatformer_root: str) -> str:
    """Gin pointing train+test loaders at one SplatFormer export root."""
    rel = os.path.relpath(os.path.abspath(splatformer_root), REPO_DIR)
    if rel.startswith(".."):
        rel = os.path.abspath(splatformer_root)
    ns = f"{rel}/nerfstudio".replace("\\", "/")
    colmap = f"{rel}/colmap".replace("\\", "/")
    return f"""# Auto-written by reexport_splatformer_by_setting.py
# SplatFormer eval root: {rel}

SplatfactoDataset.remove_outlier_ndevs=-1
SplatfactoDataset.image_per_scene = None
SplatfactoDataset.sample_ratio_test = None
SplatfactoDataset.max_gs_num = 100000
SplatfactoDataset.load_pose_src = 'nerfstudio'
SplatfactoDataset.background_color = [255, 255, 255]
SplatfactoDataset.max_test_images = 8
SplatfactoDataset.test_sample_seed = 42

collate_fn = @GS_collate_fn
build_trainloader.batch_size = 1
build_trainloader.accumulate_step = 1
build_trainloader.num_workers = 4
build_trainloader.collate_fn = %collate_fn
train_dataset/SplatfactoDataset.train_or_test = 'train'
train_dataset/SplatfactoDataset.nerfstudio_folder = '{ns}'
train_dataset/SplatfactoDataset.colmap_folder = '{colmap}'
train_dataset/SplatfactoDataset.image_per_scene = 4
train_dataset/SplatfactoDataset.sample_ratio_test = 0.7
train_dataset/SplatfactoDataset.cache_steps = 1
train_dataset/SplatfactoDataset.cache_num_scenes = 1
train_dataset/SplatfactoDataset.split_across_gpus = True

build_testloader.batch_size = 1
build_testloader.num_workers = 0
build_testloader.collate_fn = %collate_fn
test_dataset/SplatfactoDataset.train_or_test = 'test'
test_dataset/SplatfactoDataset.nerfstudio_folder = {{
  'custom_lr':'{ns}',
}}
test_dataset/SplatfactoDataset.colmap_folder = {{
   'custom_lr':'{colmap}',
}}
test_dataset/SplatfactoDataset.cache_steps = 1
test_dataset/SplatfactoDataset.cache_num_scenes = 1
test_dataset/SplatfactoDataset.split_across_gpus = False
test_dataset/SplatfactoDataset.max_test_images = 8
test_dataset/SplatfactoDataset.test_sample_seed = 42
"""


def write_gin(path: str, splatformer_root: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(custom_lr_gin_text(splatformer_root))


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_hr_ply(work_dir: str, category_id: str, stem: str, root_path: str) -> str | None:
    manifest = _load_json(os.path.join(work_dir, "manifest.json")) or {}
    candidate = manifest.get("hr_ply")
    if isinstance(candidate, str) and os.path.isfile(candidate):
        return candidate
    direct = os.path.join(root_path, category_id, f"{stem}.ply")
    if os.path.isfile(direct):
        return direct
    cat_dir = os.path.join(root_path, category_id)
    if os.path.isdir(cat_dir):
        for name in os.listdir(cat_dir):
            if name.startswith(stem) and name.endswith(".ply"):
                return os.path.join(cat_dir, name)
    return None


def _scene_ready(splatformer_root: str, stem: str) -> bool:
    ns = os.path.join(splatformer_root, "nerfstudio", stem, "splatfacto")
    images = os.path.join(splatformer_root, "colmap", stem, "images")
    pkl = os.path.join(ns, "camera_for-3d-denoise.pkl")
    ckpt_dir = os.path.join(ns, "nerfstudio_models")
    if not (os.path.isfile(pkl) and os.path.isdir(ckpt_dir) and os.path.isdir(images)):
        return False
    has_ckpt = any(n.startswith("step-") and n.endswith(".ckpt") for n in os.listdir(ckpt_dir))
    names = os.listdir(images)
    has_train = any(n.lower().startswith("train") for n in names)
    has_test = any(n.lower().startswith("test") or n.lower().startswith("frame_eval") for n in names)
    return has_ckpt and has_train and has_test


def iter_trained_scenes(output_root: str, category_id: str | None = None) -> Iterator[dict[str, str]]:
    """Yield scenes that already have a splatfacto ckpt + nerf_dataset (no retrain)."""
    if not os.path.isdir(output_root):
        return
    cats = [category_id] if category_id else sorted(os.listdir(output_root))
    for cat in cats:
        if not cat or cat.startswith("."):
            continue
        work_root = os.path.join(output_root, cat, ".work")
        if not os.path.isdir(work_root):
            continue
        for stem in sorted(os.listdir(work_root)):
            work_dir = os.path.join(work_root, stem)
            if not os.path.isdir(work_dir):
                continue
            dataset_dir = os.path.join(work_dir, "nerf_dataset")
            ns_out_dir = os.path.join(work_dir, "ns_outputs")
            transforms = os.path.join(dataset_dir, "transforms_train.json")
            if not os.path.isfile(transforms):
                continue
            config_yml = _find_latest_config(ns_out_dir)
            ckpt_path = _find_latest_ckpt(config_yml, ns_out_dir)
            if not ckpt_path:
                continue
            yield {
                "category_id": cat,
                "stem": stem,
                "work_dir": work_dir,
                "dataset_dir": dataset_dir,
                "ns_out_dir": ns_out_dir,
                "config_yml": config_yml or "",
                "ckpt_path": ckpt_path,
            }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-export trained LR splatfacto checkpoints into per-setting "
            "SplatFormer eval folders (no retraining)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=None,
        help="Setting name to export (repeatable). Default: all four slurm settings.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=None,
        metavar="NAME=OUTPUT_ROOT",
        help="Override a setting's trained OUTPUT_ROOT, e.g. dense_x2=/path/to/lr_data.",
    )
    parser.add_argument("--scratch", type=str, default=scratch_dir())
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.path.join(scratch_dir(), "data"),
        help="HR PLY root, used only to re-render Stage-2 orbit test GT.",
    )
    parser.add_argument("--category_id", type=str, default=None)
    parser.add_argument(
        "--test_camera_mode",
        type=str,
        default="stage2_orbit",
        choices=["stage2_orbit", "held_out"],
    )
    parser.add_argument("--test_image_size", type=int, default=256)
    parser.add_argument("--num_test_views", type=int, default=8)
    parser.add_argument("--focal_length", type=float, default=500.0)
    parser.add_argument("--background", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--render_device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--render_backend", type=str, default="auto", choices=["auto", "gsplat", "torch"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite scenes that already exist under the per-setting eval root.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--write_gin",
        action="store_true",
        default=True,
        help="Write configs/dataset/custom_lr_<setting>.gin for each setting.",
    )
    parser.add_argument("--no_write_gin", action="store_false", dest="write_gin")
    parser.add_argument(
        "--restore_custom_ood",
        type=str,
        default=None,
        metavar="SETTING",
        help="Also export this setting into test-set/customOOD (legacy inference path).",
    )
    return parser.parse_args()


def _export_one(
    scene: dict[str, str],
    dest_root: str,
    args: argparse.Namespace,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    hr_ply = _resolve_hr_ply(scene["work_dir"], scene["category_id"], scene["stem"], args.root_path)
    test_mode = args.test_camera_mode
    if test_mode == "stage2_orbit" and not hr_ply:
        print(
            f"  WARNING: no HR PLY for {scene['stem']}; falling back to held_out test cameras"
        )
        test_mode = "held_out"

    meta = _load_json(os.path.join(scene["dataset_dir"], "dataset_meta.json")) or {}
    fx = float(args.focal_length)
    hr_sh = meta.get("hr_sh_degree")
    hr_sh_degree = int(hr_sh) if isinstance(hr_sh, (int, float)) else None

    return export_splatformer_scene(
        scene_name=scene["stem"],
        splatformer_root=dest_root,
        ckpt_path=scene["ckpt_path"],
        dataset_dir=scene["dataset_dir"],
        config_yml=scene["config_yml"] or None,
        num_test_views=args.num_test_views,
        fx=fx,
        fy=fx,
        overwrite=overwrite,
        hr_ply=hr_ply,
        test_camera_mode=test_mode,
        test_image_size=args.test_image_size,
        render_device=args.render_device,
        render_backend=args.render_backend,
        background=args.background,
        hr_sh_degree=hr_sh_degree,
        seed=42,
    )


def main() -> None:
    args = _parse_args()
    experiments = default_experiments(scratch=args.scratch, repo_dir=REPO_DIR)
    experiments = apply_output_root_maps(experiments, args.map)
    experiments = filter_experiments(experiments, args.setting)

    restore = args.restore_custom_ood
    known_names = [e.name for e in default_experiments(args.scratch)]
    if restore and restore not in known_names:
        raise SystemExit(f"--restore_custom_ood must be one of {known_names}")
    if restore and restore not in {e.name for e in experiments}:
        raise SystemExit(
            f"--restore_custom_ood {restore} is not in this run; add --setting {restore}"
        )

    print(f"Repo: {REPO_DIR}")
    print(f"HR root: {args.root_path}")
    print(f"Mode: {args.test_camera_mode}  overwrite={args.overwrite}  dry_run={args.dry_run}")

    totals = {"ok": 0, "exists": 0, "skip": 0, "fail": 0, "missing_root": 0}

    for exp in experiments:
        print("\n" + "=" * 72)
        print(f"Setting {exp.name}")
        print(f"  trained OUTPUT_ROOT: {exp.output_root}")
        print(f"  eval splatformer_root: {exp.splatformer_root}")
        if args.write_gin and not args.dry_run:
            write_gin(exp.gin_file, exp.splatformer_root)
            print(f"  gin: {os.path.relpath(exp.gin_file, REPO_DIR)}")

        if not os.path.isdir(exp.output_root):
            print("  MISSING output_root (this setting was not trained, or used a different path)")
            print("  If the path differs, re-run with --map "
                  f"{exp.name}=/path/to/that/OUTPUT_ROOT")
            totals["missing_root"] += 1
            continue

        scenes = list(iter_trained_scenes(exp.output_root, args.category_id))
        print(f"  trained scenes with ckpt: {len(scenes)}")
        if not scenes:
            print("  nothing to export (need <cat>/.work/<stem>/ns_outputs + nerf_dataset)")
            continue

        extra_dests = []
        if restore == exp.name:
            extra_dests.append(os.path.join(REPO_DIR, "test-set", "customOOD"))

        for i, scene in enumerate(scenes, 1):
            dests = [exp.splatformer_root, *extra_dests]
            for dest in dests:
                label = os.path.relpath(dest, REPO_DIR) if dest.startswith(REPO_DIR) else dest
                if not args.overwrite and _scene_ready(dest, scene["stem"]):
                    print(f"  [{i}/{len(scenes)}] {scene['stem']} exists → {label}")
                    totals["exists"] += 1
                    continue
                if args.dry_run:
                    print(
                        f"  [{i}/{len(scenes)}] DRY {scene['stem']} "
                        f"ckpt={os.path.basename(scene['ckpt_path'])} → {label}"
                    )
                    totals["skip"] += 1
                    continue
                try:
                    info = _export_one(scene, dest, args, overwrite=args.overwrite or not _scene_ready(dest, scene["stem"]))
                    status = info.get("status", "ok")
                    print(
                        f"  [{i}/{len(scenes)}] {scene['stem']} {status} "
                        f"train={info.get('num_train_views', '?')} → {label}"
                    )
                    totals["ok" if status != "exists" else "exists"] += 1
                except Exception as exc:  # noqa: BLE001
                    totals["fail"] += 1
                    print(f"  [{i}/{len(scenes)}] FAILED {scene['stem']}: {exc}")
                    traceback.print_exc()

    print("\n" + "=" * 72)
    print(
        f"Done. ok={totals['ok']} exists={totals['exists']} "
        f"dry/skip={totals['skip']} fail={totals['fail']} "
        f"missing_output_root={totals['missing_root']}"
    )
    print("\nEvaluate a setting (from repo root):")
    for exp in default_experiments(args.scratch):
        if args.setting and exp.name not in set(args.setting):
            continue
        print(f"  bash scripts/train-on-custom-lr_inference.sh {exp.name}")
    if restore:
        print("  bash scripts/train-on-custom-lr_inference.sh   # legacy test-set/customOOD")

    if totals["fail"] or totals["missing_root"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
