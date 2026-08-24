#!/usr/bin/env python3
"""
Train LR Gaussian splats on ShapeNet HR splats via nerfstudio splatfacto,
and export each scene in SplatFormer eval format.

Matches gaussian_sr ``train_lr_splats_splatfacto.py`` training config by default:
  72 hemisphere views @ 400px HR, camera-res-scale-factor = 0.5 (200px LR),
  splatfacto sh_degree=0, 20k iters, cull_alpha_thresh=0.15.

Category mode (``--category_id``) selects the gaussian_sr stratified **test**
split for that category (70/15/15, seed=42) via ``--dataset_cache`` metadata.
Already-rendered NeRF datasets under ``--reuse_render_root`` (e.g. gaussian_sr
``lr_data/<cat>/.work/<stem>/nerf_dataset``) are reused to skip HR rendering.

Pipeline per sample:
  1. Render the HR splat from hemisphere cameras (or reuse an existing dataset).
  2. Write a NeRF Synthetic-style dataset with all 72 views in transforms_train
     (gaussian_sr-compatible); val/test are evenly spaced subsets for ns-eval.
  3. ns-train splatfacto (blender-data) with camera-res-scale-factor = LR/HR.
  4. ns-export gaussian-splat → lr_data/<category>/<stem>.ply
  5. Optional ns-eval at HR on held-out test views; append metrics to metrics.json.
  6. Export SplatFormer-ready scene under --splatformer_root:
       <root>/nerfstudio/<scene>/splatfacto/
         nerfstudio_models/step-*.ckpt
         camera_for-3d-denoise.pkl
         dataparser_transforms.json
       <root>/colmap/<scene>/
         images/{train_XXX,test_XXX}.png
         sparse/0/{cameras,images,bbox}.txt
     Default test cams are Stage-2 8-view orbit @ 256px (compare_with_input ~21 PSNR).

Usage:
    # Needs nerfstudio + gsplat (or torch backend) in the active env.
    # Single HR ply:
    python train_lr_splats_splatfacto.py \\
        --hr_ply data/02691156/02691156-xxx.ply

    # Category test-split batch (gaussian_sr split + reuse renders):
    python train_lr_splats_splatfacto.py \\
        --category_id 02691156 --offset 0 --num_samples 10 \\
        --dataset_cache ../gaussian_sr/meta.pt \\
        --reuse_render_root ../gaussian_sr/lr_data

    # Re-export SplatFormer test cams only (no LR retrain):
    python train_lr_splats_splatfacto.py \\
        --category_id 02691156 --num_samples 10 --reexport_splatformer \\
        --test_camera_mode stage2_orbit --test_image_size 256

    # Then evaluate with SplatFormer:
    #   configs/dataset/custom_lr.gin  (points at --splatformer_root)
    #   sh scripts/train-on-custom-lr_inference.sh
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lr_splat_helpers import (
    _hemisphere_unit_directions,
    find_existing_nerf_dataset,
    list_category_ply_files,
    list_category_split_ply_files,
    load_metrics_json,
    load_ply,
    merge_metrics_json_entry,
    render_hr_lr_targets,
    render_hr_stage2_orbit_views,
    resolve_hr_sh_degree,
    resolve_render_setup,
    select_category_batch_items,
    set_seed,
)


# ---------------------------------------------------------------------------
# nerfstudio CLI helpers (same as run_splatfacto_nerf.py)
# ---------------------------------------------------------------------------


def _find_ns_cmd(name: str) -> list[str]:
    found = shutil.which(name)
    if found:
        return [found]
    py_bin = Path(sys.executable).parent
    candidate = py_bin / name
    if candidate.exists():
        return [str(candidate)]
    module_map = {
        "ns-train": "nerfstudio.scripts.train",
        "ns-export": "nerfstudio.scripts.exporter",
        "ns-eval": "nerfstudio.scripts.eval",
    }
    if name in module_map:
        check = subprocess.run(
            [sys.executable, "-c", f"import {module_map[name]}"],
            capture_output=True,
        )
        if check.returncode == 0:
            return [sys.executable, "-m", module_map[name]]
    raise FileNotFoundError(
        f"'{name}' not found. Install nerfstudio in this environment: pip install nerfstudio"
    )


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _find_latest_config(ns_output_dir: str) -> str | None:
    pattern = os.path.join(ns_output_dir, "**", "config.yml")
    configs = sorted(glob.glob(pattern, recursive=True))
    return configs[-1] if configs else None


def _find_latest_ply(ns_export_dir: str) -> str | None:
    for name in ("splat.ply", "gaussian_splat.ply"):
        path = os.path.join(ns_export_dir, name)
        if os.path.exists(path):
            return path
    plys = glob.glob(os.path.join(ns_export_dir, "*.ply"))
    return plys[0] if plys else None


def _find_latest_ckpt(config_yml: str | None, ns_output_dir: str) -> str | None:
    search_roots: list[str] = []
    if config_yml:
        search_roots.append(os.path.dirname(config_yml))
    search_roots.append(ns_output_dir)
    ckpts: list[str] = []
    for root in search_roots:
        ckpts.extend(glob.glob(os.path.join(root, "**", "nerfstudio_models", "step-*.ckpt"), recursive=True))
        ckpts.extend(glob.glob(os.path.join(root, "nerfstudio_models", "step-*.ckpt")))
    ckpts = sorted(set(ckpts))
    return ckpts[-1] if ckpts else None


def _load_dataparser_transforms(config_yml: str | None) -> dict[str, Any]:
    """Identity default; override from the ns-train run when present."""
    default = {
        "transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "scale": 1.0,
    }
    if not config_yml:
        return default
    candidate = os.path.join(os.path.dirname(config_yml), "dataparser_transforms.json")
    if not os.path.isfile(candidate):
        return default
    with open(candidate, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "transform": data.get("transform", default["transform"]),
        "scale": float(data.get("scale", 1.0)),
    }


def _as_c2w_4x4(c2w: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape == (4, 4):
        return c2w
    if c2w.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = c2w
        return out
    raise ValueError(f"Expected c2w shape (3,4) or (4,4), got {c2w.shape}")


def _apply_dataparser_to_c2ws(
    c2ws_opengl: np.ndarray,
    transform: list[list[float]] | np.ndarray,
    scale: float,
) -> np.ndarray:
    """Apply nerfstudio dataparser transform+scale so cameras match ckpt means."""
    T = np.eye(4, dtype=np.float64)
    transform_np = np.asarray(transform, dtype=np.float64)
    if transform_np.shape == (3, 4):
        T[:3, :4] = transform_np
    elif transform_np.shape == (4, 4):
        T = transform_np
    else:
        raise ValueError(f"Unexpected dataparser transform shape {transform_np.shape}")

    out = []
    for c2w in c2ws_opengl:
        M = _as_c2w_4x4(c2w)
        M2 = T @ M
        M2[:3, 3] *= float(scale)
        out.append(M2[:3, :4].astype(np.float32))
    return np.stack(out, axis=0)


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) from 3x3 rotation (Eigen / scalar-first)."""
    R = np.asarray(R, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    if q[0] < 0:
        q *= -1.0
    return q


def _opengl_c2w_to_colmap_w2c(c2w_opengl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (qvec_wxyz, tvec) for COLMAP images.txt from OpenGL c2w."""
    c2w_cv = _as_c2w_4x4(c2w_opengl).copy()
    c2w_cv[:3, 1:3] *= -1.0  # OpenGL -> OpenCV
    w2c = np.linalg.inv(c2w_cv)
    qvec = _rotmat_to_qvec(w2c[:3, :3])
    tvec = w2c[:3, 3]
    return qvec, tvec


def _resolve_nerf_frame_image(dataset_dir: str, rel: str) -> str:
    """Resolve a NeRF Synthetic file_path (often without extension) to an image on disk."""
    candidates = [
        os.path.join(dataset_dir, rel),
        os.path.join(dataset_dir, rel + ".png"),
        os.path.join(dataset_dir, rel + ".jpg"),
    ]
    img_path = next((p for p in candidates if os.path.isfile(p)), None)
    if img_path is None:
        raise FileNotFoundError(f"Image for frame {rel} not found under {dataset_dir}")
    return img_path


def _load_nerf_frames(dataset_dir: str, transforms_path: str) -> tuple[list[np.ndarray], list[str], float]:
    with open(transforms_path, encoding="utf-8") as f:
        transforms = json.load(f)
    c2ws: list[np.ndarray] = []
    image_paths: list[str] = []
    for frame in transforms.get("frames", []):
        c2ws.append(np.asarray(frame["transform_matrix"], dtype=np.float32))
        image_paths.append(_resolve_nerf_frame_image(dataset_dir, frame["file_path"]))
    return c2ws, image_paths, float(transforms["camera_angle_x"])


def _load_nerf_dataset_views(
    dataset_dir: str,
) -> tuple[np.ndarray, list[str], float, list[int], list[int]]:
    """
    Load all views for SplatFormer export.

    Prefer view_split.json (canonical train/eval indices). Fall back to union of
    transforms_train.json + transforms_test.json.
    """
    split_path = os.path.join(dataset_dir, "view_split.json")
    train_path = os.path.join(dataset_dir, "transforms_train.json")
    test_path = os.path.join(dataset_dir, "transforms_test.json")

    if os.path.isfile(split_path):
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        train_ids = [int(i) for i in split["train_indices"]]
        eval_ids = [int(i) for i in split["eval_indices"]]
        # Reconstruct ordered full view list from on-disk r_*.png + poses in transforms.
        frames_by_idx: dict[int, dict[str, Any]] = {}
        camera_angle_x = None
        for path in (train_path, test_path, os.path.join(dataset_dir, "transforms_val.json")):
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                transforms = json.load(f)
            if camera_angle_x is None:
                camera_angle_x = float(transforms["camera_angle_x"])
            for frame in transforms.get("frames", []):
                rel = frame["file_path"]
                # Expect ./train/r_<idx>
                stem = os.path.basename(rel)
                if stem.startswith("r_"):
                    idx = int(stem.split("_", 1)[1])
                    frames_by_idx[idx] = frame
        if camera_angle_x is None:
            raise RuntimeError(f"No transforms_*.json under {dataset_dir}")
        all_ids = sorted(frames_by_idx.keys())
        if not all_ids:
            raise RuntimeError(f"No frames found under {dataset_dir}")
        c2ws = [np.asarray(frames_by_idx[i]["transform_matrix"], dtype=np.float32) for i in all_ids]
        image_paths = [_resolve_nerf_frame_image(dataset_dir, frames_by_idx[i]["file_path"]) for i in all_ids]
        # Remap split indices into positions in all_ids if needed (usually identity).
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_ids)}
        train_pos = [idx_to_pos[i] for i in train_ids if i in idx_to_pos]
        eval_pos = [idx_to_pos[i] for i in eval_ids if i in idx_to_pos]
        return np.stack(c2ws, axis=0), image_paths, camera_angle_x, train_pos, eval_pos

    # Legacy: no view_split.json — use train transforms as the full set and recompute split.
    c2ws, image_paths, camera_angle_x = _load_nerf_frames(dataset_dir, train_path)
    n_views = len(image_paths)
    train_ids, eval_ids = _disjoint_train_eval_indices(
        n_views, max_eval_views=8, max_train_views=4
    )
    return np.stack(c2ws, axis=0), image_paths, camera_angle_x, train_ids, eval_ids


def _export_splatformer_ckpt(src_ckpt: str, dst_ckpt: str) -> None:
    """Rewrite a splatfacto ckpt to flat '_model.gauss_params.*' keys for GS.py."""
    from utils.gs_utils import extract_gauss_params_from_nerfstudio_ckpt

    raw = torch.load(src_ckpt, map_location="cpu")
    parsed = extract_gauss_params_from_nerfstudio_ckpt(raw)
    out = {f"_model.gauss_params.{name}": tensor for name, tensor in parsed.items()}
    torch.save(out, dst_ckpt)


def export_splatformer_scene(
    *,
    scene_name: str,
    splatformer_root: str,
    ckpt_path: str,
    dataset_dir: str,
    config_yml: str | None,
    num_test_views: int,
    fx: float | None = None,
    fy: float | None = None,
    overwrite: bool = False,
    hr_ply: str | None = None,
    test_camera_mode: str = "stage2_orbit",
    test_image_size: int = 256,
    render_device: str = "auto",
    render_backend: str = "auto",
    background: str = "white",
    hr_sh_degree: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Write a SplatFormer-compatible scene:
      <root>/nerfstudio/<scene>/splatfacto/{nerfstudio_models,camera_for-3d-denoise.pkl,...}
      <root>/colmap/<scene>/{images,sparse/0}

    ``test_camera_mode``:
      - ``stage2_orbit`` (default): 8 Stage-2-style orbit cameras + HR GT renders
        (matches ``infer_stage2_diffusion`` ~21 PSNR input baseline).
      - ``held_out``: copy held-out training-hemisphere views (higher ns-eval-like PSNR).
    """
    ns_scene_dir = os.path.join(splatformer_root, "nerfstudio", scene_name, "splatfacto")
    colmap_scene_dir = os.path.join(splatformer_root, "colmap", scene_name)
    images_dir = os.path.join(colmap_scene_dir, "images")
    sparse_dir = os.path.join(colmap_scene_dir, "sparse", "0")
    models_dir = os.path.join(ns_scene_dir, "nerfstudio_models")
    pkl_path = os.path.join(ns_scene_dir, "camera_for-3d-denoise.pkl")
    marker = os.path.join(ns_scene_dir, "splatformer_export.json")

    if os.path.isfile(pkl_path) and os.path.isdir(models_dir) and not overwrite:
        # Only skip if train/test image counts match the pickle and camera mode matches.
        try:
            with open(pkl_path, "rb") as f:
                existing_meta = pickle.load(f)
            n_train_cam = len(existing_meta["train_camera_to_worlds"])
            n_test_cam = len(existing_meta["test_camera_to_worlds"])
            img_names = [
                n for n in os.listdir(images_dir) if not n.startswith(".")
            ] if os.path.isdir(images_dir) else []
            n_train_img = sum(1 for n in img_names if n.lower().startswith("train"))
            n_test_img = sum(
                1 for n in img_names if n.lower().startswith("test") or n.lower().startswith("frame_eval")
            )
            existing_mode = None
            if os.path.isfile(marker):
                with open(marker, encoding="utf-8") as f:
                    existing_mode = json.load(f).get("test_camera_mode")
            mode_ok = existing_mode == test_camera_mode or (
                existing_mode is None and test_camera_mode == "held_out"
            )
            if (
                n_train_img == n_train_cam
                and n_test_img == n_test_cam
                and n_test_cam > 0
                and mode_ok
            ):
                print(f"SplatFormer scene exists (use --reexport_splatformer): {ns_scene_dir}")
                return {
                    "scene_name": scene_name,
                    "nerfstudio_dir": os.path.abspath(ns_scene_dir),
                    "colmap_dir": os.path.abspath(colmap_scene_dir),
                    "status": "exists",
                }
            print(
                f"SplatFormer scene incomplete or mode mismatch "
                f"(train imgs/cams={n_train_img}/{n_train_cam}, "
                f"test imgs/cams={n_test_img}/{n_test_cam}, "
                f"mode={existing_mode!r}→{test_camera_mode!r}); re-exporting {scene_name}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"SplatFormer scene exists but failed validation ({exc}); re-exporting")

    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing splatfacto ckpt for SplatFormer export: {ckpt_path}")

    c2ws_gl, image_paths, camera_angle_x, train_ids, test_ids_sorted = _load_nerf_dataset_views(
        dataset_dir
    )
    n_views = len(image_paths)
    if n_views == 0:
        raise RuntimeError(f"No frames in {dataset_dir}")

    # Prefer dataset split; if missing/empty test, fall back to evenly spaced hold-out.
    if not test_ids_sorted:
        test_ids = set(_eval_view_indices(n_views, max_eval_views=max(1, num_test_views)))
        train_ids = [i for i in range(n_views) if i not in test_ids]
        test_ids_sorted = sorted(test_ids)
    if not train_ids:
        # Keep at least one train view for loader compatibility
        train_ids = [test_ids_sorted.pop(0)]
    test_ids_sorted = list(test_ids_sorted)

    # Train intrinsics from training images (may differ from Stage-2 test size).
    with Image.open(image_paths[0]) as im0:
        train_width, train_height = im0.size
    if fx is None or fy is None:
        train_fx = train_fy = 0.5 * train_width / math.tan(0.5 * camera_angle_x)
    else:
        train_fx = float(fx)
        train_fy = float(fy if fy is not None else fx)

    dp = _load_dataparser_transforms(config_yml)
    c2ws_ns = _apply_dataparser_to_c2ws(c2ws_gl, dp["transform"], dp["scale"])

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # Fresh image dir to avoid stale train/test mixes
    for old in glob.glob(os.path.join(images_dir, "*")):
        if os.path.isfile(old):
            os.remove(old)

    ckpt_name = os.path.basename(ckpt_path)
    dst_ckpt = os.path.join(models_dir, ckpt_name)
    # Rewrite to SplatFormer-friendly flat keys: '_model.gauss_params.<name>'
    _export_splatformer_ckpt(ckpt_path, dst_ckpt)

    with open(os.path.join(ns_scene_dir, "dataparser_transforms.json"), "w", encoding="utf-8") as f:
        json.dump(dp, f, indent=2)

    train_c2ws = []
    test_c2ws = []
    image_records: list[tuple[str, np.ndarray]] = []  # (filename, c2w_3x4)

    for local_i, view_i in enumerate(train_ids):
        name = f"train_{local_i:03d}.png"
        shutil.copy2(image_paths[view_i], os.path.join(images_dir, name))
        train_c2ws.append(c2ws_ns[view_i])
        image_records.append((name, c2ws_ns[view_i]))

    if test_camera_mode == "stage2_orbit":
        if not hr_ply or not os.path.isfile(hr_ply):
            raise FileNotFoundError(
                f"stage2_orbit test cameras need --hr_ply; missing: {hr_ply}"
            )
        set_seed(seed)
        resolved_sh = resolve_hr_sh_degree(hr_ply, hr_sh_degree)
        resolved_device, resolved_backend = resolve_render_setup(render_device, render_backend)
        hr_gaussians = load_ply(hr_ply, sh_degree=resolved_sh)
        test_bundle = render_hr_stage2_orbit_views(
            hr_gaussians,
            num_views=max(1, num_test_views),
            image_size=test_image_size,
            focal_length=float(fx if fx is not None else 500.0),
            background=background,
            device=resolved_device,
            backend=resolved_backend,
            hr_sh_degree=resolved_sh,
        )
        test_c2w_cv = test_bundle["camera_to_worlds"].numpy()
        test_c2w_gl = np.stack(
            [_opencv_c2w_to_opengl(test_c2w_cv[i]) for i in range(test_c2w_cv.shape[0])],
            axis=0,
        )
        test_c2w_ns = _apply_dataparser_to_c2ws(test_c2w_gl, dp["transform"], dp["scale"])
        width = int(test_bundle["width"])
        height = int(test_bundle["height"])
        fx = float(test_bundle["fx"])
        fy = float(test_bundle["fy"])
        cx = float(test_bundle["cx"])
        cy = float(test_bundle["cy"])
        for local_i in range(test_c2w_ns.shape[0]):
            name = f"test_{local_i:03d}.png"
            arr = (test_bundle["hr_images"][local_i].clamp(0, 1).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(images_dir, name))
            test_c2ws.append(test_c2w_ns[local_i])
            image_records.append((name, test_c2w_ns[local_i]))
        test_ids_sorted = list(range(len(test_c2ws)))  # synthetic indices for bookkeeping
    elif test_camera_mode == "held_out":
        width, height = train_width, train_height
        fx, fy = float(train_fx), float(train_fy)
        cx = width * 0.5
        cy = height * 0.5
        for local_i, view_i in enumerate(test_ids_sorted):
            name = f"test_{local_i:03d}.png"
            shutil.copy2(image_paths[view_i], os.path.join(images_dir, name))
            test_c2ws.append(c2ws_ns[view_i])
            image_records.append((name, c2ws_ns[view_i]))
    else:
        raise ValueError(
            f"Unknown test_camera_mode={test_camera_mode!r} (use stage2_orbit or held_out)"
        )

    if not test_c2ws:
        raise RuntimeError("Need at least one test view for SplatFormer eval.")

    # SplatFormer uses one set of intrinsics per scene; for stage2_orbit the pickle
    # stores the *test* intrinsics (eval cameras). Train images may differ in size
    # but are only used as conditioning poses during training, not in --only_eval.
    train_image_names = [f"train_{i:03d}.png" for i in range(len(train_c2ws))]
    test_image_names = [f"test_{i:03d}.png" for i in range(len(test_c2ws))]
    meta = {
        "train_camera_to_worlds": torch.from_numpy(np.stack(train_c2ws, axis=0)).float(),
        "test_camera_to_worlds": torch.from_numpy(np.stack(test_c2ws, axis=0)).float(),
        "train_image_names": train_image_names,
        "test_image_names": test_image_names,
        "fx": torch.tensor(fx, dtype=torch.float32),
        "fy": torch.tensor(fy, dtype=torch.float32),
        "cx": torch.tensor(cx, dtype=torch.float32),
        "cy": torch.tensor(cy, dtype=torch.float32),
        "width": torch.tensor(width, dtype=torch.int64),
        "height": torch.tensor(height, dtype=torch.int64),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f)

    # Minimal COLMAP text model (optional; used if load_pose_src='colmap')
    with open(os.path.join(sparse_dir, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 SIMPLE_PINHOLE {width} {height} {fx} {cx} {cy}\n")

    with open(os.path.join(sparse_dir, "images.txt"), "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, (name, c2w) in enumerate(image_records):
            qvec, tvec = _opengl_c2w_to_colmap_w2c(c2w)
            f.write(
                f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} 1 {name}\n\n"
            )

    # Loose AABB from camera centers (SplatFormer OOD ships bbox.txt)
    centers = np.stack([_as_c2w_4x4(c)[:3, 3] for _, c in image_records], axis=0)
    xyz_min = centers.min(axis=0)
    xyz_max = centers.max(axis=0)
    with open(os.path.join(sparse_dir, "bbox.txt"), "w", encoding="utf-8") as f:
        f.write(f"{xyz_min[0]} {xyz_min[1]} {xyz_min[2]}\n")
        f.write(f"{xyz_max[0]} {xyz_max[1]} {xyz_max[2]}\n")

    export_info = {
        "scene_name": scene_name,
        "nerfstudio_dir": os.path.abspath(ns_scene_dir),
        "colmap_dir": os.path.abspath(colmap_scene_dir),
        "ckpt": os.path.abspath(dst_ckpt),
        "camera_pkl": os.path.abspath(pkl_path),
        "num_train_views": len(train_ids),
        "num_test_views": len(test_c2ws),
        "train_indices": train_ids,
        "test_indices": test_ids_sorted if test_camera_mode == "held_out" else [],
        "test_camera_mode": test_camera_mode,
        "test_image_size": width,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "dataparser": dp,
        "status": "ok",
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(export_info, f, indent=2)

    print(
        f"SplatFormer scene → {splatformer_root} "
        f"(train={len(train_ids)}, test={len(test_c2ws)}, mode={test_camera_mode}) "
        f"scene={scene_name}"
    )
    return export_info


def _metric_mean(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("mean", "avg", "average"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


def _parse_eval_json(eval_json_path: str) -> dict[str, float]:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)
    out: dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips"):
        mean = _metric_mean(results.get(key))
        if mean is not None:
            out[key] = mean
    return out


def _prepare_hr_eval_config(train_config_yml: str, work_dir: str, eval_scale: float) -> str:
    eval_config_path = os.path.join(work_dir, "config_eval_hr.yml")
    with open(train_config_yml, encoding="utf-8") as f:
        text = f.read()
    pattern = r"(camera_res_scale_factor:\s*)[\d.]+"
    if not re.search(pattern, text):
        raise RuntimeError(f"camera_res_scale_factor not found in {train_config_yml}")
    new_text = re.sub(pattern, rf"\g<1>{eval_scale}", text, count=1)
    with open(eval_config_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return eval_config_path


def _run_ns_eval(train_config_yml: str, work_dir: str, eval_json_path: str, eval_scale: float) -> dict[str, float]:
    eval_config = _prepare_hr_eval_config(train_config_yml, work_dir, eval_scale)
    os.makedirs(os.path.dirname(eval_json_path) or ".", exist_ok=True)
    eval_cmd = _find_ns_cmd("ns-eval") + [
        "--load-config",
        eval_config,
        "--output-path",
        eval_json_path,
    ]
    _run(eval_cmd)
    metrics = _parse_eval_json(eval_json_path)
    if not metrics:
        raise RuntimeError(f"ns-eval wrote {eval_json_path} but no PSNR/SSIM/LPIPS found.")
    return metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SplatfactoShapeNetConfig:
    # Rendering (matches gaussian_sr train_lr_splats_splatfacto / sample_0005)
    num_views: int = 72
    # 0 / None / >= num_views → train on all hemisphere views (gaussian_sr default).
    # Smaller values keep the sparse-view disjoint-train path.
    num_train_views: int = 0
    hr_image_size: int = 400
    lr_image_size: int = 200
    focal_length: float = 500.0
    background: str = "white"
    hr_sh_degree: int | None = None
    seed: int = 42

    # splatfacto (matches gaussian_sr / run_splatfacto_nerf.py)
    sh_degree: int = 0
    max_num_iterations: int = 20_000
    cull_alpha_thresh: float = 0.15
    background_color: str = "white"
    vis: str = "none"  # sentinel → tensorboard; viewer stalls ns-train on TRUBA GPU nodes
    # splatfacto default is 2 (coarse-to-fine). That is for 1k+ photos; at 100px LR
    # it trains the first 3k steps at 25px and screen-size splitting explodes.
    # 0 → train at camera-res-scale-factor resolution from step 0.
    num_downscales: int = 0
    # None → scale splatfacto's 0.05 so a Gaussian covers the same *pixels* as at 200px.
    split_screen_size: float | None = None
    # None → splatfacto default 15000 when dense; 5000 when sparse (stop densifying
    # before Gaussians fill the 4 camera frustums).
    stop_split_at: int | None = None
    # None → splatfacto default (50000) when dense; 5000 when sparse to avoid
    # filling frustums with unused random points that outnumber the object.
    num_random: int | None = None
    # None → splatfacto default (10.0) when dense; 1.0 when sparse to confine
    # initial Gaussians near the origin where the object actually lives.
    random_scale: float | None = None

    # SplatFormer export
    splatformer_root: str = "test-set/customOOD"
    num_test_views: int = 8
    skip_splatformer_export: bool = False
    # stage2_orbit → 8-view 256px orbit GT (matches infer_stage2 ~21 PSNR baseline)
    # held_out → training-hemisphere holdouts (higher / ns-eval-like PSNR)
    test_camera_mode: str = "stage2_orbit"
    test_image_size: int = 256
    reexport_splatformer: bool = False

    @property
    def camera_res_scale_factor(self) -> float:
        return self.lr_image_size / float(self.hr_image_size)

    @property
    def effective_split_screen_size(self) -> float:
        if self.split_screen_size is not None:
            return float(self.split_screen_size)
        # splatfacto default 0.05 is a *fraction* of the image. At 100px that is
        # 5px vs 10px at the working 200px (2×) setting, so 4× over-splits.
        return 0.05 * (200.0 / float(self.lr_image_size))

    @property
    def train_all_views(self) -> bool:
        """True when splatfacto should fit on the full hemisphere (gaussian_sr)."""
        return self.num_train_views is None or int(self.num_train_views) <= 0 or int(
            self.num_train_views
        ) >= int(self.num_views)

    @property
    def effective_num_train_views(self) -> int:
        return int(self.num_views) if self.train_all_views else int(self.num_train_views)

    @property
    def effective_max_num_iterations(self) -> int:
        """Sparse 4-view fits explode if densification runs for the full 20k dense schedule."""
        if self.train_all_views:
            return int(self.max_num_iterations)
        if int(self.max_num_iterations) == 20_000:
            return 10_000
        return int(self.max_num_iterations)

    @property
    def effective_stop_split_at(self) -> int | None:
        if self.stop_split_at is not None:
            return int(self.stop_split_at)
        if self.train_all_views:
            return None
        return 5_000

    @property
    def effective_num_random(self) -> int:
        if self.num_random is not None:
            return int(self.num_random)
        if self.train_all_views:
            return 50_000
        return 5_000

    @property
    def effective_random_scale(self) -> float:
        if self.random_scale is not None:
            return float(self.random_scale)
        if self.train_all_views:
            return 10.0
        return 1.0

    @property
    def effective_background_color(self) -> str:
        """Random BG for sparse views prevents the model from hiding white splats."""
        if self.train_all_views:
            return self.background_color
        return "random"


# ---------------------------------------------------------------------------
# NeRF Synthetic dataset export
# ---------------------------------------------------------------------------


def _opencv_c2w_to_opengl(c2w_cv: np.ndarray) -> np.ndarray:
    """Convert train_lr_splats OpenCV c2w (+Z forward) to Blender/OpenGL (+Z backward)."""
    c2w_gl = c2w_cv.astype(np.float64).copy()
    c2w_gl[:3, 1] *= -1.0
    c2w_gl[:3, 2] *= -1.0
    return c2w_gl.astype(np.float32)


def _evenly_spaced_indices(num_views: int, k: int) -> list[int]:
    """Pick up to ``k`` evenly spaced indices in ``[0, num_views)``."""
    if num_views <= 0 or k <= 0:
        return []
    if k >= num_views:
        return list(range(num_views))
    if num_views == 1:
        return [0]
    return [int(round(i * (num_views - 1) / (k - 1))) for i in range(k)]


# Sparse-view layout: 1 near-zenith camera + (k-1) sides at ~30 deg elevation,
# spaced evenly in azimuth. Four co-elevation 45 deg cameras leave depth along
# each ray unconstrained, so splatfacto fills the 4 frustums instead of the object.
# Evenly-spaced Fibonacci *indices* are worse still: they bunch into a ~77 deg
# azimuth wedge (see _hemisphere_unit_directions).
_SPARSE_VIEW_LAYOUT = "top_plus_sides"
_SPARSE_TOP_ELEV_DEG = 80.0
_SPARSE_SIDE_ELEV_DEG = 30.0


def _elev_az_unit(elev_deg: float, az_rad: float) -> tuple[float, float, float]:
    el = math.radians(elev_deg)
    z = math.sin(el)
    r = math.cos(el)
    return (r * math.cos(az_rad), r * math.sin(az_rad), z)


def _nearest_unused_direction(
    directions: list[tuple[float, float, float]],
    target: tuple[float, float, float],
    selected: list[int],
) -> int:
    tx, ty, tz = target
    selected_set = set(selected)
    best_dist = float("inf")
    best_idx = 0
    for i, (dx, dy, dz) in enumerate(directions):
        if i in selected_set:
            continue
        dist = (dx - tx) ** 2 + (dy - ty) ** 2 + (dz - tz) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return int(best_idx)


def _spread_view_indices(num_views: int, k: int) -> list[int]:
    """Pick ``k`` views: 1 near the top, the rest around the object at ~30 deg.

    Greedily assigns the nearest unused Fibonacci-spiral direction to each target.
    For k=4 this is 1 top + 3 sides at 120 deg azimuth (not 4 cameras on a 45 deg ring).
    """
    if num_views <= 0 or k <= 0:
        return []
    if k >= num_views:
        return list(range(num_views))
    directions = _hemisphere_unit_directions(num_views)
    selected: list[int] = []
    selected.append(
        _nearest_unused_direction(directions, _elev_az_unit(_SPARSE_TOP_ELEV_DEG, 0.0), selected)
    )
    n_side = k - 1
    for a in range(n_side):
        az = 2.0 * math.pi * a / n_side
        selected.append(
            _nearest_unused_direction(
                directions, _elev_az_unit(_SPARSE_SIDE_ELEV_DEG, az), selected
            )
        )
    return sorted(selected)


def _eval_view_indices(num_views: int, max_eval_views: int = 8) -> list[int]:
    """Pick held-out views for nerfstudio val/test (matches gaussian_sr step sampling)."""
    if num_views <= 0:
        return []
    if num_views == 1:
        return [0]
    step = max(1, num_views // max_eval_views)
    return list(range(0, num_views, step))[:max_eval_views]


def _disjoint_train_eval_indices(
    num_views: int, max_eval_views: int = 8, max_train_views: int = 4
) -> tuple[list[int], list[int]]:
    """
    Return (train_indices, eval_indices) with no overlap.

    Train gets at most ``max_train_views`` views (1 near-zenith + the rest at
    ~30 deg elevation, even azimuth); eval is sampled from the remainder.
    """
    train_ids = _spread_view_indices(num_views, max_train_views)
    train_set = set(train_ids)
    remaining = [i for i in range(num_views) if i not in train_set]
    if not remaining:
        # Degenerate: all views used for train; keep ≥1 train, rest for eval if possible.
        if len(train_ids) > 1 and max_eval_views > 0:
            eval_ids = train_ids[1 : 1 + max_eval_views]
            train_ids = train_ids[:1]
            return train_ids, eval_ids
        return train_ids, []
    eval_local = _evenly_spaced_indices(len(remaining), max_eval_views)
    eval_ids = [remaining[i] for i in eval_local]
    return train_ids, eval_ids


def _write_transforms_json(
    path: str, camera_angle_x: float, frames: list[dict[str, Any]]
) -> None:
    payload = {"camera_angle_x": camera_angle_x, "frames": frames}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _write_disjoint_nerf_splits(
    dataset_dir: str,
    all_frames: list[dict[str, Any]],
    camera_angle_x: float,
    max_eval_views: int = 8,
    max_train_views: int = 4,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Write transforms_{train,val,test}.json with disjoint train vs eval frames.

    Images stay on disk (e.g. ./train/r_i.png); only the transform lists change.
    Also writes view_split.json as the canonical index split.
    """
    n_views = len(all_frames)
    if n_views == 0:
        raise RuntimeError(f"No frames to split under {dataset_dir}")

    train_ids, eval_ids = _disjoint_train_eval_indices(
        n_views, max_eval_views=max_eval_views, max_train_views=max_train_views
    )
    train_frames = [all_frames[i] for i in train_ids]
    eval_frames = [all_frames[i] for i in eval_ids]

    train_path = os.path.join(dataset_dir, "transforms_train.json")
    val_path = os.path.join(dataset_dir, "transforms_val.json")
    test_path = os.path.join(dataset_dir, "transforms_test.json")
    split_path = os.path.join(dataset_dir, "view_split.json")

    if (
        not overwrite
        and os.path.isfile(split_path)
        and os.path.isfile(train_path)
        and os.path.isfile(val_path)
        and os.path.isfile(test_path)
    ):
        with open(split_path, encoding="utf-8") as f:
            existing = json.load(f)
        if (
            existing.get("disjoint")
            and set(existing.get("train_indices", [])).isdisjoint(
                set(existing.get("eval_indices", []))
            )
            and len(existing.get("train_indices", [])) == max_train_views
            and existing.get("view_layout") == _SPARSE_VIEW_LAYOUT
        ):
            return existing

    _write_transforms_json(train_path, camera_angle_x, train_frames)
    _write_transforms_json(val_path, camera_angle_x, eval_frames)
    _write_transforms_json(test_path, camera_angle_x, eval_frames)

    split = {
        "num_views": n_views,
        "train_indices": train_ids,
        "eval_indices": eval_ids,
        "disjoint": True,
        "max_train_views": max_train_views,
        "max_eval_views": max_eval_views,
        "view_layout": _SPARSE_VIEW_LAYOUT,
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(
        f"Wrote disjoint NeRF splits → {dataset_dir} "
        f"(train={len(train_ids)}, eval/test={len(eval_ids)})"
    )
    return split


def _load_all_frames_for_resplit(dataset_dir: str) -> tuple[list[dict[str, Any]], float] | None:
    """
    Best-effort reconstruct the full frame list for repairing old overlapping splits.

    Prefer union of train+val+test keyed by r_<idx>. If only train exists and looks
    complete, use that.
    """
    frames_by_idx: dict[int, dict[str, Any]] = {}
    camera_angle_x = None
    for name in ("transforms_train.json", "transforms_val.json", "transforms_test.json"):
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            transforms = json.load(f)
        if camera_angle_x is None:
            camera_angle_x = float(transforms["camera_angle_x"])
        for frame in transforms.get("frames", []):
            stem = os.path.basename(frame["file_path"])
            if not stem.startswith("r_"):
                continue
            idx = int(stem.split("_", 1)[1])
            frames_by_idx[idx] = frame

    if not frames_by_idx or camera_angle_x is None:
        return None
    all_frames = [frames_by_idx[i] for i in sorted(frames_by_idx.keys())]
    return all_frames, camera_angle_x


def _ensure_nerf_eval_transforms(
    dataset_dir: str,
    max_eval_views: int = 8,
    max_train_views: int = 4,
    *,
    force_disjoint: bool = True,
) -> None:
    """
    Ensure transforms_val/test exist.

    When force_disjoint=True and max_train_views < num views, rewrite so held-out
    eval views are excluded from train (sparse-view experiments).

    When force_disjoint=False (gaussian_sr / train-all-views), keep all frames in
    transforms_train and write overlapping val/test subsets if missing.
    """
    split_path = os.path.join(dataset_dir, "view_split.json")
    train_path = os.path.join(dataset_dir, "transforms_train.json")
    if not os.path.isfile(train_path):
        return

    if not force_disjoint:
        # gaussian_sr-compatible: all views remain in train; val/test are subsets.
        val_path = os.path.join(dataset_dir, "transforms_val.json")
        test_path = os.path.join(dataset_dir, "transforms_test.json")
        if os.path.isfile(val_path) and os.path.isfile(test_path):
            return
        with open(train_path, encoding="utf-8") as f:
            train = json.load(f)
        camera_angle_x = float(train["camera_angle_x"])
        train_frames: list[dict[str, Any]] = train.get("frames", [])
        if not train_frames:
            raise RuntimeError(f"No frames found in {train_path}")
        indices = _eval_view_indices(len(train_frames), max_eval_views=max_eval_views)
        eval_frames = [train_frames[i] for i in indices]
        for split_name, path in (("val", val_path), ("test", test_path)):
            if not os.path.isfile(path):
                _write_transforms_json(path, camera_angle_x, eval_frames)
                print(f"Wrote {split_name} split ({len(eval_frames)} views) → {path}")
        return

    if os.path.isfile(split_path):
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        if (
            split.get("disjoint")
            and set(split.get("train_indices", [])).isdisjoint(set(split.get("eval_indices", [])))
            and len(split.get("train_indices", [])) == max_train_views
            and split.get("view_layout") == _SPARSE_VIEW_LAYOUT
            and os.path.isfile(os.path.join(dataset_dir, "transforms_val.json"))
            and os.path.isfile(os.path.join(dataset_dir, "transforms_test.json"))
        ):
            return

    loaded = _load_all_frames_for_resplit(dataset_dir)
    if loaded is None:
        raise RuntimeError(f"Cannot build disjoint splits under {dataset_dir}")
    all_frames, camera_angle_x = loaded
    _write_disjoint_nerf_splits(
        dataset_dir,
        all_frames,
        camera_angle_x,
        max_eval_views=max_eval_views,
        max_train_views=max_train_views,
        overwrite=True,
    )


def export_nerf_synthetic_dataset(
    bundle: dict[str, Any],
    cfg: SplatfactoShapeNetConfig,
    dataset_dir: str,
) -> str:
    """Write images + transforms for nerfstudio blender-data.

    Default (``train_all_views``): gaussian_sr-compatible — all views in
    ``transforms_train.json``, overlapping val/test subsets.

    Sparse (``num_train_views`` < ``num_views``): disjoint train/eval splits.
    """
    train_dir = os.path.join(dataset_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    hr_images = bundle["hr_images"]
    c2ws = bundle["camera_to_worlds"]
    fx_hr = float(bundle["fx_hr"])
    width = cfg.hr_image_size
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx_hr))

    frames: list[dict[str, Any]] = []
    for view_idx in tqdm(range(hr_images.shape[0]), desc="Writing NeRF dataset"):
        rel_path = f"./train/r_{view_idx}"
        img_path = os.path.join(dataset_dir, f"train/r_{view_idx}.png")
        arr = (hr_images[view_idx].clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_path)

        c2w_gl = _opencv_c2w_to_opengl(c2ws[view_idx].numpy())
        frames.append(
            {
                "file_path": rel_path,
                "transform_matrix": c2w_gl.tolist(),
            }
        )

    if cfg.train_all_views:
        _write_transforms_json(
            os.path.join(dataset_dir, "transforms_train.json"), camera_angle_x, frames
        )
        eval_ids = _eval_view_indices(len(frames), max_eval_views=cfg.num_test_views)
        eval_frames = [frames[i] for i in eval_ids]
        _write_transforms_json(
            os.path.join(dataset_dir, "transforms_val.json"), camera_angle_x, eval_frames
        )
        _write_transforms_json(
            os.path.join(dataset_dir, "transforms_test.json"), camera_angle_x, eval_frames
        )
        split = {
            "num_views": len(frames),
            "train_indices": list(range(len(frames))),
            "eval_indices": eval_ids,
            "disjoint": False,
            "max_train_views": len(frames),
            "max_eval_views": cfg.num_test_views,
        }
        with open(os.path.join(dataset_dir, "view_split.json"), "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(
            f"Wrote gaussian_sr-style NeRF dataset → {dataset_dir} "
            f"(train={len(frames)}, eval/test={len(eval_ids)})"
        )
    else:
        split = _write_disjoint_nerf_splits(
            dataset_dir,
            frames,
            camera_angle_x,
            max_eval_views=cfg.num_test_views,
            max_train_views=cfg.effective_num_train_views,
            overwrite=True,
        )

    meta = {
        "dataset_format": "nerf_synthetic",
        "num_views": len(frames),
        "hr_image_size": cfg.hr_image_size,
        "lr_image_size": cfg.lr_image_size,
        "camera_res_scale_factor": cfg.camera_res_scale_factor,
        "camera_angle_x": camera_angle_x,
        "background": cfg.background,
        "hr_sh_degree": bundle.get("hr_sh_degree"),
        "train_indices": split["train_indices"],
        "eval_indices": split["eval_indices"],
        "num_train_views": cfg.effective_num_train_views,
        "disjoint_splits": bool(split.get("disjoint", False)),
        "train_all_views": cfg.train_all_views,
        "view_layout": split.get("view_layout"),
    }
    with open(os.path.join(dataset_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return os.path.join(dataset_dir, "transforms_train.json")


def _materialize_nerf_dataset(src_dataset_dir: str, dst_dataset_dir: str) -> None:
    """Link or copy an existing gaussian_sr nerf_dataset into this sample's work dir."""
    src_dataset_dir = os.path.abspath(src_dataset_dir)
    dst_dataset_dir = os.path.abspath(dst_dataset_dir)
    if src_dataset_dir == dst_dataset_dir:
        return
    if os.path.islink(dst_dataset_dir):
        os.unlink(dst_dataset_dir)
    elif os.path.isdir(dst_dataset_dir):
        # Prefer reusing in place when already populated.
        if os.path.isfile(os.path.join(dst_dataset_dir, "transforms_train.json")):
            return
        shutil.rmtree(dst_dataset_dir)
    parent = os.path.dirname(dst_dataset_dir)
    os.makedirs(parent, exist_ok=True)
    try:
        os.symlink(src_dataset_dir, dst_dataset_dir, target_is_directory=True)
        print(f"Reusing rendered NeRF dataset (symlink): {src_dataset_dir}")
    except OSError:
        shutil.copytree(src_dataset_dir, dst_dataset_dir)
        print(f"Reusing rendered NeRF dataset (copy): {src_dataset_dir}")


def _render_hr_bundle(
    hr_ply: str,
    cfg: SplatfactoShapeNetConfig,
    render_device: str,
    render_backend: str,
) -> tuple[dict[str, Any], int]:
    hr_sh_degree = resolve_hr_sh_degree(hr_ply, cfg.hr_sh_degree)
    hr_gaussians = load_ply(hr_ply, sh_degree=hr_sh_degree)
    resolved_device, resolved_backend = resolve_render_setup(render_device, render_backend)
    print(
        f"Rendering {cfg.num_views} HR views at {cfg.hr_image_size}px "
        f"(hr_sh_degree={hr_sh_degree}) via {resolved_backend} on {resolved_device}"
    )
    bundle = render_hr_lr_targets(
        hr_gaussians,
        cfg,
        resolved_device,
        resolved_backend,
        hr_sh_degree,
    )
    bundle["hr_sh_degree"] = hr_sh_degree
    bundle["hr_num_gaussians"] = int(hr_gaussians.shape[0])
    return bundle, hr_sh_degree


def _format_metrics_record(
    cfg: SplatfactoShapeNetConfig,
    eval_metrics: dict[str, float],
    ply_path: str,
    hr_sh_degree: int,
) -> dict[str, str]:
    gaussian_nums = ""
    if os.path.isfile(ply_path):
        try:
            from plyfile import PlyData

            gaussian_nums = str(len(PlyData.read(ply_path)["vertex"].data))
        except Exception:  # noqa: BLE001
            gaussian_nums = ""
    file_size_mb = os.path.getsize(ply_path) / (1024 * 1024) if os.path.isfile(ply_path) else 0.0
    return {
        "iteration": str(cfg.effective_max_num_iterations),
        "l1_loss": "",
        "psnr": str(eval_metrics.get("psnr", "")),
        "ssim": str(eval_metrics.get("ssim", "")),
        "lpips": str(eval_metrics.get("lpips", "")),
        "file_size": str(file_size_mb),
        "gaussian_nums": gaussian_nums,
        "hr_sh_degree": str(hr_sh_degree),
        "method": "splatfacto",
    }


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------


def _resolve_ns_vis(vis: str | None) -> str:
    """Map our ``none`` sentinel to a valid ns-train ``--vis`` choice.

    nerfstudio's ns-train has no ``none`` option (choices: viewer, wandb,
    tensorboard, comet, ...). Use ``tensorboard`` (file logging only, no
    viewer/websocket server) to disable interactive vis without the viewer
    stall seen on TRUBA GPU nodes.
    """
    return "tensorboard" if vis in (None, "", "none") else vis


def process_sample(
    hr_ply: str,
    work_dir: str,
    final_ply_path: str,
    cfg: SplatfactoShapeNetConfig,
    *,
    skip_render: bool,
    skip_train: bool,
    skip_eval: bool,
    overwrite: bool,
    render_device: str,
    render_backend: str,
    reuse_render_roots: list[str] | None = None,
    category_id: str | None = None,
) -> dict[str, Any]:
    """Render HR targets, export NeRF dataset, train splatfacto, export PLY + SplatFormer scene."""
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_ply_path) or ".", exist_ok=True)

    dataset_dir = os.path.join(work_dir, "nerf_dataset")
    ns_out_dir = os.path.join(work_dir, "ns_outputs")
    export_dir = os.path.join(work_dir, "export")
    transforms_path = os.path.join(dataset_dir, "transforms_train.json")
    eval_json_path = os.path.join(work_dir, "eval_metrics.json")
    sample_stem = os.path.splitext(os.path.basename(hr_ply))[0]
    if category_id is None:
        # ``02691156-xxx.ply`` → ``02691156``
        category_id = sample_stem.split("-", 1)[0] if "-" in sample_stem else None

    def _maybe_export_splatformer(config_yml: str | None) -> dict[str, Any] | None:
        if cfg.skip_splatformer_export:
            return None
        if not os.path.isfile(transforms_path):
            print("WARNING: skip SplatFormer export (missing nerf_dataset/transforms_train.json)")
            return None
        ckpt_path = _find_latest_ckpt(config_yml, ns_out_dir)
        if ckpt_path is None:
            print("WARNING: skip SplatFormer export (no nerfstudio_models/step-*.ckpt found)")
            return None
        test_mode = cfg.test_camera_mode
        if test_mode == "stage2_orbit" and not os.path.isfile(hr_ply):
            print(
                "WARNING: HR PLY missing; falling back to held_out test cameras for "
                f"SplatFormer export ({hr_ply})"
            )
            test_mode = "held_out"
        return export_splatformer_scene(
            scene_name=sample_stem,
            splatformer_root=cfg.splatformer_root,
            ckpt_path=ckpt_path,
            dataset_dir=dataset_dir,
            config_yml=config_yml,
            num_test_views=cfg.num_test_views,
            fx=cfg.focal_length,
            fy=cfg.focal_length,
            overwrite=overwrite or cfg.reexport_splatformer,
            hr_ply=hr_ply if os.path.isfile(hr_ply) else None,
            test_camera_mode=test_mode,
            test_image_size=cfg.test_image_size,
            render_device=render_device,
            render_backend=render_backend,
            background=cfg.background,
            hr_sh_degree=cfg.hr_sh_degree,
            seed=cfg.seed,
        )

    if os.path.exists(final_ply_path) and not overwrite and not skip_train:
        print(f"Already trained (use --overwrite): {final_ply_path}")
        config_yml = _find_latest_config(ns_out_dir)
        sf_info = _maybe_export_splatformer(config_yml)
        record = {
            "hr_ply": os.path.abspath(hr_ply),
            "lr_ply": os.path.abspath(final_ply_path),
            "output_dir": os.path.abspath(work_dir),
            "status": "ok",
        }
        if sf_info is not None:
            record["splatformer"] = sf_info
        return record

    set_seed(cfg.seed)
    hr_sh_degree = 0
    if os.path.isfile(hr_ply):
        hr_sh_degree = resolve_hr_sh_degree(hr_ply, cfg.hr_sh_degree)
    elif cfg.hr_sh_degree is not None:
        hr_sh_degree = int(cfg.hr_sh_degree)

    # --- 1. Render HR splat + export NeRF Synthetic dataset ---
    reused = False
    if os.path.isfile(transforms_path) and skip_render:
        print(f"Reusing NeRF dataset: {dataset_dir}")
        meta_path = os.path.join(dataset_dir, "dataset_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            hr_sh_degree = int(meta.get("hr_sh_degree", hr_sh_degree))
        reused = True
    else:
        existing = None
        if category_id and reuse_render_roots:
            existing = find_existing_nerf_dataset(
                sample_stem, category_id, reuse_render_roots
            )
        if existing is not None and not overwrite:
            _materialize_nerf_dataset(existing, dataset_dir)
            meta_path = os.path.join(dataset_dir, "dataset_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                hr_sh_degree = int(meta.get("hr_sh_degree", hr_sh_degree))
            reused = True
        elif os.path.isfile(transforms_path) and not overwrite:
            print(f"Reusing existing NeRF dataset: {dataset_dir}")
            meta_path = os.path.join(dataset_dir, "dataset_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                hr_sh_degree = int(meta.get("hr_sh_degree", hr_sh_degree))
            reused = True
        elif skip_render:
            raise FileNotFoundError(
                f"--skip_render set but no transforms_train.json at {transforms_path} "
                f"and no reusable dataset under {reuse_render_roots}"
            )
        else:
            if not os.path.isfile(hr_ply):
                raise FileNotFoundError(
                    f"HR PLY not found for rendering: {hr_ply}. "
                    "Provide --reuse_render_root with an existing nerf_dataset, or fix --root_path."
                )
            bundle, hr_sh_degree = _render_hr_bundle(hr_ply, cfg, render_device, render_backend)
            export_nerf_synthetic_dataset(bundle, cfg, dataset_dir)
            print(f"NeRF Synthetic dataset → {transforms_path}")

    if skip_train:
        return {
            "hr_ply": os.path.abspath(hr_ply),
            "lr_ply": None,
            "output_dir": os.path.abspath(work_dir),
            "nerf_dataset": os.path.abspath(dataset_dir),
            "transforms_train_json": os.path.abspath(transforms_path),
            "status": "render_only",
            "reused_render": reused,
        }

    # Prepare val/test transforms. Default: keep all 72 train views (gaussian_sr).
    _ensure_nerf_eval_transforms(
        dataset_dir,
        max_eval_views=cfg.num_test_views,
        max_train_views=cfg.effective_num_train_views,
        force_disjoint=not cfg.train_all_views,
    )

    os.makedirs(ns_out_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    config_yml = _find_latest_config(ns_out_dir)

    # --- 2. ns-train splatfacto ---
    if config_yml and os.path.exists(final_ply_path) and not overwrite:
        print(f"Reusing ns-train run: {config_yml}")
        print(
            "NOTE: If this model was trained before disjoint train/eval splits, "
            "ns-eval PSNR may still be on seen views. Re-run with --overwrite to retrain."
        )
    else:
        print(
            f"=== ns-train splatfacto "
            f"(train_views={cfg.effective_num_train_views}/{cfg.num_views}, "
            f"iters={cfg.effective_max_num_iterations}, "
            f"stop_split_at={cfg.effective_stop_split_at}, "
            f"num_random={cfg.effective_num_random}, "
            f"random_scale={cfg.effective_random_scale}, "
            f"bg_color={cfg.effective_background_color}, "
            f"scale={cfg.camera_res_scale_factor}, "
            f"num_downscales={cfg.num_downscales}, "
            f"split_screen_size={cfg.effective_split_screen_size:.4f}) ==="
        )
        train_cmd = _find_ns_cmd("ns-train") + [
            "splatfacto",
            "--output-dir",
            ns_out_dir,
            "--experiment-name",
            sample_stem,
            "--vis",
            _resolve_ns_vis(cfg.vis),
            "--max-num-iterations",
            str(cfg.effective_max_num_iterations),
            # Skip in-training eval; it freezes the progress table (~15%) and
            # makes ETA jump to many hours. Final quality is from ns-eval / SplatFormer.
            "--steps-per-eval-image",
            "100000",
            "--steps-per-eval-batch",
            "100000",
            "--steps-per-eval-all-images",
            "100000",
            "--pipeline.model.sh-degree",
            str(cfg.sh_degree),
            "--pipeline.model.cull-alpha-thresh",
            str(cfg.cull_alpha_thresh),
            "--pipeline.model.background-color",
            cfg.effective_background_color,
            "--pipeline.model.num-downscales",
            str(cfg.num_downscales),
            "--pipeline.model.split-screen-size",
            str(cfg.effective_split_screen_size),
            "--pipeline.model.num-random",
            str(cfg.effective_num_random),
            "--pipeline.model.random-scale",
            str(cfg.effective_random_scale),
        ]
        if cfg.effective_stop_split_at is not None:
            train_cmd += [
                "--pipeline.model.stop-split-at",
                str(cfg.effective_stop_split_at),
            ]
        train_cmd += [
            "--pipeline.datamanager.camera-res-scale-factor",
            str(cfg.camera_res_scale_factor),
            "blender-data",
            "--data",
            dataset_dir,
        ]
        _run(train_cmd)
        config_yml = _find_latest_config(ns_out_dir)
        if config_yml is None:
            raise RuntimeError(f"ns-train finished but no config.yml found in {ns_out_dir}")

    # --- 3. ns-export gaussian-splat ---
    print("=== ns-export gaussian-splat ===")
    if config_yml is None:
        config_yml = _find_latest_config(ns_out_dir)
    if config_yml is None:
        raise RuntimeError("No config.yml found; cannot export.")

    export_cmd = _find_ns_cmd("ns-export") + [
        "gaussian-splat",
        "--load-config",
        config_yml,
        "--output-dir",
        export_dir,
    ]
    _run(export_cmd)

    exported_ply = _find_latest_ply(export_dir)
    if exported_ply is None:
        raise RuntimeError(f"ns-export finished but no PLY found in {export_dir}")

    shutil.copy2(exported_ply, final_ply_path)
    print(f"LR PLY saved → {final_ply_path}")

    # --- 4. ns-eval at HR resolution (stored images are HR px) ---
    eval_metrics: dict[str, float] = {}
    sample_metrics: dict[str, str] | None = None
    if not skip_eval:
        print(f"=== ns-eval at {cfg.hr_image_size}×{cfg.hr_image_size} ===")
        try:
            eval_metrics = _run_ns_eval(config_yml, work_dir, eval_json_path, eval_scale=1.0)
            sample_metrics = _format_metrics_record(cfg, eval_metrics, final_ply_path, hr_sh_degree)
            print(
                f"Eval: PSNR={sample_metrics['psnr']}  "
                f"SSIM={sample_metrics['ssim']}  LPIPS={sample_metrics['lpips']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: eval failed ({exc})")

    # --- 5. SplatFormer-compatible scene export ---
    print("=== SplatFormer scene export ===")
    sf_info = _maybe_export_splatformer(config_yml)

    record: dict[str, Any] = {
        "hr_ply": os.path.abspath(hr_ply),
        "lr_ply": os.path.abspath(final_ply_path),
        "output_dir": os.path.abspath(work_dir),
        "nerf_dataset": os.path.abspath(dataset_dir),
        "transforms_train_json": os.path.abspath(transforms_path),
        "ns_output_dir": os.path.abspath(ns_out_dir),
        "config_yml": os.path.abspath(config_yml) if config_yml else None,
        "hr_sh_degree": hr_sh_degree,
        "lr_image_size": cfg.lr_image_size,
        "hr_image_size": cfg.hr_image_size,
        "camera_res_scale_factor": cfg.camera_res_scale_factor,
        "splatformer_root": os.path.abspath(cfg.splatformer_root),
        "reused_render": reused,
        "train_all_views": cfg.train_all_views,
        "status": "ok",
        "config": asdict(cfg),
    }
    if eval_metrics:
        record["eval_metrics"] = eval_metrics
    if sample_metrics is not None:
        record["sample_metrics"] = sample_metrics
    if sf_info is not None:
        record["splatformer"] = sf_info

    with open(os.path.join(work_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LR splats via splatfacto (gaussian_sr-compatible 72-view config) "
            "and export SplatFormer scenes. Category mode uses the gaussian_sr test split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hr_ply", type=str, default=None, help="Single HR PLY path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Work dir for single-sample mode.")
    parser.add_argument(
        "--final_ply",
        type=str,
        default=None,
        help="Output PLY path for single-sample mode (default: output_dir/lr_splat_final.ply).",
    )

    parser.add_argument(
        "--category_id",
        type=str,
        default=None,
        help="ShapeNet category id for batch mode (uses --split of --dataset_cache).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process. Omit with --category_id to process the whole split.",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/arf/scratch/batubal/data",
        help="HR PLY root (expects <root>/<category_id>/<category_id>-*.ply).",
    )
    parser.add_argument("--output_root", type=str, default="lr_data")
    _default_gs_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "gaussian_sr")
    )
    parser.add_argument(
        "--dataset_cache",
        type=str,
        default=os.path.join(_default_gs_root, "meta.pt"),
        help="gaussian_sr dataset metadata (.pt or *_store/meta.pt) for stratified splits.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which gaussian_sr stratified split to train when --category_id is set.",
    )
    parser.add_argument(
        "--reuse_render_root",
        type=str,
        action="append",
        default=None,
        help=(
            "Root(s) containing <category>/.work/<stem>/nerf_dataset to reuse "
            "(e.g. ../gaussian_sr/lr_data). May be passed multiple times."
        ),
    )

    parser.add_argument("--num_views", type=int, default=72)
    parser.add_argument(
        "--num_train_views",
        type=int,
        default=0,
        help=(
            "Views used for splatfacto training. 0 (default) = all --num_views "
            "(gaussian_sr). Set e.g. 4 for sparse-view disjoint training "
            "(1 near-top + 3 sides at 30 deg / 120 deg azimuth)."
        ),
    )
    parser.add_argument("--hr_image_size", type=int, default=400)
    parser.add_argument("--lr_image_size", type=int, default=200)
    parser.add_argument("--focal_length", type=float, default=500.0)
    parser.add_argument("--background", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--hr_sh_degree", type=int, default=None)

    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--max_num_iterations", type=int, default=20_000)
    parser.add_argument(
        "--stop_split_at",
        type=int,
        default=None,
        help=(
            "splatfacto step to stop densify/split. Default: omit (15000) when training "
            "all views; 5000 when --num_train_views is sparse."
        ),
    )
    parser.add_argument(
        "--num_random",
        type=int,
        default=None,
        help=(
            "Number of random initial Gaussians. Default: 50000 (dense); "
            "5000 (sparse) to avoid filling frustums."
        ),
    )
    parser.add_argument(
        "--random_scale",
        type=float,
        default=None,
        help=(
            "Spatial extent of random init box (half-side). Default: 10.0 (dense); "
            "1.0 (sparse) to confine points near the object."
        ),
    )
    parser.add_argument("--cull_alpha_thresh", type=float, default=0.15)
    parser.add_argument(
        "--num_downscales",
        type=int,
        default=0,
        help=(
            "splatfacto coarse-to-fine extra downscales. Default 0: train at "
            "camera-res-scale-factor from step 0. splatfacto's own default (2) "
            "makes 100px 4× start at 25px and explode Gaussian count."
        ),
    )
    parser.add_argument(
        "--split_screen_size",
        type=float,
        default=None,
        help=(
            "splatfacto screen-size split threshold (image fraction). "
            "Default: 0.05 scaled to 200px so 4× uses 0.10 instead of 0.05."
        ),
    )
    parser.add_argument("--background_color", type=str, default="white", choices=["white", "black", "random"])
    parser.add_argument("--vis", type=str, default="none", choices=["tensorboard", "wandb", "viewer", "none"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--splatformer_root",
        type=str,
        default="test-set/customOOD",
        help="Root for SplatFormer-format export (nerfstudio/ + colmap/).",
    )
    parser.add_argument(
        "--num_test_views",
        type=int,
        default=8,
        help="Number of test_XXX.png views for SplatFormer eval (default 8).",
    )
    parser.add_argument(
        "--test_camera_mode",
        type=str,
        default="stage2_orbit",
        choices=["stage2_orbit", "held_out"],
        help="stage2_orbit: 8-view 256px orbit GT (~21 PSNR input baseline). "
        "held_out: training-hemisphere holdouts (higher PSNR).",
    )
    parser.add_argument(
        "--test_image_size",
        type=int,
        default=256,
        help="Resolution for stage2_orbit test GT renders (match infer_stage2).",
    )
    parser.add_argument(
        "--reexport_splatformer",
        action="store_true",
        help="Rewrite SplatFormer scene (test cams/images) without retraining LR.",
    )
    parser.add_argument(
        "--skip_splatformer_export",
        action="store_true",
        help="Do not write the SplatFormer nerfstudio/colmap scene layout.",
    )

    parser.add_argument("--render_device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--render_backend", type=str, default="auto", choices=["auto", "gsplat", "torch"])
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_train", action="store_true", help="Only build transforms_train.json + images.")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fill_missing",
        action="store_true",
        help="Category mode: train only HR samples whose LR PLY is missing under "
        "output_root/<category_id>/. offset/num_samples slice the missing list; "
        "if num_samples is omitted, all missing samples from offset are processed.",
    )
    parser.add_argument(
        "--rerun_wrong_iterations",
        action="store_true",
        help="Category mode: rerun samples whose .work/<stem>/manifest.json records "
        "iteration=--wrong_iterations (default 30000). Implies --overwrite for those samples.",
    )
    parser.add_argument(
        "--wrong_iterations",
        type=int,
        default=30_000,
        help="Iteration value to detect in .work manifests when using --rerun_wrong_iterations.",
    )
    return parser.parse_args()


def _default_reuse_render_roots(args: argparse.Namespace) -> list[str]:
    roots: list[str] = []
    if args.reuse_render_root:
        roots.extend(args.reuse_render_root)
    # Prefer gaussian_sr lr_data next to this repo, then local output_root.
    gs_lr = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "gaussian_sr", "lr_data")
    )
    for candidate in (gs_lr, os.path.abspath(args.output_root)):
        if candidate not in roots and os.path.isdir(candidate):
            roots.append(candidate)
    return roots


def _build_splatfacto_config(args: argparse.Namespace) -> SplatfactoShapeNetConfig:
    return SplatfactoShapeNetConfig(
        num_views=args.num_views,
        num_train_views=args.num_train_views,
        hr_image_size=args.hr_image_size,
        lr_image_size=args.lr_image_size,
        focal_length=args.focal_length,
        background=args.background,
        hr_sh_degree=args.hr_sh_degree,
        seed=args.seed,
        sh_degree=args.sh_degree,
        max_num_iterations=args.max_num_iterations,
        stop_split_at=args.stop_split_at,
        num_random=args.num_random,
        random_scale=args.random_scale,
        cull_alpha_thresh=args.cull_alpha_thresh,
        num_downscales=args.num_downscales,
        split_screen_size=args.split_screen_size,
        background_color=args.background_color,
        vis=args.vis,
        splatformer_root=args.splatformer_root,
        num_test_views=args.num_test_views,
        skip_splatformer_export=args.skip_splatformer_export,
        test_camera_mode=args.test_camera_mode,
        test_image_size=args.test_image_size,
        reexport_splatformer=args.reexport_splatformer,
    )


def run_category_batch(args: argparse.Namespace, cfg: SplatfactoShapeNetConfig) -> None:
    if args.fill_missing and args.rerun_wrong_iterations:
        raise SystemExit("Use only one of --fill_missing or --rerun_wrong_iterations.")

    if (
        not args.fill_missing
        and not args.rerun_wrong_iterations
        and args.split == "all"
        and (args.num_samples is None or args.num_samples <= 0)
    ):
        raise SystemExit("--num_samples must be positive with --category_id --split all.")

    data_root = args.root_path
    output_root = args.output_root
    category_id = args.category_id
    reuse_roots = _default_reuse_render_roots(args)

    if args.split != "all":
        if not os.path.isfile(args.dataset_cache) and not os.path.isdir(
            os.path.splitext(args.dataset_cache)[0] + "_store"
        ):
            raise SystemExit(
                f"--dataset_cache not found: {args.dataset_cache}. "
                "Pass the gaussian_sr meta.pt used for Stage-1/2 splits."
            )
        ply_files = list_category_split_ply_files(
            data_root,
            category_id,
            args.dataset_cache,
            split=args.split,
            seed=cfg.seed,
        )
        print(
            f"Category {category_id} [{args.split}]: {len(ply_files)} samples "
            f"from {args.dataset_cache} (seed={cfg.seed})"
        )
        if not ply_files:
            print(
                f"WARNING: no {args.split}-split samples for {category_id} in the "
                "dataset cache (held-out category?). Falling back to every HR PLY "
                f"under {os.path.join(data_root, category_id)} (--split all)."
            )
            ply_files = list_category_ply_files(data_root, category_id)
    else:
        ply_files = list_category_ply_files(data_root, category_id)

    rerun_iteration = args.wrong_iterations if args.rerun_wrong_iterations else None
    batch_items = select_category_batch_items(
        data_root,
        output_root,
        category_id,
        offset=args.offset,
        num_samples=args.num_samples,
        fill_missing=args.fill_missing,
        rerun_iteration=rerun_iteration,
        dataset_cache=args.dataset_cache if args.split != "all" else None,
        split=args.split,
        seed=cfg.seed,
        ply_files=ply_files,
    )
    if not batch_items:
        return

    cat_out_dir = os.path.join(output_root, category_id)
    work_root = os.path.join(cat_out_dir, ".work")
    metrics_path = os.path.join(cat_out_dir, "metrics.json")
    os.makedirs(cat_out_dir, exist_ok=True)
    force_overwrite = args.overwrite or args.rerun_wrong_iterations

    print(
        f"Training config: views={cfg.num_views}, train_views={cfg.effective_num_train_views}, "
        f"LR={cfg.lr_image_size}px (scale={cfg.camera_res_scale_factor}), "
        f"iters={cfg.effective_max_num_iterations}, "
        f"stop_split_at={cfg.effective_stop_split_at}, "
        f"num_random={cfg.effective_num_random}, "
        f"random_scale={cfg.effective_random_scale}, "
        f"bg_color={cfg.effective_background_color}, "
        f"cull_alpha={cfg.cull_alpha_thresh}, "
        f"num_downscales={cfg.num_downscales}, "
        f"split_screen_size={cfg.effective_split_screen_size:.4f}"
    )
    if reuse_roots:
        print(f"Reuse render roots: {reuse_roots}")

    if args.rerun_wrong_iterations:
        print(
            f"Category {category_id} [{args.split}]: rerun iteration={args.wrong_iterations} "
            f"batch of {len(batch_items)} samples -> {cat_out_dir}"
        )
    elif args.fill_missing:
        print(
            f"Category {category_id} [{args.split}]: fill-missing batch of {len(batch_items)} "
            f"samples -> {cat_out_dir}"
        )
    else:
        print(
            f"Category {category_id} [{args.split}]: {len(batch_items)} samples -> {cat_out_dir}"
        )

    for sample_idx, hr_ply in batch_items:
        sample_stem = os.path.splitext(os.path.basename(hr_ply))[0]
        final_ply = os.path.join(cat_out_dir, f"{sample_stem}.ply")
        work_dir = os.path.join(work_root, sample_stem)

        print(f"\n[{sample_idx}] {sample_stem}")
        if not os.path.isfile(hr_ply):
            existing = find_existing_nerf_dataset(sample_stem, category_id, reuse_roots)
            if existing is None:
                print(f"[{sample_idx}] SKIP: missing HR PLY and no reusable nerf_dataset: {hr_ply}")
                continue
            print(f"[{sample_idx}] HR PLY missing; will reuse renders from {existing}")

        try:
            record = process_sample(
                hr_ply,
                work_dir,
                final_ply,
                cfg,
                skip_render=args.skip_render,
                skip_train=args.skip_train,
                skip_eval=args.skip_eval,
                overwrite=force_overwrite,
                render_device=args.render_device,
                render_backend=args.render_backend,
                reuse_render_roots=reuse_roots,
                category_id=category_id,
            )
            sample_metrics = record.get("sample_metrics")
            if sample_metrics is not None:
                merge_metrics_json_entry(metrics_path, sample_stem, sample_metrics)
        except Exception as exc:  # noqa: BLE001
            print(f"[{sample_idx}] FAILED: {exc}")
            traceback.print_exc()

    batch_stems = {os.path.splitext(os.path.basename(hr_ply))[0] for _, hr_ply in batch_items}
    metrics = load_metrics_json(metrics_path)
    batch_done = sum(1 for stem in batch_stems if stem in metrics)
    print(f"\nDone. {batch_done}/{len(batch_stems)} samples have metrics. JSON: {metrics_path}")
    print(f"SplatFormer scenes root: {os.path.abspath(cfg.splatformer_root)}")


def main() -> None:
    args = parse_args()
    cfg = _build_splatfacto_config(args)
    reuse_roots = _default_reuse_render_roots(args)

    if not torch.cuda.is_available() and not args.skip_train:
        print("WARNING: CUDA not available; splatfacto training requires a GPU.")

    if args.category_id is not None:
        run_category_batch(args, cfg)
        return

    if not args.hr_ply:
        raise SystemExit("Provide --hr_ply or --category_id.")

    ply_stem = os.path.splitext(os.path.basename(args.hr_ply))[0]
    work_dir = args.output_dir or os.path.join("lr_retrain_outputs", f"{ply_stem}_splatfacto")
    final_ply = args.final_ply or os.path.join(work_dir, "lr_splat_final.ply")

    record = process_sample(
        args.hr_ply,
        work_dir,
        final_ply,
        cfg,
        skip_render=args.skip_render,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        overwrite=args.overwrite,
        render_device=args.render_device,
        render_backend=args.render_backend,
        reuse_render_roots=reuse_roots,
        category_id=None,
    )
    print(f"Done. status={record.get('status')}  lr_ply={record.get('lr_ply')}")
    sf = record.get("splatformer")
    if sf:
        print(f"SplatFormer scene: {sf.get('nerfstudio_dir')} + {sf.get('colmap_dir')}")
        print(f"Eval with: configs/dataset/custom_lr.gin (root={cfg.splatformer_root})")


if __name__ == "__main__":
    main()
