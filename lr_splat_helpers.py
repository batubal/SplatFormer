"""Self-contained helpers for train_lr_splats_splatfacto.py (no gaussian_sr imports)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np


def patch_numpy_pickle_aliases() -> None:
    """Allow pickle/torch.load to read arrays saved under a different NumPy major.

    NumPy 2 renamed ``numpy.core`` → ``numpy._core``. Loading a NumPy-2 pickle
    on NumPy 1 raises ``ModuleNotFoundError: No module named 'numpy._core'``.
    No-op on a working NumPy 2 install.
    """
    try:
        import numpy._core  # noqa: F401

        return
    except ImportError:
        pass

    import numpy.core as np_core  # type: ignore

    sys.modules.setdefault("numpy._core", np_core)
    for sub in (
        "multiarray",
        "numeric",
        "umath",
        "_multiarray_umath",
        "numerictypes",
        "overrides",
        "_dtype_ctypes",
        "_internal",
        "fromnumeric",
        "shape_base",
        "function_base",
        "multiarray_umath",
    ):
        alias = f"numpy._core.{sub}"
        if alias in sys.modules:
            continue
        legacy = f"numpy.core.{sub}"
        try:
            sys.modules[alias] = __import__(legacy, fromlist=["*"])
        except ImportError:
            if hasattr(np_core, sub):
                sys.modules[alias] = getattr(np_core, sub)


patch_numpy_pickle_aliases()

import torch
import torch.nn.functional as F
from plyfile import PlyData
from tqdm import tqdm

RenderBackend = Literal["auto", "gsplat", "torch"]


class _RenderCfg(Protocol):
    num_views: int
    hr_image_size: int
    lr_image_size: int
    focal_length: float
    background: str


@dataclass
class RenderConfig:
    camera_num_views: int = 72
    camera_image_size: int = 400
    camera_focal_length: float = 500.0
    camera_fit_ratio: float = 0.8


# ---------------------------------------------------------------------------
# Seeds / PLY I/O
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ply(
    path: str,
    sh_degree: int = 3,
    num_gaussians: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    plydata = PlyData.read(path)
    verts = plydata.elements[0]

    xyz = np.stack(
        (np.asarray(verts["x"]), np.asarray(verts["y"]), np.asarray(verts["z"])),
        axis=1,
    )
    xyz = torch.from_numpy(xyz)
    opacity = torch.from_numpy(np.asarray(verts["opacity"])[..., np.newaxis])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(verts["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(verts["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(verts["f_dc_2"])
    features_dc = torch.from_numpy(features_dc).transpose(1, 2)

    if sh_degree:
        extra_f_names = [p.name for p in verts.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(verts[attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )
        features_rest = torch.from_numpy(features_extra).transpose(1, 2)
        shs = torch.cat([features_dc, features_rest], dim=1).view(-1, (sh_degree + 1) ** 2 * 3)
    else:
        shs = features_dc.view(-1, 3)

    scale_names = sorted(
        (p.name for p in verts.properties if p.name.startswith("scale_")),
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(verts[attr_name])
    scaling = torch.from_numpy(scales)

    rot_names = sorted(
        (p.name for p in verts.properties if p.name.startswith("rot")),
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(verts[attr_name])
    rotation = torch.from_numpy(rots)

    gaussians = torch.cat([xyz, shs, opacity, scaling, rotation], dim=1).numpy()
    if num_gaussians is not None:
        np.random.seed(seed)
        np.random.shuffle(gaussians)
        gaussians = gaussians[:num_gaussians]
    return gaussians


def infer_sh_degree_from_ply(ply_path: str) -> int:
    props = [p.name for p in PlyData.read(ply_path).elements[0].properties]
    n_rest = sum(1 for name in props if name.startswith("f_rest_"))
    if n_rest == 0:
        return 0
    return int(round((n_rest / 3 + 1) ** 0.5 - 1))


def resolve_hr_sh_degree(ply_path: str, hr_sh_degree: int | None) -> int:
    if hr_sh_degree is not None:
        return hr_sh_degree
    return infer_sh_degree_from_ply(ply_path)


# ---------------------------------------------------------------------------
# Render device / backend
# ---------------------------------------------------------------------------


_BLOCK_WIDTH = 16


def gsplat_cuda_available() -> bool:
    """True if CUDA + a usable gsplat API (v0.1.x or v1+) is available."""
    if not torch.cuda.is_available():
        return False
    try:
        import gsplat

        if hasattr(gsplat, "project_gaussians") and hasattr(gsplat, "rasterize_gaussians"):
            return True
        if hasattr(gsplat, "rasterization"):
            return True
        from gsplat.cuda._backend import _C  # type: ignore

        return _C is not None
    except Exception:
        return False


def _gsplat_api_kind() -> str:
    """Return 'legacy' (v0.1.x), 'modern' (v1+), or raise."""
    import gsplat

    if hasattr(gsplat, "project_gaussians") and hasattr(gsplat, "rasterize_gaussians"):
        return "legacy"
    if hasattr(gsplat, "rasterization"):
        return "modern"
    raise ImportError(
        "Unsupported gsplat install: need project_gaussians/rasterize_gaussians (v0.1.x) "
        "or rasterization (v1+)."
    )


def resolve_render_setup(
    device: str,
    backend: RenderBackend,
) -> tuple[str, RenderBackend]:
    """Pick (device, backend). 'torch' uses a pure-PyTorch splat renderer."""
    if backend == "gsplat":
        if not gsplat_cuda_available():
            raise RuntimeError("gsplat backend requested but CUDA gsplat is not available.")
        return "cuda", "gsplat"

    if backend == "torch":
        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps", "torch"
            return "cpu", "torch"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
        if device == "cuda":
            print("Warning: --device cuda with --backend torch; using MPS/CPU splat instead of gsplat.")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps", "torch"
            return "cpu", "torch"
        return device, "torch"

    # auto backend
    if device == "auto":
        if gsplat_cuda_available():
            return "cuda", "gsplat"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", "torch"
        return "cpu", "torch"

    if device == "cuda":
        if gsplat_cuda_available():
            return "cuda", "gsplat"
        print("Warning: CUDA/gsplat unavailable; falling back to torch splat on MPS/CPU.")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", "torch"
        return "cpu", "torch"

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps", "torch"

    return device, "torch"


# ---------------------------------------------------------------------------
# HR splat rendering
# ---------------------------------------------------------------------------


def _num_sh_bands(sh_degree: int) -> int:
    return (sh_degree + 1) ** 2


def _gaussian_dim(sh_degree: int) -> int:
    appearance = 3 if sh_degree == 0 else _num_sh_bands(sh_degree) * 3
    return 3 + appearance + 1 + 3 + 4


def _validate_hr_gaussians(hr_gaussians: np.ndarray, sh_degree: int) -> None:
    expected = _gaussian_dim(sh_degree)
    actual = hr_gaussians.shape[1]
    if actual != expected:
        raise ValueError(
            f"HR PLY feature dim {actual} != expected {expected} for sh_degree={sh_degree}."
        )


def _unpack_hr_gaussians(
    gaussians: torch.Tensor,
    sh_degree: int,
) -> dict[str, torch.Tensor]:
    means = gaussians[:, :3]
    off = 3
    if sh_degree > 0:
        n_bands = _num_sh_bands(sh_degree)
        sh_coeffs = gaussians[:, off : off + n_bands * 3].reshape(-1, n_bands, 3)
        off += n_bands * 3
        dc_colors = None
    else:
        sh_coeffs = None
        dc_colors = gaussians[:, off : off + 3]
        off += 3
    opacities_logit = gaussians[:, off]
    scales_log = gaussians[:, off + 1 : off + 4]
    quats = gaussians[:, off + 4 : off + 8]
    return {
        "means": means,
        "sh_coeffs": sh_coeffs,
        "dc_colors": dc_colors,
        "opacities_logit": opacities_logit,
        "scales_log": scales_log,
        "quats": quats,
    }


def _view_dependent_rgb(
    means: torch.Tensor,
    sh_coeffs: torch.Tensor,
    camera_to_world: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """Evaluate view-dependent RGB from SH (pre +0.5 clamp). Compatible with gsplat 0.1.x/1.x."""
    import gsplat

    campos = camera_to_world[:3, 3]
    dirs = F.normalize(means - campos.unsqueeze(0), dim=-1)
    # gsplat v0.1.x: spherical_harmonics(degrees, dirs, coeffs)
    # gsplat v1+: may accept optional masks=; keep the 3-arg form for compatibility.
    return gsplat.spherical_harmonics(sh_degree, dirs, sh_coeffs)


def _scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    hr_size: int,
    lr_size: int,
) -> tuple[float, float, float, float]:
    scale = lr_size / float(hr_size)
    return fx * scale, fy * scale, cx * scale, cy * scale


def _hemisphere_unit_directions(num_views: int) -> list[tuple[float, float, float]]:
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions: list[tuple[float, float, float]] = []
    for i in range(num_views):
        t = (i + 0.5) / num_views
        cos_phi = 1.0 - t
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
        theta = golden_angle * i
        directions.append((sin_phi * math.cos(theta), sin_phi * math.sin(theta), cos_phi))
    return directions


def _look_at_camera_to_world(camera_pos: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    forward = center - camera_pos
    forward = forward / torch.norm(forward)
    up_world = torch.tensor([0.0, 0.0, 1.0], dtype=camera_pos.dtype, device=camera_pos.device)
    up_ref = (
        torch.tensor([0.0, 1.0, 0.0], dtype=camera_pos.dtype, device=camera_pos.device)
        if torch.dot(forward, up_world).abs() > 0.999
        else up_world
    )
    right = torch.linalg.cross(forward, up_ref)
    right = right / torch.norm(right)
    up = torch.linalg.cross(right, forward)

    c2w = torch.eye(4, dtype=camera_pos.dtype, device=camera_pos.device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


def generate_camera_params(points: torch.Tensor, config: RenderConfig) -> dict:
    num_views = config.camera_num_views
    image_size = config.camera_image_size
    focal_length = config.camera_focal_length
    fit_ratio = config.camera_fit_ratio

    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]
    center = (min_bounds + max_bounds) / 2
    radius = torch.norm(max_bounds - min_bounds) / 2

    proj_radius_target = max(float(fit_ratio * image_size) * 0.5, 1.0)
    fx_px = float(focal_length)
    if float(radius) > 0:
        distance_fit = fx_px * float(radius) / proj_radius_target
    else:
        distance_fit = 1.0
    camera_distance = max(2.0 * float(radius), distance_fit) * 1.05

    cameras = {
        "camera_to_worlds": [],
        "fx": torch.tensor(focal_length, dtype=center.dtype, device=center.device),
        "fy": torch.tensor(focal_length, dtype=center.dtype, device=center.device),
        "cx": torch.tensor(image_size / 2, dtype=center.dtype, device=center.device),
        "cy": torch.tensor(image_size / 2, dtype=center.dtype, device=center.device),
        "width": torch.tensor(image_size, dtype=center.dtype, device=center.device),
        "height": torch.tensor(image_size, dtype=center.dtype, device=center.device),
        "background_color": torch.tensor([0.0, 0.0, 0.0], dtype=center.dtype, device=center.device),
    }

    for dx, dy, dz in _hemisphere_unit_directions(num_views):
        offset = torch.tensor([dx, dy, dz], dtype=center.dtype, device=center.device) * camera_distance
        camera_pos = center + offset
        cameras["camera_to_worlds"].append(_look_at_camera_to_world(camera_pos, center))

    cameras["camera_to_worlds"] = torch.stack(cameras["camera_to_worlds"])
    return cameras


def generate_stage2_orbit_cameras(points: torch.Tensor, config: RenderConfig) -> dict:
    """Match gaussian_sr ``Renderer.generate_camera_params`` (8-view 45° orbit).

    Used for SplatFormer test cameras so input-baseline PSNR aligns with
    ``infer_stage2_diffusion.py`` (~21 dB) rather than ns-eval train-view scores.
    """
    num_views = config.camera_num_views
    image_size = config.camera_image_size
    focal_length = config.camera_focal_length
    fit_ratio = config.camera_fit_ratio

    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]
    center = (min_bounds + max_bounds) / 2
    radius = torch.norm(max_bounds - min_bounds) / 2

    proj_radius_target = max(float(fit_ratio * image_size) * 0.5, 1.0)
    fx_px = float(focal_length)
    if float(radius) > 0:
        distance_fit = fx_px * float(radius) / proj_radius_target
    else:
        distance_fit = 1.0
    camera_distance = max(2.0 * float(radius), distance_fit) * 1.05

    cameras = {
        "camera_to_worlds": [],
        "fx": torch.tensor(focal_length, dtype=center.dtype, device=center.device),
        "fy": torch.tensor(focal_length, dtype=center.dtype, device=center.device),
        "cx": torch.tensor(image_size / 2, dtype=center.dtype, device=center.device),
        "cy": torch.tensor(image_size / 2, dtype=center.dtype, device=center.device),
        "width": torch.tensor(image_size, dtype=center.dtype, device=center.device),
        "height": torch.tensor(image_size, dtype=center.dtype, device=center.device),
        "background_color": torch.tensor([0.0, 0.0, 0.0], dtype=center.dtype, device=center.device),
    }

    for i in range(num_views):
        azimuth = 2 * math.pi * i / num_views
        elevation = math.pi / 4
        x = center[0] + camera_distance * math.cos(azimuth) * math.sin(elevation)
        y = center[1] + camera_distance * math.sin(azimuth) * math.sin(elevation)
        z = center[2] + camera_distance * math.cos(elevation)
        camera_pos = torch.stack([x, y, z])
        cameras["camera_to_worlds"].append(_look_at_camera_to_world(camera_pos, center))

    cameras["camera_to_worlds"] = torch.stack(cameras["camera_to_worlds"])
    return cameras


def _background_tensor(background: str, device: torch.device) -> torch.Tensor:
    if background.lower() == "white":
        rgb = (1.0, 1.0, 1.0)
    elif background.lower() == "black":
        rgb = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unsupported background '{background}' (use white or black)")
    return torch.tensor(rgb, dtype=torch.float32, device=device)


def _hr_view_colors_for_gsplat(
    means: torch.Tensor,
    sh_coeffs: torch.Tensor | None,
    dc_colors: torch.Tensor | None,
    camera_to_world: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    if sh_degree > 0:
        assert sh_coeffs is not None
        return torch.clamp_min(
            _view_dependent_rgb(means, sh_coeffs, camera_to_world, sh_degree) + 0.5, 0.0
        )
    assert dc_colors is not None
    return dc_colors


def _rasterize_view_gsplat(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
) -> torch.Tensor:
    """Rasterize one view; supports gsplat v0.1.x (SplatFormer) and v1+."""
    import gsplat

    opa = opacities.reshape(-1, 1) if opacities.ndim == 1 else opacities
    api = _gsplat_api_kind()

    if api == "legacy":
        # Cameras from generate_camera_params are OpenCV (+Z forward); viewmat = w2c.
        xys, depths, radii, conics, _comp, num_tiles_hit, _cov3d = gsplat.project_gaussians(
            means,
            scales,
            1.0,
            quats,
            viewmat[:3, :].float(),
            fx,
            fy,
            cx,
            cy,
            height,
            width,
            _BLOCK_WIDTH,
        )
        rgb, _alpha = gsplat.rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            opa,
            height,
            width,
            _BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )
        return rgb.clamp(0, 1)

    # modern v1+ API
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=means.device,
        dtype=means.dtype,
    ).unsqueeze(0)
    rgb, alpha, _info = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.reshape(-1),
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=k,
        width=width,
        height=height,
        render_mode="RGB",
        packed=True,
    )
    return (rgb[0] + (1.0 - alpha[0]) * background).clamp(0, 1)


@torch.no_grad()
def _render_view_torch_with_alpha(
    means: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    world_to_cam: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = means.device
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    means_cam = (means @ R.T) + t

    z = means_cam[:, 2]
    valid = z > 1e-2
    if not bool(valid.any()):
        zeros = torch.zeros(height, width, 3, device=device)
        alpha = torch.zeros(height, width, device=device)
        return zeros, alpha

    u = fx * means_cam[:, 0] / z + cx
    v = fy * means_cam[:, 1] / z + cy
    max_scale = scales.max(dim=-1).values
    radius = (fx * max_scale / z.clamp(min=1e-2)).clamp(min=0.75, max=48.0)

    order = torch.argsort(z)
    image = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
    alpha_acc = torch.zeros(height, width, device=device, dtype=torch.float32)

    for idx in order.tolist():
        if not bool(valid[idx]):
            continue

        ui = float(u[idx])
        vi = float(v[idx])
        ri = float(radius[idx])
        alpha_i = opacities[idx]
        color_i = colors[idx]

        x0 = max(0, int(ui - ri - 1))
        x1 = min(width, int(ui + ri + 2))
        y0 = max(0, int(vi - ri - 1))
        y1 = min(height, int(vi + ri + 2))
        if x0 >= x1 or y0 >= y1:
            continue

        patch_x = torch.arange(x0, x1, device=device, dtype=torch.float32)
        patch_y = torch.arange(y0, y1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(patch_y, patch_x, indexing="ij")

        dist2 = (xx - ui) ** 2 + (yy - vi) ** 2
        g = torch.exp(-0.5 * dist2 / (ri * ri + 1e-6))

        a_patch = alpha_acc[y0:y1, x0:x1]
        contrib = (1.0 - a_patch) * alpha_i * g
        image[y0:y1, x0:x1] += contrib.unsqueeze(-1) * color_i
        alpha_acc[y0:y1, x0:x1] = a_patch + contrib

    return image.clamp(0, 1), alpha_acc.clamp(0, 1)


@torch.no_grad()
def _render_hr_views_with_background(
    hr_gaussians: np.ndarray,
    hr_cfg: RenderConfig,
    device: torch.device,
    backend: str,
    background: torch.Tensor,
    sh_degree: int,
    camera_params: dict | None = None,
    progress_desc: str = "Rendering HR views",
) -> tuple[torch.Tensor, dict]:
    tensor = torch.from_numpy(hr_gaussians).float().to(device)
    parts = _unpack_hr_gaussians(tensor, sh_degree)
    means = parts["means"]
    opacities = torch.sigmoid(parts["opacities_logit"]).reshape(-1)
    scales = torch.exp(parts["scales_log"].clamp(min=-6.0, max=1.5))
    quats = F.normalize(parts["quats"], dim=-1)
    sh_coeffs = parts["sh_coeffs"]
    dc_colors = parts["dc_colors"].sigmoid() if sh_degree == 0 else None

    if camera_params is None:
        camera_params = generate_camera_params(means, hr_cfg)
    c2w = camera_params["camera_to_worlds"].float().to(device)
    w2c = torch.linalg.inv(c2w)
    fx = float(camera_params["fx"])
    fy = float(camera_params["fy"])
    cx = float(camera_params["cx"])
    cy = float(camera_params["cy"])
    height = int(camera_params["height"])
    width = int(camera_params["width"])
    bg = background.to(device=device, dtype=means.dtype)

    views: list[torch.Tensor] = []
    for view_idx in tqdm(range(w2c.shape[0]), desc=progress_desc):
        if backend == "gsplat":
            view_colors = _hr_view_colors_for_gsplat(
                means, sh_coeffs, dc_colors, c2w[view_idx], sh_degree
            )
            views.append(
                _rasterize_view_gsplat(
                    means,
                    quats,
                    scales,
                    opacities,
                    view_colors,
                    w2c[view_idx],
                    fx,
                    fy,
                    cx,
                    cy,
                    width,
                    height,
                    bg,
                )
            )
        else:
            if sh_degree > 0:
                colors = torch.clamp_min(
                    _view_dependent_rgb(means, sh_coeffs, c2w[view_idx], sh_degree) + 0.5,
                    0.0,
                )
            else:
                colors = dc_colors
            rgb, alpha = _render_view_torch_with_alpha(
                means,
                colors,
                opacities,
                scales,
                w2c[view_idx],
                fx,
                fy,
                cx,
                cy,
                height,
                width,
            )
            rgb = rgb + (1.0 - alpha.unsqueeze(-1)) * bg
            views.append(rgb.clamp(0, 1))

    return torch.stack(views, dim=0), camera_params


@torch.no_grad()
def render_hr_lr_targets(
    hr_gaussians: np.ndarray,
    cfg: _RenderCfg,
    device: str,
    backend: str,
    hr_sh_degree: int,
) -> dict[str, Any]:
    """Render HR splat views and downsample to LR training targets."""
    _validate_hr_gaussians(hr_gaussians, hr_sh_degree)
    hr_cfg = RenderConfig(
        camera_num_views=cfg.num_views,
        camera_image_size=cfg.hr_image_size,
        camera_focal_length=cfg.focal_length,
    )
    torch_device = torch.device(device)
    background = _background_tensor(cfg.background, torch_device)
    hr_images, camera_params = _render_hr_views_with_background(
        hr_gaussians,
        hr_cfg,
        torch_device,
        backend,
        background,
        hr_sh_degree,
    )

    hr_chw = hr_images.permute(0, 3, 1, 2).contiguous()
    lr_chw = F.interpolate(
        hr_chw,
        size=(cfg.lr_image_size, cfg.lr_image_size),
        mode="area",
    )
    lr_images = lr_chw.permute(0, 2, 3, 1).contiguous()

    fx_hr = float(camera_params["fx"])
    fy_hr = float(camera_params["fy"])
    cx_hr = float(camera_params["cx"])
    cy_hr = float(camera_params["cy"])
    fx_lr, fy_lr, cx_lr, cy_lr = _scale_intrinsics(
        fx_hr, fy_hr, cx_hr, cy_hr, cfg.hr_image_size, cfg.lr_image_size
    )

    c2w = camera_params["camera_to_worlds"].float()
    return {
        "lr_images": lr_images.cpu(),
        "hr_images": hr_images.cpu(),
        "camera_to_worlds": c2w.cpu(),
        "fx": fx_lr,
        "fy": fy_lr,
        "cx": cx_lr,
        "cy": cy_lr,
        "width": cfg.lr_image_size,
        "height": cfg.lr_image_size,
        "fx_hr": fx_hr,
        "fy_hr": fy_hr,
        "cx_hr": cx_hr,
        "cy_hr": cy_hr,
    }


@torch.no_grad()
def render_hr_stage2_orbit_views(
    hr_gaussians: np.ndarray,
    *,
    num_views: int = 8,
    image_size: int = 256,
    focal_length: float = 500.0,
    background: str = "white",
    device: str = "cpu",
    backend: str = "torch",
    hr_sh_degree: int = 0,
) -> dict[str, Any]:
    """Render HR GT at Stage-2 orbit cameras (OpenCV +Z forward c2w)."""
    _validate_hr_gaussians(hr_gaussians, hr_sh_degree)
    hr_cfg = RenderConfig(
        camera_num_views=num_views,
        camera_image_size=image_size,
        camera_focal_length=focal_length,
    )
    torch_device = torch.device(device)
    means = torch.from_numpy(hr_gaussians[:, :3]).float()
    camera_params = generate_stage2_orbit_cameras(means, hr_cfg)
    bg = _background_tensor(background, torch_device)
    hr_images, camera_params = _render_hr_views_with_background(
        hr_gaussians,
        hr_cfg,
        torch_device,
        backend,
        bg,
        hr_sh_degree,
        camera_params=camera_params,
        progress_desc="Rendering Stage-2 test views",
    )
    return {
        "hr_images": hr_images.cpu(),
        "camera_to_worlds": camera_params["camera_to_worlds"].float().cpu(),
        "fx": float(camera_params["fx"]),
        "fy": float(camera_params["fy"]),
        "cx": float(camera_params["cx"]),
        "cy": float(camera_params["cy"]),
        "width": int(image_size),
        "height": int(image_size),
    }


# ---------------------------------------------------------------------------
# Category batch / metrics helpers
# ---------------------------------------------------------------------------


def list_category_ply_files(data_root: str, category_id: str) -> list[str]:
    cat_dir = os.path.join(data_root, category_id)
    if not os.path.isdir(cat_dir):
        raise FileNotFoundError(f"Category directory not found: {cat_dir}")
    names = sorted(
        name
        for name in os.listdir(cat_dir)
        if name.endswith(".ply") and name.startswith(f"{category_id}-")
    )
    return [os.path.join(cat_dir, name) for name in names]


def lr_ply_path_for_hr_ply(hr_ply: str, output_root: str, category_id: str) -> str:
    sample_stem = os.path.splitext(os.path.basename(hr_ply))[0]
    return os.path.join(output_root, category_id, f"{sample_stem}.ply")


def resolve_split_metadata_path(cache_path: str) -> str:
    """Return the file holding {filenames, category_labels} (gaussian_sr cache / store)."""
    if os.path.isfile(cache_path):
        return cache_path
    store_meta = os.path.join(os.path.splitext(cache_path)[0] + "_store", "meta.pt")
    if os.path.isfile(store_meta):
        return store_meta
    raise FileNotFoundError(
        f"Dataset split metadata not found. Looked for:\n  {cache_path}\n  {store_meta}\n"
        "Point --dataset_cache at gaussian_sr meta.pt (or the lazy-store meta.pt)."
    )


def load_split_metadata(cache_path: str) -> dict[str, Any]:
    """Load gaussian_sr split metadata, tolerating NumPy 1↔2 pickle module renames."""
    meta_path = resolve_split_metadata_path(cache_path)
    patch_numpy_pickle_aliases()
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        msg = str(exc)
        if "numpy._core" not in msg and "numpy.core" not in msg:
            raise
        # Map the exact missing submodule, then retry once.
        import sys

        missing = msg.split("'")[1] if "'" in msg else ""
        if missing.startswith("numpy._core."):
            legacy = "numpy.core." + missing[len("numpy._core.") :]
            try:
                sys.modules[missing] = __import__(legacy, fromlist=["*"])
            except ImportError as inner:
                raise exc from inner
        elif missing == "numpy._core":
            import numpy.core as np_core  # type: ignore

            sys.modules["numpy._core"] = np_core
        else:
            raise
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    if not isinstance(meta, dict) or "filenames" not in meta or "category_labels" not in meta:
        raise KeyError(
            f"{meta_path} must contain 'filenames' and 'category_labels' "
            f"(got keys={list(meta.keys()) if isinstance(meta, dict) else type(meta)})"
        )
    return meta


def compute_dataset_split_indices(
    cache_path: str,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], list[str]]:
    """Reproduce gaussian_sr's stratified 70/15/15 train/val/test split."""
    from sklearn.model_selection import train_test_split

    meta = load_split_metadata(cache_path)
    filenames = list(meta["filenames"])
    category_labels = meta["category_labels"]
    if hasattr(category_labels, "tolist"):
        category_labels = category_labels.tolist()

    n = len(filenames)
    train_idx, temp_idx = train_test_split(
        list(range(n)), test_size=0.3, random_state=seed, stratify=category_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=[category_labels[i] for i in temp_idx],
    )
    return train_idx, val_idx, test_idx, filenames


def _filename_category_id(filename: str) -> str:
    """``02691156/02691156-xxx.ply`` → ``02691156``."""
    rel = filename.replace("\\", "/")
    if "/" in rel:
        return rel.split("/", 1)[0]
    stem = os.path.splitext(os.path.basename(rel))[0]
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem


def list_category_split_ply_files(
    data_root: str,
    category_id: str,
    dataset_cache: str,
    *,
    split: str = "test",
    seed: int = 42,
) -> list[str]:
    """
    HR PLY paths for one ShapeNet category that fall in gaussian_sr's ``split``.

    ``split`` is one of: train, val, test, all.
    """
    split = split.lower().strip()
    if split == "all":
        return list_category_ply_files(data_root, category_id)

    train_idx, val_idx, test_idx, filenames = compute_dataset_split_indices(
        dataset_cache, seed=seed
    )
    if split == "train":
        indices = train_idx
    elif split in ("val", "valid", "validation"):
        indices = val_idx
    elif split == "test":
        indices = test_idx
    else:
        raise ValueError(f"Unknown split={split!r}; use train, val, test, or all.")

    out: list[str] = []
    for i in indices:
        rel = filenames[i].replace("\\", "/")
        if _filename_category_id(rel) != category_id:
            continue
        abs_path = os.path.join(data_root, rel)
        out.append(abs_path)
    return out


def find_existing_nerf_dataset(
    sample_stem: str,
    category_id: str,
    reuse_roots: list[str] | tuple[str, ...] | None,
) -> str | None:
    """Return an existing ``nerf_dataset`` dir (72-view HR renders) if present."""
    if not reuse_roots:
        return None
    for root in reuse_roots:
        if not root:
            continue
        candidate = os.path.join(root, category_id, ".work", sample_stem, "nerf_dataset")
        transforms = os.path.join(candidate, "transforms_train.json")
        if os.path.isfile(transforms):
            return os.path.abspath(candidate)
    return None


def list_missing_category_lr_samples(
    data_root: str,
    output_root: str,
    category_id: str,
    ply_files: list[str] | None = None,
) -> list[tuple[int, str, str]]:
    ply_files = list_category_ply_files(data_root, category_id) if ply_files is None else ply_files
    missing: list[tuple[int, str, str]] = []
    for idx, hr_ply in enumerate(ply_files):
        lr_ply = lr_ply_path_for_hr_ply(hr_ply, output_root, category_id)
        if not os.path.isfile(lr_ply):
            missing.append((idx, hr_ply, lr_ply))
    return missing


def _manifest_iteration(manifest: dict[str, Any]) -> int | None:
    config = manifest.get("config")
    if isinstance(config, dict) and config.get("max_num_iterations") is not None:
        try:
            return int(config["max_num_iterations"])
        except (TypeError, ValueError):
            pass

    sample_metrics = manifest.get("sample_metrics")
    if isinstance(sample_metrics, dict) and sample_metrics.get("iteration") not in (None, ""):
        try:
            return int(sample_metrics["iteration"])
        except (TypeError, ValueError):
            pass
    return None


def list_category_samples_with_manifest_iteration(
    data_root: str,
    output_root: str,
    category_id: str,
    iteration: int,
    ply_files: list[str] | None = None,
) -> list[tuple[int, str]]:
    ply_files = list_category_ply_files(data_root, category_id) if ply_files is None else ply_files
    ply_by_stem = {
        os.path.splitext(os.path.basename(hr_ply))[0]: (idx, hr_ply)
        for idx, hr_ply in enumerate(ply_files)
    }

    work_root = os.path.join(output_root, category_id, ".work")
    items: list[tuple[int, str]] = []
    if not os.path.isdir(work_root):
        return items

    for stem in sorted(os.listdir(work_root)):
        manifest_path = os.path.join(work_root, stem, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "ok":
            continue
        if _manifest_iteration(manifest) != iteration:
            continue
        if stem in ply_by_stem:
            items.append(ply_by_stem[stem])

    items.sort(key=lambda pair: pair[0])
    return items


def select_category_batch_items(
    data_root: str,
    output_root: str,
    category_id: str,
    *,
    offset: int,
    num_samples: int | None,
    fill_missing: bool,
    rerun_iteration: int | None = None,
    dataset_cache: str | None = None,
    split: str = "test",
    seed: int = 42,
    ply_files: list[str] | None = None,
) -> list[tuple[int, str]]:
    if fill_missing and rerun_iteration is not None:
        raise ValueError("Use only one of fill_missing or rerun_iteration.")

    if ply_files is None:
        if dataset_cache and split.lower().strip() != "all":
            ply_files = list_category_split_ply_files(
                data_root,
                category_id,
                dataset_cache,
                split=split,
                seed=seed,
            )
        else:
            ply_files = list_category_ply_files(data_root, category_id)

    if rerun_iteration is not None:
        items = list_category_samples_with_manifest_iteration(
            data_root, output_root, category_id, rerun_iteration, ply_files=ply_files
        )
        print(
            f"Category {category_id} [{split}]: {len(items)}/{len(ply_files)} samples in .work "
            f"manifests with iteration={rerun_iteration}"
        )
        if not items:
            print(f"Nothing to do — no manifests with iteration={rerun_iteration}.")
            return []
    elif fill_missing:
        missing = list_missing_category_lr_samples(
            data_root, output_root, category_id, ply_files=ply_files
        )
        items = [(idx, hr_ply) for idx, hr_ply, _ in missing]
        print(
            f"Category {category_id} [{split}]: {len(missing)}/{len(ply_files)} HR samples "
            f"missing LR output in {os.path.join(output_root, category_id)}"
        )
        if not items:
            print("Nothing to do — all LR plys are present.")
            return []
    else:
        items = list(enumerate(ply_files))

    if offset < 0:
        raise ValueError(f"--offset must be >= 0, got {offset}")

    flexible_batch = fill_missing or rerun_iteration is not None
    if num_samples is None or num_samples <= 0:
        if flexible_batch or (dataset_cache and split.lower().strip() != "all"):
            # Test/val/train split batches may omit --num_samples (process the whole split).
            end = len(items)
        else:
            raise ValueError(
                "--num_samples must be positive when not using --fill_missing, "
                "--rerun_wrong_iterations, or a dataset split."
            )
    else:
        end = min(offset + num_samples, len(items))

    if offset >= len(items):
        if flexible_batch or (dataset_cache and split.lower().strip() != "all"):
            print(f"--offset {offset} is beyond the selected list ({len(items)} items); nothing to do.")
            return []
        raise IndexError(
            f"--offset {offset} is out of range for category {category_id} ({len(items)} items)."
        )

    selected = items[offset:end]
    if fill_missing:
        print(f"Processing missing samples [{offset}, {offset + len(selected)}) ({len(selected)} items).")
    elif rerun_iteration is not None:
        print(
            f"Rerunning iteration={rerun_iteration} samples "
            f"[{offset}, {offset + len(selected)}) ({len(selected)} items)."
        )
    else:
        print(
            f"Category {category_id} [{split}]: processing [{offset}, {offset + len(selected)}) "
            f"of {len(items)} samples."
        )
    return selected


def load_metrics_json(path: str) -> dict[str, Any]:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metrics_json(path: str, metrics: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def merge_metrics_json_entry(path: str, stem: str, entry: dict[str, Any]) -> None:
    """Merge one sample into metrics.json with an exclusive file lock (parallel-job safe)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-Unix fallback
        metrics = load_metrics_json(path)
        metrics[stem] = entry
        write_metrics_json(path, metrics)
        return

    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            metrics = json.loads(raw) if raw else {}
            metrics[stem] = entry
            f.seek(0)
            f.truncate()
            json.dump(metrics, f, indent=4)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
