"""Shared LR experiment settings (dense/sparse × 2×/4×). No heavy deps."""

from __future__ import annotations

import os
from dataclasses import dataclass


REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# (setting_name, scratch_folder) — matches train_lr_splats_splatfacto.slurm comments.
SETTING_SPECS: tuple[tuple[str, str], ...] = (
    ("dense_x2", "lr_data_splatformer"),
    ("sparse4_x2", "lr_data_splatformer_sparse4_x2"),
    ("dense_x4", "lr_data_splatformer_x4"),
    ("sparse4_x4", "lr_data_splatformer_sparse4_x4"),
)

SETTING_NAMES: tuple[str, ...] = tuple(name for name, _ in SETTING_SPECS)


def scratch_dir(scratch: str | None = None) -> str:
    if scratch:
        return scratch
    return os.environ.get("SCRATCH", "/arf/scratch/batubal")


@dataclass(frozen=True)
class Experiment:
    name: str
    output_root: str
    splatformer_root: str
    gin_file: str


def default_experiments(
    scratch: str | None = None, repo_dir: str | None = None
) -> list[Experiment]:
    scratch = scratch_dir(scratch)
    repo_dir = repo_dir or REPO_DIR
    out: list[Experiment] = []
    for name, folder in SETTING_SPECS:
        out.append(
            Experiment(
                name=name,
                output_root=os.path.join(scratch, folder),
                splatformer_root=os.path.join(repo_dir, "test-set", f"customOOD_{name}"),
                gin_file=os.path.join(repo_dir, "configs", "dataset", f"custom_lr_{name}.gin"),
            )
        )
    return out


def apply_output_root_maps(
    experiments: list[Experiment], maps: list[str] | None
) -> list[Experiment]:
    """Apply ``NAME=OUTPUT_ROOT`` overrides. Unknown names raise SystemExit."""
    if not maps:
        return experiments
    overrides: dict[str, str] = {}
    for item in maps:
        if "=" not in item:
            raise SystemExit(f"--map must be NAME=OUTPUT_ROOT, got {item!r}")
        name, path = item.split("=", 1)
        overrides[name.strip()] = os.path.abspath(path.strip())
    out: list[Experiment] = []
    for exp in experiments:
        if exp.name in overrides:
            out.append(
                Experiment(
                    name=exp.name,
                    output_root=overrides.pop(exp.name),
                    splatformer_root=exp.splatformer_root,
                    gin_file=exp.gin_file,
                )
            )
        else:
            out.append(exp)
    unknown = sorted(overrides)
    if unknown:
        raise SystemExit(
            f"Unknown --map setting(s): {unknown}. Known: {[e.name for e in experiments]}"
        )
    return out


def filter_experiments(
    experiments: list[Experiment], names: list[str] | None
) -> list[Experiment]:
    if not names:
        return experiments
    wanted = set(names)
    known = {e.name for e in experiments}
    missing = sorted(wanted - known)
    if missing:
        raise SystemExit(f"Unknown --setting {missing}. Known: {sorted(known)}")
    return [e for e in experiments if e.name in wanted]
