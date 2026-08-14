#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A batubal
#SBATCH -J lr-splat-sf
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -C H200
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xcvbatuhanbal@gmail.com
set -euo pipefail

# --- Paths (edit if needed) ---
REPO_DIR=/arf/home/batubal/repos/SplatFormer
GS_REPO_DIR=/arf/home/batubal/repos/gaussian_sr
SCRATCH_DIR=${SCRATCH:-/arf/scratch/batubal}
ENV_DIR=${SCRATCH_DIR}/conda/envs/nerfstudio

# Train ≤N gaussian_sr *test-split* samples per category (fewer if the split is smaller).
#
# Default = dense 2× (72 train views, LR=200 / HR=400):
#   sbatch train_lr_splats_splatfacto.slurm
#
# Sparse-view 2× (4 of 72 views, LR=200):
#   sbatch --export=ALL,NUM_TRAIN_VIEWS=4,OUTPUT_ROOT=/arf/scratch/batubal/lr_data_splatformer_sparse4_x2 \
#     train_lr_splats_splatfacto.slurm
#
# Dense 4× (72 views, LR=100 / HR=400):
#   sbatch --export=ALL,LR_IMAGE_SIZE=100,OUTPUT_ROOT=/arf/scratch/batubal/lr_data_splatformer_x4 \
#     train_lr_splats_splatfacto.slurm
#
# Sparse-view 4× (4 views + LR=100):
#   sbatch --export=ALL,NUM_TRAIN_VIEWS=4,LR_IMAGE_SIZE=100,OUTPUT_ROOT=/arf/scratch/batubal/lr_data_splatformer_sparse4_x4 \
#     train_lr_splats_splatfacto.slurm
#
# Single category / overwrite examples:
#   sbatch --export=ALL,CATEGORY_ID=02691156,NUM_SAMPLES=10 train_lr_splats_splatfacto.slurm
#   sbatch --export=ALL,NUM_SAMPLES=30,OVERWRITE=1 train_lr_splats_splatfacto.slurm
CATEGORY_ID=${CATEGORY_ID:-}          # empty → all categories in dataset_cache
OFFSET=${OFFSET:-0}
NUM_SAMPLES=${NUM_SAMPLES:-10}        # at most this many test-split samples / category
SPLIT=${SPLIT:-test}
ROOT_PATH=${ROOT_PATH:-${SCRATCH_DIR}/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SCRATCH_DIR}/lr_data_splatformer}
REUSE_RENDER_ROOT=${REUSE_RENDER_ROOT:-${SCRATCH_DIR}/lr_data}
DATASET_CACHE=${DATASET_CACHE:-${GS_REPO_DIR}/dataset_percentile_dynamic_v2_store/meta.pt}
# Fallbacks if the store meta is missing on this node
if [[ ! -f "${DATASET_CACHE}" ]]; then
  DATASET_CACHE=${GS_REPO_DIR}/meta.pt
fi

# Training knobs (match gaussian_sr defaults unless overridden)
NUM_VIEWS=${NUM_VIEWS:-72}            # hemisphere cameras rendered / available
NUM_TRAIN_VIEWS=${NUM_TRAIN_VIEWS:-0} # 0 = all views (dense); 4 = sparse-view
HR_IMAGE_SIZE=${HR_IMAGE_SIZE:-400}
LR_IMAGE_SIZE=${LR_IMAGE_SIZE:-200}   # 200 → 2×, 100 → 4× (scale = LR/HR)

OVERWRITE=${OVERWRITE:-0}
SKIP_EVAL=${SKIP_EVAL:-1}
SKIP_SPLATFORMER_EXPORT=${SKIP_SPLATFORMER_EXPORT:-0}
FILL_MISSING=${FILL_MISSING:-0}
RERUN_WRONG_ITERATIONS=${RERUN_WRONG_ITERATIONS:-0}
WRONG_ITERATIONS=${WRONG_ITERATIONS:-30000}

CPU_KEEPALIVE=${CPU_KEEPALIVE:-1}

mkdir -p "${REPO_DIR}/slurm_logs"
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

# --- Modules (TRUBA) ---
module purge
module load miniconda3
module load lib/cuda/12.4
module load comp/gcc/12.3.0

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

export CONDA_PKGS_DIRS="${SCRATCH_DIR}/conda/pkgs"
export PIP_CACHE_DIR="${SCRATCH_DIR}/pip-cache"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1

KEEPALIVE_PID=""
start_cpu_keepalive() {
  if [[ "${CPU_KEEPALIVE}" != "1" ]]; then
    echo "CPU keepalive disabled (CPU_KEEPALIVE=${CPU_KEEPALIVE})"
    return
  fi
  local workers=$((SLURM_CPUS_PER_TASK - 1))
  if [[ "${workers}" -lt 1 ]]; then
    echo "CPU keepalive skipped (SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK})"
    return
  fi
  if [[ -f "${GS_REPO_DIR}/slurm_cpu_keepalive.py" ]]; then
    python -u "${GS_REPO_DIR}/slurm_cpu_keepalive.py" --workers "${workers}" &
    KEEPALIVE_PID=$!
    echo "Started slurm_cpu_keepalive.py pid=${KEEPALIVE_PID} workers=${workers}"
  else
    echo "CPU keepalive skipped (missing ${GS_REPO_DIR}/slurm_cpu_keepalive.py)"
  fi
}

stop_cpu_keepalive() {
  if [[ -n "${KEEPALIVE_PID}" ]]; then
    kill "${KEEPALIVE_PID}" 2>/dev/null || true
    wait "${KEEPALIVE_PID}" 2>/dev/null || true
    echo "Stopped slurm_cpu_keepalive.py pid=${KEEPALIVE_PID}"
  fi
}

trap stop_cpu_keepalive EXIT
start_cpu_keepalive

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "REPO_DIR: ${REPO_DIR}"
echo "CATEGORY_ID: ${CATEGORY_ID:-ALL}  SPLIT: ${SPLIT}  OFFSET: ${OFFSET}  NUM_SAMPLES: ${NUM_SAMPLES}"
echo "NUM_VIEWS: ${NUM_VIEWS}  NUM_TRAIN_VIEWS: ${NUM_TRAIN_VIEWS} (0=all)  HR: ${HR_IMAGE_SIZE}  LR: ${LR_IMAGE_SIZE}"
echo "ROOT_PATH: ${ROOT_PATH}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "REUSE_RENDER_ROOT: ${REUSE_RENDER_ROOT}"
echo "DATASET_CACHE: ${DATASET_CACHE}"
echo "OVERWRITE: ${OVERWRITE}  SKIP_EVAL: ${SKIP_EVAL}  FILL_MISSING: ${FILL_MISSING}"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "
import importlib.metadata
import nerfstudio
try:
    ver = importlib.metadata.version('nerfstudio')
except importlib.metadata.PackageNotFoundError:
    ver = 'installed (version unknown)'
print('nerfstudio', ver)
"
python -c "import nerfstudio.scripts.train; print('ns-train module OK')"

test -f "${DATASET_CACHE}" || {
  echo "ERROR: dataset cache not found: ${DATASET_CACHE}"
  exit 1
}
test -d "${ROOT_PATH}" || {
  echo "ERROR: ROOT_PATH not found: ${ROOT_PATH}"
  exit 1
}

# Resolve category list: single CATEGORY_ID, or every category present in the test split.
if [[ -n "${CATEGORY_ID}" ]]; then
  CATEGORIES=("${CATEGORY_ID}")
else
  mapfile -t CATEGORIES < <(python - <<PY
from lr_splat_helpers import compute_dataset_split_indices
_,_,test_idx,filenames = compute_dataset_split_indices("${DATASET_CACHE}", seed=42)
cats=sorted({filenames[i].replace("\\\\","/").split("/",1)[0] for i in test_idx})
cats=[c for c in cats if c and not c.startswith(".")]
print("\\n".join(cats))
PY
)
fi

echo "Categories (${#CATEGORIES[@]}): ${CATEGORIES[*]}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${SKIP_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_eval)
fi
if [[ "${SKIP_SPLATFORMER_EXPORT}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_splatformer_export)
fi
if [[ "${FILL_MISSING}" == "1" ]]; then
  EXTRA_ARGS+=(--fill_missing)
fi
if [[ "${RERUN_WRONG_ITERATIONS}" == "1" ]]; then
  EXTRA_ARGS+=(--rerun_wrong_iterations)
  EXTRA_ARGS+=(--wrong_iterations "${WRONG_ITERATIONS}")
fi

FAILED=0
for CAT in "${CATEGORIES[@]}"; do
  echo "============================================================"
  echo "Category ${CAT}: ≤${NUM_SAMPLES} samples from split=${SPLIT}"
  echo "============================================================"
  mkdir -p "${OUTPUT_ROOT}/${CAT}"

  if [[ ! -d "${ROOT_PATH}/${CAT}" ]]; then
    echo "WARNING: missing ${ROOT_PATH}/${CAT}; skipping ${CAT}"
    continue
  fi

  set +e
  python -u train_lr_splats_splatfacto.py \
    --category_id "${CAT}" \
    --split "${SPLIT}" \
    --offset "${OFFSET}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_views "${NUM_VIEWS}" \
    --num_train_views "${NUM_TRAIN_VIEWS}" \
    --hr_image_size "${HR_IMAGE_SIZE}" \
    --lr_image_size "${LR_IMAGE_SIZE}" \
    --root_path "${ROOT_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --dataset_cache "${DATASET_CACHE}" \
    --reuse_render_root "${REUSE_RENDER_ROOT}" \
    "${EXTRA_ARGS[@]}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: category ${CAT} failed with exit code ${rc}"
    FAILED=$((FAILED + 1))
  else
    echo "Done category ${CAT}. LR plys → ${OUTPUT_ROOT}/${CAT}/"
  fi
done

echo "All categories finished. failures=${FAILED}/${#CATEGORIES[@]}"
echo "Outputs under ${OUTPUT_ROOT}/"
exit "${FAILED}"
