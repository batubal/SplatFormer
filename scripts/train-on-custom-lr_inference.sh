#!/bin/bash
# Expects scenes exported with:
#   python reexport_splatformer_by_setting.py
#   (or train_lr_splats_splatfacto.py ... --test_camera_mode stage2_orbit --reexport_splatformer)
# so compare_with_input reports ~21 PSNR (Stage-2 8-view orbit @ 256px), not ~30 ns-eval.
#
# Usage:
#   bash scripts/train-on-custom-lr_inference.sh                 # test-set/customOOD
#   bash scripts/train-on-custom-lr_inference.sh dense_x2
#   bash scripts/train-on-custom-lr_inference.sh sparse4_x2
#   bash scripts/train-on-custom-lr_inference.sh dense_x4
#   bash scripts/train-on-custom-lr_inference.sh sparse4_x4
set -euo pipefail

SETTING="${SETTING:-${1:-}}"
case "${SETTING}" in
  ""|customOOD|default)
    GIN_FILE=configs/dataset/custom_lr.gin
    OUTPUT_DIR=outputs/custom_lr_splatformer
    EVAL_SUBDIR="${EVAL_SUBDIR:-test}"
    ;;
  dense_x2|sparse4_x2|dense_x4|sparse4_x4)
    GIN_FILE="configs/dataset/custom_lr_${SETTING}.gin"
    OUTPUT_DIR="outputs/custom_lr_splatformer_${SETTING}"
    EVAL_SUBDIR="${EVAL_SUBDIR:-eval}"
    ;;
  *)
    echo "Unknown setting '${SETTING}'."
    echo "Use: (empty)|customOOD|dense_x2|sparse4_x2|dense_x4|sparse4_x4"
    exit 1
    ;;
esac

if [[ ! -f "${GIN_FILE}" ]]; then
  echo "Missing ${GIN_FILE}. Run: python reexport_splatformer_by_setting.py"
  exit 1
fi

echo "SETTING=${SETTING:-customOOD}"
echo "GIN_FILE=${GIN_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}/${EVAL_SUBDIR}"

torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir="${OUTPUT_DIR}" \
    --gin_file="${GIN_FILE}" \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="FeaturePredictor.input_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
    --gin_param="FeaturePredictor.output_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
    --gin_param="FeaturePredictor.sh_degree=0" \
    --gin_param="build_trainloader.batch_size=1" \
    --only_eval --eval_subdir "${EVAL_SUBDIR}" --compare_with_input --save_viewer \
    --gin_param="FeaturePredictor.resume_ckpt='train-on-shapenet.pth'"
