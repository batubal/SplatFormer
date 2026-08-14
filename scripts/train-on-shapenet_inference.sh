torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs/shapenet_splatformer \
    --gin_file=configs/dataset/shapenet.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="FeaturePredictor.input_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
    --gin_param="FeaturePredictor.output_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
    --gin_param="FeaturePredictor.sh_degree=0" \
    --gin_param="build_trainloader.batch_size=1" \
    --only_eval --eval_subdir test --compare_with_input \
    --gin_param="FeaturePredictor.resume_ckpt='train-on-shapenet.pth'" 


