import torch, os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sys
from torch.optim import SGD
import wandb, json
import argparse
import cv2
from models.feature_predictor import FeaturePredictor

from collections import OrderedDict
import random
import gin 
from absl import app, flags
from dataset.Loader import build_trainloader, build_testloader
from utils import gpu_utils, gs_utils, loss_utils
from utils.optimizers import build_optimizer, build_scheduler
from utils.metrics import MetricComputer
from utils.log_utils import ProcessSafeLogger
from utils.metrics import psnr
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', '/cluster/scratch/chenyut/wandb', 'Wandbs Output directory')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '', 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS

@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_grid(imgs, nrow=3, ncols=3):
    img_h, img_w = imgs[0].shape[:2]
    if imgs[0].ndim == 3:
        grid = np.zeros((img_h*nrow, img_w*ncols, 3), dtype=np.uint8)
    elif imgs[0].ndim == 2:
        grid = np.zeros((img_h*nrow, img_w*ncols), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncols):
            if i*ncols+j >= len(imgs):
                break
            grid[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = imgs[i*ncols+j]
    return grid

@gin.configurable
def evaluation(model, test_loader, output_dir, output_gt, compare_with_pseudo, 
              compare_with_input=False,
              save_as_single=False,
              save_viewer=False,
              evaluate_input=False):
    model.eval()
    metric_computer = MetricComputer()
    if compare_with_input:
      metric_computer_input = MetricComputer()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
      cnt = 0
      num_images, num_scenes = 0, 0
      pseudo_loss = {}
      for test_batch in tqdm(test_loader): #A scene at a time
        test_batch_gs = gpu_utils.move_to_device([data['gs_params'] for data in test_batch],model.device)
        test_batch_cameras = gpu_utils.move_to_device([data['cameras'] for data in test_batch],model.device)
        test_batch_images = gpu_utils.move_to_device([data['images'] for data in test_batch],model.device)
        test_batch_idx = [data['scene_idx'] for data in test_batch]
        test_batch_name = [data['scene_name'] for data in test_batch]
        test_batch_imgname = [data['images_name'] for data in test_batch]
        forward_kwargs = {'batch_normalized_gs': test_batch_gs, 'batch_scene_idx': test_batch_idx}

        out_test_batch_gs = model(**forward_kwargs)
        for iii, (out_gs, in_gs, cameras, gt_imgs, scene_idx) in enumerate(zip(out_test_batch_gs, test_batch_gs, test_batch_cameras, test_batch_images, test_batch_idx)):
          if evaluate_input:
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras)
          else:
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, cameras) # List of torch.tensor([H,W,3])
          pred_imgs = torch.stack(pred_imgs, dim=0) #torch.tensor([N,H,W,3])
          gt_imgs = torch.stack(gt_imgs, dim=0) #torch.tensor([N,H,W,3])
          
          if gt_imgs.shape[-1] == 4:
            # only for real images
            masks = gt_imgs[...,3].unsqueeze(-1)
            pred_imgs = pred_imgs*masks
            gt_imgs = (gt_imgs[...,:3]*255).to(torch.uint8)
            pred_imgs = (pred_imgs*255).to(torch.uint8)
          else:
            masks = None
            gt_imgs = (gt_imgs*255).to(torch.uint8)
            pred_imgs = (pred_imgs*255).to(torch.uint8)

          imgs = [im.cpu().numpy().astype(np.uint8) for im in pred_imgs]
          grid = make_grid(imgs)
          grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(output_dir, f'scene{scene_idx}_pred.png'), grid)
          
          if output_gt:
            gt_imgs_ = [im.cpu().numpy().astype(np.uint8) for im in gt_imgs]
            grid = make_grid(gt_imgs_)
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f'scene{scene_idx}_gt.png'), grid)

          metric_computer.update(pred_imgs, gt_imgs, name=f'{scene_idx}')
          if compare_with_input:
            input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras)
            input_imgs = torch.stack(input_imgs, dim=0)
            if masks is not None:
              input_imgs = input_imgs*masks
              input_imgs = (input_imgs*255).to(torch.uint8)
            else:
              input_imgs = (input_imgs*255).to(torch.uint8)

            metric_computer_input.update(input_imgs, gt_imgs, name=f'{scene_idx}')
            output_dir_thisscene = os.path.join(output_dir, f'compare/{test_batch_name[iii]}')
            os.makedirs(output_dir_thisscene, exist_ok=True)

            #save ([gt, input, pred])
            for ii, (gt_img, input_img, pred_img) in enumerate(zip(gt_imgs, input_imgs, pred_imgs)):
              gt_img = gt_img.cpu().numpy().astype(np.uint8)
              input_img = input_img.cpu().numpy().astype(np.uint8)
              pred_img = pred_img.cpu().numpy().astype(np.uint8)
              cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1) #H,W*3
              cv2.imwrite(os.path.join(output_dir_thisscene, f'{ii:02d}.png'), cmp_img[:,:,::-1])

          if save_as_single:
            output_dir_thisscene_single = os.path.join(output_dir, f'pred/{test_batch_name[iii]}')
            os.makedirs(output_dir_thisscene_single, exist_ok=True)
            for ii,pred_img in enumerate(pred_imgs):
              pred_img = pred_img.cpu().numpy().astype(np.uint8)
              cv2.imwrite(os.path.join(output_dir_thisscene_single, test_batch_imgname[iii][ii]), pred_img[:,:,::-1])
          if save_viewer:
            viewerdir = os.path.join(output_dir, f'viewer/{test_batch_name[iii]}')
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(cameras, viewerdir, model.module.sh_degree)
            # Save input 3dgs
            gs_utils.export_ply_forviewer(gs_params=in_gs, filename=os.path.join(viewerdir, 'point_cloud/iteration_0/point_cloud.ply'))
            gs_utils.export_ply_forviewer(gs_params=out_gs, filename=os.path.join(viewerdir, 'point_cloud/iteration_1/point_cloud.ply'))
          cnt += 1
          num_images += len(pred_imgs)
      metrics = metric_computer.sum() # We need to sum the metrics
      metric_computer.write_to_file(os.path.join(output_dir, f'metrics.rank{dist.get_rank()}.json'))
      if compare_with_input:
        metric_computer_input.write_to_file(os.path.join(output_dir, f'metrics_input.rank{dist.get_rank()}.json'))
    model.train()

    num_images = torch.tensor([num_images]).to(model.device)
    num_scenes = torch.tensor([num_scenes]).to(model.device)
    torch.distributed.reduce(num_images, dst=0) #Sum
    torch.distributed.reduce(num_scenes, dst=0) #Sum
    for key in metrics:
      torch.distributed.reduce(metrics[key], dst=0) #Sum
      if dist.get_rank() == 0:
        metrics[key] = (metrics[key]/num_images).item()
    if compare_with_pseudo:
      for key in pseudo_loss:
        torch.distributed.reduce(pseudo_loss[key], dst=0)
        if dist.get_rank() == 0:
          metrics[f'pseudo_loss_{key}'] = (pseudo_loss[key]/num_scenes).item()
        
    if compare_with_input:
      metrics_input = metric_computer_input.sum()
      for key in metrics_input:
        torch.distributed.reduce(metrics_input[key], dst=0)
        if dist.get_rank() == 0:
          metrics_input[key] = (metrics_input[key]/num_images).item()
    else:
      metrics_input = {}
    return metrics, metrics_input

       
@gin.configurable
def training(
    model, optimizer_, scheduler_, train_loader, 
    output_dir,
    total_steps: gin.REQUIRED,
    pretrain_steps: gin.REQUIRED,
    eval_interval: gin.REQUIRED,
    log_interval: gin.REQUIRED,
    save_interval: gin.REQUIRED,
    log_image_interval: gin.REQUIRED,
    grad_clip_norm: gin.REQUIRED,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=0.0,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1
):   
    if dist.get_rank() == 0:
      logger = ProcessSafeLogger(os.path.join(output_dir, 'train.log')).get_logger()
    if enable_amp:
      scaler = torch.cuda.amp.GradScaler()
      torch.autograd.set_detect_anomaly(False)
    else:
      torch.autograd.set_detect_anomaly(False)

    if lpips_loss_weight > 0:
      lpips_loss_func = loss_utils.lpips_loss_fn()

    train_iterator = iter(train_loader)
    accumulate_step = gin.query_parameter('build_trainloader.accumulate_step')
    if dist.get_rank() == 0:
      logger.info(f'Accumulate step: {accumulate_step}')
    for step in tqdm(range(resume_from_step*accumulate_step, total_steps*accumulate_step), disable=dist.get_rank()!=0):
      step_consider_accum = step//accumulate_step
      try:
        batch = next(train_iterator)
      except StopIteration:
        batch = next(train_iterator)

      batch_gs = gpu_utils.move_to_device([data['gs_params'] for data in batch],model.device)
      batch_cameras = gpu_utils.move_to_device([data['cameras'] for data in batch],model.device)
      batch_images = gpu_utils.move_to_device([data['images'] for data in batch],model.device)
      batch_scene_idx = [data['scene_idx'] for data in batch]
      forward_kwargs = {'batch_normalized_gs': batch_gs, 'batch_scene_idx': batch_scene_idx}

      with torch.cuda.amp.autocast(enabled=enable_amp):
          out_batch_gs = model(**forward_kwargs)

      loss_dict = {}
      metric_dict = {}
      if step_consider_accum < pretrain_steps:
        loss = 0
        for ii, (out_gs, in_gs) in enumerate(zip(out_batch_gs, batch_gs)):
          with torch.no_grad():
            pseudo_target = gs_utils.create_pseudo_target(
              sh_degree=model.module.sh_degree, 
              N=in_gs['means'].shape[0],
              input_gs=in_gs,)

          for key in pseudo_target:
            target = pseudo_target[key].to(model.device)
            pred = out_gs[key]
            value = (pred - target).abs().mean()
            if key == 'features_rest':
              if model.module.sh_degree>0:
                loss += value
            else:
              loss += value
            metric_dict['pretrain/'+key] = value
        loss = loss/len(out_batch_gs)
        loss_dict['pretrain_loss'] = loss/len(out_batch_gs)
        optimizer, scheduler = optimizer_['pretrain'], scheduler_['pretrain']
      else:
          loss_dict['image_l1'], metric_dict['train_psnr'] = 0, 0
          if lpips_loss_weight > 0:
            loss_dict['lpips'] = 0
          num_images = 0
          for out_gs, cameras, images, in_gs in zip(out_batch_gs, batch_cameras, batch_images, batch_gs):
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, cameras) #a List
            for pred_img, gt_img in zip(pred_imgs, images):
                loss_dict['image_l1'] += (pred_img - gt_img).abs().mean()#/len(pred_imgs)
                if lpips_loss_weight > 0:
                  loss_dict['lpips'] += lpips_loss_func(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean()
                metric_dict['train_psnr'] += (psnr(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean())#/len(pred_imgs)
                num_images += 1
          loss_dict['image_l1'] = loss_dict['image_l1']/num_images/len(out_batch_gs)*image_l1_loss_weight
          if lpips_loss_weight > 0:
            loss_dict['lpips'] = loss_dict['lpips']/num_images/len(out_batch_gs)*lpips_loss_weight
          metric_dict['train_psnr'] = metric_dict['train_psnr']/num_images/len(out_batch_gs)

          optimizer, scheduler = optimizer_['train2D'], scheduler_['train2D']
      total_loss = sum(loss_dict.values())/accumulate_step

      if enable_amp:
        scaler.scale(total_loss).backward()
      else:
        total_loss.backward()
      if (step+1) % accumulate_step == 0:
        if grad_clip_norm > 0:
          if enable_amp:
            scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if enable_amp:
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

      if empty_cache_fre > 0 and (step+1) % empty_cache_fre == 0:
        torch.cuda.empty_cache()
    
      if (step_consider_accum % log_interval == 0) and step%accumulate_step==0:
        for key, value in list(loss_dict.items()) + list(metric_dict.items()):
            torch.distributed.reduce(value, dst=0)
            if dist.get_rank() == 0:
              value = (value/torch.cuda.device_count()).item()
              wandb.log({key: value}, step=step_consider_accum)
              if step_consider_accum % (log_interval*10)==0:
                logger.info(f'Training-Step {step_consider_accum}: {key}: {value:.3f}')
              wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=step_consider_accum)
      if step_consider_accum % log_image_interval == 0 and step%accumulate_step==0:
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        with torch.no_grad():
          imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(
            gpu_utils.move_to_device(out_batch_gs[0], device=model.device), batch_cameras[0])
          imgs = [(im*255).cpu().numpy().astype(np.uint8) for im in imgs]
        grid = make_grid(imgs)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'train/{step_consider_accum:08d}_pred-rank{dist.get_rank()}.png'), grid)

      if ((step%accumulate_step==0) and ((step_consider_accum % eval_interval == 0) or (step_consider_accum+1)==pretrain_steps)):
        model.eval()
        for test_dataset, test_loader in build_testloader().items():
            metrics, metrics_input = evaluation(model, test_loader = test_loader,
                                output_dir=output_dir+f'/eval/{test_dataset}/{step_consider_accum}', output_gt=(step_consider_accum==0), compare_with_pseudo=step_consider_accum<pretrain_steps,
                                evaluate_input=(step_consider_accum==0)) #when step==0, we evaluate the input
            if dist.get_rank() == 0:
                wandb.log({f'metrics_testscenes/{test_dataset}/{k}_testviews':v for k,v in metrics.items()}, step=step_consider_accum)
                metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
                logger.info(f'Test {test_dataset} Step {step_consider_accum}: {metric_str}')
            dist.barrier()

      if (step%accumulate_step==0) and ((step_consider_accum+1) % save_interval == 0 or (step_consider_accum+1)==pretrain_steps): 
        if dist.get_rank()==0:
            os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(output_dir, f'checkpoints/model_{step_consider_accum:08d}.pth'))
            logger.info(f'Save model at step {step_consider_accum}')
        dist.barrier()
      model.train()
        
      if step==resume_from_step and dist.get_rank() == 0:
        with open(os.path.join(FLAGS.output_dir, 'config.gin'),'w') as f:
            f.writelines(gin.operative_config_str())
      
    return step


def main(argv):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    gin.bind_parameter('training.output_dir', FLAGS.output_dir)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed()
    # 1. Dataloading (skip train loader for eval-only)
    if not FLAGS.only_eval:
      train_loader = build_trainloader()
    # 2. Build Model
    model = FeaturePredictor()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if model.resume_ckpt is not None:
      model.load_state_dict(torch.load(model.resume_ckpt,map_location='cpu'))
      print(f'Load model from {model.resume_ckpt}')

    if FLAGS.only_eval:
      assert model.resume_ckpt is not None, 'Need to specify the model checkpoint for evaluation'

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    
    if FLAGS.only_eval == False:
      model.train()
      # 3. Optimizer
      optimizer, scheduler = {}, {}
      with gin.config_scope('pretrain'):
        optimizer['pretrain'] = build_optimizer(model.module)
        scheduler['pretrain'] = build_scheduler(optimizer['pretrain'])
      with gin.config_scope('train2D'):
        optimizer['train2D'] = build_optimizer(model.module)
        scheduler['train2D'] = build_scheduler(optimizer['train2D'])

      if rank==0:
        wandb_run = wandb.init(project='3dgs_multiple-scenes', dir=FLAGS.wandb_dir) #resume=?
        if FLAGS.output_dir[-1]=='/':
          FLAGS.output_dir = FLAGS.output_dir[:-1]
        wandb.run.name = '/'.join(FLAGS.output_dir.split('/')[-2:])
      final_step = training(model, optimizer, scheduler, train_loader, output_dir=FLAGS.output_dir)

    model.eval()
    for test_dataset, test_loader in build_testloader().items():
        metrics, metrics_input = evaluation(model, test_loader = test_loader, 
                            output_dir=FLAGS.output_dir+f'/{FLAGS.eval_subdir}/{test_dataset}', 
                            compare_with_input=FLAGS.compare_with_input,
                            save_as_single=True,
                            save_viewer=FLAGS.save_viewer,
                            output_gt=True, compare_with_pseudo=False)
        if dist.get_rank() == 0:
            logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, FLAGS.eval_subdir, 'eval.log')).get_logger()
            metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
            logger.info(f'Test-{test_dataset}: {metric_str}')
            if FLAGS.compare_with_input:
              metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics_input.items()])
              logger.info(f'Input 3DGS: Test-{test_dataset}: {metric_str}')
        dist.barrier()
    
    dist.destroy_process_group()

app.run(main)
