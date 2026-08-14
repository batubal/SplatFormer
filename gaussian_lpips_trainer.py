import torch
import lpips
from synthetic_cameras import SyntheticCameraGenerator, create_sphere_cameras
from utils.gs_utils import rasterize_gaussians_to_multiimgs

class GaussianLPIPSTrainer:
    """
    Complete training setup for Gaussian splats with LPIPS loss using synthetic cameras
    """
    
    def __init__(self, image_size=512, focal_length=None):
        self.image_size = image_size
        self.camera_gen = SyntheticCameraGenerator(
            image_size=image_size, 
            default_focal_length=focal_length
        )
        
        # Initialize LPIPS loss
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
        
        print(f"LPIPS trainer initialized with {image_size}x{image_size} images")
    
    def render_gaussians(self, gs_params, cameras):
        """Render Gaussians to images using the provided cameras"""
        try:
            pred_images, _ = rasterize_gaussians_to_multiimgs(gs_params, cameras)
            return pred_images
        except Exception as e:
            print(f"Rendering error: {e}")
            return []
    
    def compute_lpips_loss(self, pred_images, gt_images=None, loss_type='self_consistency'):
        """
        Compute LPIPS loss
        
        Args:
            pred_images: List of rendered images
            gt_images: List of ground truth images (optional)
            loss_type: 'self_consistency' or 'supervised'
        """
        if not pred_images:
            return torch.tensor(0.0, device='cuda')
        
        if loss_type == 'self_consistency' or gt_images is None:
            # Self-consistency: compare different views
            total_loss = 0
            count = 0
            for i in range(len(pred_images)):
                for j in range(i+1, len(pred_images)):
                    # Ensure images are in [0,1] range and correct format
                    img1 = torch.clamp(pred_images[i], 0, 1)
                    img2 = torch.clamp(pred_images[j], 0, 1)
                    
                    loss = self.lpips_fn(
                        img1.unsqueeze(0).permute(0, 3, 1, 2),
                        img2.unsqueeze(0).permute(0, 3, 1, 2),
                        normalize=True
                    )
                    total_loss += loss
                    count += 1
            return total_loss / count if count > 0 else torch.tensor(0.0, device='cuda')
        
        else:
            # Supervised: compare with ground truth
            total_loss = 0
            for pred_img, gt_img in zip(pred_images, gt_images):
                pred_img = torch.clamp(pred_img, 0, 1)
                gt_img = torch.clamp(gt_img, 0, 1)
                
                loss = self.lpips_fn(
                    pred_img.unsqueeze(0).permute(0, 3, 1, 2),
                    gt_img.unsqueeze(0).permute(0, 3, 1, 2),
                    normalize=True
                )
                total_loss += loss
            return total_loss / len(pred_images)
    
    def training_step(self, gs_params, camera_type='sphere', num_views=8, **camera_kwargs):
        """
        Single training step with LPIPS loss
        
        Args:
            gs_params: Gaussian splat parameters
            camera_type: 'sphere', 'circle', 'random', or 'fixed'
            num_views: Number of camera views
            **camera_kwargs: Additional camera generation parameters
        """
        # Generate cameras based on type
        if camera_type == 'sphere':
            cameras = self.camera_gen.create_sphere_cameras(
                gs_params['means'], num_views=num_views, **camera_kwargs
            )
        elif camera_type == 'circle':
            cameras = self.camera_gen.create_circle_cameras(
                gs_params['means'], num_views=num_views, **camera_kwargs
            )
        elif camera_type == 'random':
            cameras = self.camera_gen.create_random_cameras(
                gs_params['means'], num_views=num_views, **camera_kwargs
            )
        elif camera_type == 'fixed':
            cameras = self.camera_gen.create_fixed_cameras(**camera_kwargs)
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        
        # Render images
        pred_images = self.render_gaussians(gs_params, cameras)
        
        if not pred_images:
            return torch.tensor(0.0, device='cuda'), []
        
        # Compute LPIPS loss
        lpips_loss = self.compute_lpips_loss(pred_images, loss_type='self_consistency')
        
        return lpips_loss, pred_images, cameras
    
    def evaluate(self, gs_params, gt_images, camera_type='sphere', num_views=8):
        """
        Evaluate with ground truth images
        """
        # Generate cameras
        if camera_type == 'sphere':
            cameras = self.camera_gen.create_sphere_cameras(gs_params['means'], num_views)
        elif camera_type == 'circle':
            cameras = self.camera_gen.create_circle_cameras(gs_params['means'], num_views)
        else:
            cameras = self.camera_gen.create_random_cameras(gs_params['means'], num_views)
        
        # Render images
        pred_images = self.render_gaussians(gs_params, cameras)
        
        if not pred_images:
            return torch.tensor(float('inf'), device='cuda'), []
        
        # Compute supervised LPIPS loss
        lpips_loss = self.compute_lpips_loss(pred_images, gt_images, loss_type='supervised')
        
        return lpips_loss, pred_images


# Example usage
def example_usage():
    """Example of how to use the Gaussian LPIPS trainer"""
    
    # Create dummy Gaussian data
    gaussian_params = {
        'means': torch.randn(1000, 3).cuda() * 2,  # Random positions
        'scales': torch.randn(1000, 3).cuda() * 0.1,  # Random scales
        'quats': torch.randn(1000, 4).cuda(),  # Random rotations
        'features_dc': torch.randn(1000, 3).cuda(),  # Random colors
        'opacities': torch.randn(1000, 1).cuda()  # Random opacities
    }
    
    # Normalize quaternions
    gaussian_params['quats'] = gaussian_params['quats'] / torch.norm(
        gaussian_params['quats'], dim=-1, keepdim=True
    )
    
    # Initialize trainer
    trainer = GaussianLPIPSTrainer(image_size=512, focal_length=500)
    
    print("=== Training Step Example ===")
    
    # Training step with sphere cameras
    loss, rendered_images, cameras = trainer.training_step(
        gaussian_params, 
        camera_type='sphere', 
        num_views=8,
        elevation_range=(20, 70)
    )
    
    print(f"LPIPS Loss: {loss.item():.4f}")
    print(f"Number of rendered images: {len(rendered_images)}")
    print(f"Camera info: {trainer.camera_gen.get_camera_info(cameras)}")
    
    # Training step with circle cameras
    loss2, rendered_images2, cameras2 = trainer.training_step(
        gaussian_params,
        camera_type='circle',
        num_views=6,
        height_offset=1.0
    )
    
    print(f"Circle cameras LPIPS Loss: {loss2.item():.4f}")
    
    # Training step with random cameras
    loss3, rendered_images3, cameras3 = trainer.training_step(
        gaussian_params,
        camera_type='random',
        num_views=8,
        min_distance_factor=1.5,
        max_distance_factor=3.0
    )
    
    print(f"Random cameras LPIPS Loss: {loss3.item():.4f}")
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    example_usage()
