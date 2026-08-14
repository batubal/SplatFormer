import torch
import lpips
from simple_cameras import create_cameras
from utils.gs_utils import rasterize_gaussians_to_multiimgs

class LPIPSTrainer:
    def __init__(self, image_size=512, focal_length=500):
        self.image_size = image_size
        self.focal_length = focal_length
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    def compute_loss(self, gs_params, num_views=8):
        """Compute LPIPS loss using synthetic cameras"""
        # Create cameras
        cameras = create_cameras(gs_params['means'], num_views, self.image_size, self.focal_length)
        
        # Render images
        pred_images, _ = rasterize_gaussians_to_multiimgs(gs_params, cameras)
        
        if not pred_images:
            return torch.tensor(0.0, device='cuda')
        
        # Self-consistency LPIPS loss
        total_loss = 0
        count = 0
        for i in range(len(pred_images)):
            for j in range(i+1, len(pred_images)):
                loss = self.lpips_fn(
                    pred_images[i].unsqueeze(0).permute(0, 3, 1, 2),
                    pred_images[j].unsqueeze(0).permute(0, 3, 1, 2),
                    normalize=True
                )
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device='cuda')

# Usage
trainer = LPIPSTrainer()
loss = trainer.compute_loss(your_gaussian_params)
