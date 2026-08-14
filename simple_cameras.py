import torch
import math

def create_cameras(gaussian_means, num_views=8, image_size=512, focal_length=500):
    """Create cameras on a sphere around Gaussian data"""
    
    # Calculate object bounds
    min_bounds = gaussian_means.min(dim=0)[0]
    max_bounds = gaussian_means.max(dim=0)[0]
    center = (min_bounds + max_bounds) / 2
    radius = torch.norm(max_bounds - min_bounds) / 2
    
    # Camera distance
    camera_distance = radius * 2.0
    
    cameras = {
        'camera_to_worlds': [],
        'fx': torch.tensor(focal_length),
        'fy': torch.tensor(focal_length),
        'cx': torch.tensor(image_size / 2),
        'cy': torch.tensor(image_size / 2),
        'width': torch.tensor(image_size),
        'height': torch.tensor(image_size),
        'background_color': torch.tensor([0.0, 0.0, 0.0])
    }
    
    # Generate cameras on sphere
    for i in range(num_views):
        azimuth = 2 * math.pi * i / num_views
        elevation = math.pi / 4  # 45 degrees
        
        x = center[0] + camera_distance * math.cos(azimuth) * math.sin(elevation)
        y = center[1] + camera_distance * math.sin(azimuth) * math.sin(elevation)
        z = center[2] + camera_distance * math.cos(elevation)
        
        camera_pos = torch.tensor([x, y, z])
        
        # Look at center
        forward = center - camera_pos
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0, 0, 1])
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # Camera-to-world matrix
        c2w = torch.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = camera_pos
        
        cameras['camera_to_worlds'].append(c2w)
    
    cameras['camera_to_worlds'] = torch.stack(cameras['camera_to_worlds'])
    return cameras