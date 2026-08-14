import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class SyntheticCameraGenerator:
    """
    Generate synthetic cameras for rendering Gaussian splats without real camera data.
    Supports multiple camera placement strategies and automatic parameter estimation.
    """
    
    def __init__(self, 
                 image_size: int = 512,
                 default_focal_length: Optional[float] = None,
                 default_fov: float = 50.0):
        """
        Args:
            image_size: Size of rendered images (assumes square)
            default_focal_length: Fixed focal length in pixels (if None, calculated from FOV)
            default_fov: Field of view in degrees (used if focal_length is None)
        """
        self.image_size = image_size
        self.default_fov = default_fov
        
        if default_focal_length is None:
            self.default_focal_length = self._fov_to_focal(default_fov, image_size)
        else:
            self.default_focal_length = default_focal_length
            
        print(f"Camera generator initialized:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Focal length: {self.default_focal_length:.1f} pixels")
        print(f"  Field of view: {default_fov:.1f} degrees")
    
    def _fov_to_focal(self, fov_degrees: float, image_size: int) -> float:
        """Convert field of view to focal length"""
        fov_radians = math.radians(fov_degrees)
        focal_length = image_size / (2 * math.tan(fov_radians / 2))
        return focal_length
    
    def _focal_to_fov(self, focal_length: float, image_size: int) -> float:
        """Convert focal length to field of view"""
        fov_radians = 2 * math.atan(image_size / (2 * focal_length))
        return math.degrees(fov_radians)
    
    def estimate_camera_distance(self, gaussian_means: torch.Tensor, 
                               margin_factor: float = 2.0) -> float:
        """
        Estimate appropriate camera distance based on Gaussian bounds
        
        Args:
            gaussian_means: [N, 3] positions of Gaussian centers
            margin_factor: Multiplier for bounding box diagonal (higher = farther cameras)
        """
        min_bounds = gaussian_means.min(dim=0)[0]
        max_bounds = gaussian_means.max(dim=0)[0]
        diagonal = torch.norm(max_bounds - min_bounds)
        return diagonal.item() * margin_factor
    
    def create_sphere_cameras(self, 
                            gaussian_means: torch.Tensor,
                            num_views: int = 8,
                            elevation_range: Tuple[float, float] = (15, 75),
                            margin_factor: float = 2.0,
                            look_at_center: bool = True) -> Dict[str, torch.Tensor]:
        """
        Create cameras positioned on a sphere around the Gaussian data
        
        Args:
            gaussian_means: [N, 3] positions of Gaussian centers
            num_views: Number of camera views to generate
            elevation_range: (min_elevation, max_elevation) in degrees
            margin_factor: Distance multiplier from object bounds
            look_at_center: If True, cameras look at object center; if False, look at origin
        """
        # Calculate object center and radius
        min_bounds = gaussian_means.min(dim=0)[0]
        max_bounds = gaussian_means.max(dim=0)[0]
        center = (min_bounds + max_bounds) / 2
        radius = torch.norm(max_bounds - min_bounds) / 2
        
        # Calculate camera distance
        camera_distance = self.estimate_camera_distance(gaussian_means, margin_factor)
        
        cameras = {
            'camera_to_worlds': [],
            'fx': torch.tensor(self.default_focal_length),
            'fy': torch.tensor(self.default_focal_length),
            'cx': torch.tensor(self.image_size / 2),
            'cy': torch.tensor(self.image_size / 2),
            'width': torch.tensor(self.image_size),
            'height': torch.tensor(self.image_size),
            'background_color': torch.tensor([0.0, 0.0, 0.0])
        }
        
        # Generate camera positions
        for i in range(num_views):
            # Azimuth: uniform distribution
            azimuth = 2 * math.pi * i / num_views
            
            # Elevation: random within range
            min_elev, max_elev = elevation_range
            elevation = math.radians(np.random.uniform(min_elev, max_elev))
            
            # Convert spherical to Cartesian coordinates
            x = center[0] + camera_distance * math.cos(azimuth) * math.sin(elevation)
            y = center[1] + camera_distance * math.sin(azimuth) * math.sin(elevation)
            z = center[2] + camera_distance * math.cos(elevation)
            
            camera_pos = torch.tensor([x, y, z])
            
            # Determine look-at target
            if look_at_center:
                look_at_target = center
            else:
                look_at_target = torch.zeros(3)
            
            # Create camera-to-world matrix
            c2w = self._create_look_at_matrix(camera_pos, look_at_target)
            cameras['camera_to_worlds'].append(c2w)
        
        cameras['camera_to_worlds'] = torch.stack(cameras['camera_to_worlds'])
        return cameras
    
    def create_circle_cameras(self,
                            gaussian_means: torch.Tensor,
                            num_views: int = 8,
                            height_offset: float = 0.0,
                            radius_factor: float = 2.0,
                            look_at_center: bool = True) -> Dict[str, torch.Tensor]:
        """
        Create cameras positioned on a circle around the Gaussian data
        
        Args:
            gaussian_means: [N, 3] positions of Gaussian centers
            num_views: Number of camera views to generate
            height_offset: Height offset from object center (positive = above)
            radius_factor: Radius multiplier from object bounds
            look_at_center: If True, cameras look at object center
        """
        # Calculate object center and radius
        min_bounds = gaussian_means.min(dim=0)[0]
        max_bounds = gaussian_means.max(dim=0)[0]
        center = (min_bounds + max_bounds) / 2
        radius = torch.norm(max_bounds - min_bounds) / 2
        
        camera_radius = radius * radius_factor
        
        cameras = {
            'camera_to_worlds': [],
            'fx': torch.tensor(self.default_focal_length),
            'fy': torch.tensor(self.default_focal_length),
            'cx': torch.tensor(self.image_size / 2),
            'cy': torch.tensor(self.image_size / 2),
            'width': torch.tensor(self.image_size),
            'height': torch.tensor(self.image_size),
            'background_color': torch.tensor([0.0, 0.0, 0.0])
        }
        
        # Generate camera positions on circle
        for i in range(num_views):
            angle = 2 * math.pi * i / num_views
            
            x = center[0] + camera_radius * math.cos(angle)
            y = center[1] + camera_radius * math.sin(angle)
            z = center[2] + height_offset
            
            camera_pos = torch.tensor([x, y, z])
            
            # Determine look-at target
            if look_at_center:
                look_at_target = center
            else:
                look_at_target = torch.zeros(3)
            
            # Create camera-to-world matrix
            c2w = self._create_look_at_matrix(camera_pos, look_at_target)
            cameras['camera_to_worlds'].append(c2w)
        
        cameras['camera_to_worlds'] = torch.stack(cameras['camera_to_worlds'])
        return cameras
    
    def create_random_cameras(self,
                            gaussian_means: torch.Tensor,
                            num_views: int = 8,
                            min_distance_factor: float = 1.5,
                            max_distance_factor: float = 3.0,
                            look_at_center: bool = True) -> Dict[str, torch.Tensor]:
        """
        Create cameras at random positions around the Gaussian data
        
        Args:
            gaussian_means: [N, 3] positions of Gaussian centers
            num_views: Number of camera views to generate
            min_distance_factor: Minimum distance multiplier from object bounds
            max_distance_factor: Maximum distance multiplier from object bounds
            look_at_center: If True, cameras look at object center
        """
        # Calculate object center and radius
        min_bounds = gaussian_means.min(dim=0)[0]
        max_bounds = gaussian_means.max(dim=0)[0]
        center = (min_bounds + max_bounds) / 2
        radius = torch.norm(max_bounds - min_bounds) / 2
        
        cameras = {
            'camera_to_worlds': [],
            'fx': torch.tensor(self.default_focal_length),
            'fy': torch.tensor(self.default_focal_length),
            'cx': torch.tensor(self.image_size / 2),
            'cy': torch.tensor(self.image_size / 2),
            'width': torch.tensor(self.image_size),
            'height': torch.tensor(self.image_size),
            'background_color': torch.tensor([0.0, 0.0, 0.0])
        }
        
        # Generate random camera positions
        for i in range(num_views):
            # Random distance
            distance = radius * np.random.uniform(min_distance_factor, max_distance_factor)
            
            # Random direction (uniform on sphere)
            azimuth = np.random.uniform(0, 2 * math.pi)
            elevation = np.random.uniform(0, math.pi)
            
            x = center[0] + distance * math.cos(azimuth) * math.sin(elevation)
            y = center[1] + distance * math.sin(azimuth) * math.sin(elevation)
            z = center[2] + distance * math.cos(elevation)
            
            camera_pos = torch.tensor([x, y, z])
            
            # Determine look-at target
            if look_at_center:
                look_at_target = center
            else:
                look_at_target = torch.zeros(3)
            
            # Create camera-to-world matrix
            c2w = self._create_look_at_matrix(camera_pos, look_at_target)
            cameras['camera_to_worlds'].append(c2w)
        
        cameras['camera_to_worlds'] = torch.stack(cameras['camera_to_worlds'])
        return cameras
    
    def create_fixed_cameras(self,
                           positions: List[List[float]],
                           look_at_target: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Create cameras at fixed positions
        
        Args:
            positions: List of [x, y, z] camera positions
            look_at_target: [x, y, z] target to look at (if None, looks at origin)
        """
        if look_at_target is None:
            look_at_target = [0.0, 0.0, 0.0]
        
        look_at_tensor = torch.tensor(look_at_target)
        
        cameras = {
            'camera_to_worlds': [],
            'fx': torch.tensor(self.default_focal_length),
            'fy': torch.tensor(self.default_focal_length),
            'cx': torch.tensor(self.image_size / 2),
            'cy': torch.tensor(self.image_size / 2),
            'width': torch.tensor(self.image_size),
            'height': torch.tensor(self.image_size),
            'background_color': torch.tensor([0.0, 0.0, 0.0])
        }
        
        for pos in positions:
            camera_pos = torch.tensor(pos)
            c2w = self._create_look_at_matrix(camera_pos, look_at_tensor)
            cameras['camera_to_worlds'].append(c2w)
        
        cameras['camera_to_worlds'] = torch.stack(cameras['camera_to_worlds'])
        return cameras
    
    def _create_look_at_matrix(self, camera_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Create a camera-to-world matrix using look-at convention
        
        Args:
            camera_pos: [3] camera position
            target: [3] target position to look at
            
        Returns:
            [4, 4] camera-to-world transformation matrix
        """
        # Forward direction (camera looks towards target)
        forward = target - camera_pos
        forward = forward / torch.norm(forward)
        
        # Up direction (assume Z-up)
        up = torch.tensor([0.0, 0.0, 1.0])
        
        # Right direction
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        # Recalculate up to ensure orthogonality
        up = torch.cross(right, forward)
        
        # Create camera-to-world matrix
        c2w = torch.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = camera_pos
        
        return c2w
    
    def adjust_focal_length(self, cameras: Dict[str, torch.Tensor], 
                           new_focal_length: float) -> Dict[str, torch.Tensor]:
        """Adjust focal length of existing cameras"""
        cameras = cameras.copy()
        cameras['fx'] = torch.tensor(new_focal_length)
        cameras['fy'] = torch.tensor(new_focal_length)
        return cameras
    
    def get_camera_info(self, cameras: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get information about the cameras"""
        fov = self._focal_to_fov(cameras['fx'].item(), cameras['width'].item())
        return {
            'focal_length': cameras['fx'].item(),
            'field_of_view': fov,
            'image_size': cameras['width'].item(),
            'num_cameras': len(cameras['camera_to_worlds'])
        }


# Convenience functions for quick usage
def create_sphere_cameras(gaussian_means: torch.Tensor, 
                         num_views: int = 8,
                         image_size: int = 512,
                         focal_length: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Quick function to create sphere cameras"""
    generator = SyntheticCameraGenerator(image_size=image_size, default_focal_length=focal_length)
    return generator.create_sphere_cameras(gaussian_means, num_views)

def create_circle_cameras(gaussian_means: torch.Tensor,
                         num_views: int = 8,
                         image_size: int = 512,
                         focal_length: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Quick function to create circle cameras"""
    generator = SyntheticCameraGenerator(image_size=image_size, default_focal_length=focal_length)
    return generator.create_circle_cameras(gaussian_means, num_views)

def create_random_cameras(gaussian_means: torch.Tensor,
                         num_views: int = 8,
                         image_size: int = 512,
                         focal_length: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Quick function to create random cameras"""
    generator = SyntheticCameraGenerator(image_size=image_size, default_focal_length=focal_length)
    return generator.create_random_cameras(gaussian_means, num_views)


# Example usage and testing
if __name__ == "__main__":
    # Create some dummy Gaussian data
    gaussian_means = torch.randn(1000, 3) * 2  # Random points in a 2-unit cube
    
    # Initialize camera generator
    camera_gen = SyntheticCameraGenerator(image_size=512, default_fov=50.0)
    
    # Test different camera generation methods
    print("\n=== Testing Camera Generation ===")
    
    # Sphere cameras
    sphere_cameras = camera_gen.create_sphere_cameras(gaussian_means, num_views=8)
    print(f"Sphere cameras: {camera_gen.get_camera_info(sphere_cameras)}")
    
    # Circle cameras
    circle_cameras = camera_gen.create_circle_cameras(gaussian_means, num_views=8)
    print(f"Circle cameras: {camera_gen.get_camera_info(circle_cameras)}")
    
    # Random cameras
    random_cameras = camera_gen.create_random_cameras(gaussian_means, num_views=8)
    print(f"Random cameras: {camera_gen.get_camera_info(random_cameras)}")
    
    # Fixed cameras
    fixed_positions = [
        [2, 0, 1],
        [-2, 0, 1],
        [0, 2, 1],
        [0, -2, 1]
    ]
    fixed_cameras = camera_gen.create_fixed_cameras(fixed_positions)
    print(f"Fixed cameras: {camera_gen.get_camera_info(fixed_cameras)}")
    
    print("\n=== Camera Generation Complete ===")
