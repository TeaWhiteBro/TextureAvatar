import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointsRasterizationSettings, PointsRenderer,
    PointsRasterizer, AlphaCompositor, look_at_view_transform
)
from pytorch3d.renderer.lighting import PointLights

def render_point_cloud(points, save_path="rendered_point_cloud.png"):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = points.to(device)
    

    point_cloud = PointClouds(points=points, features=torch.ones_like(points) * 0.5)
    

    R, T = look_at_view_transform(eye=((0, 0, 3),), at=((0, 0, 0),), up=((0, 1, 0),), device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    

    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.01,
        points_per_pixel=10
    )
    

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    

    images = renderer(point_cloud, lights=lights)
    image = images[0, ..., :3].cpu().numpy()
    

    plt.imsave(save_path, image)
    print(f"Point cloud image saved to {save_path}")


points = torch.rand((1, 202738, 3))
render_point_cloud(points, "output_point_cloud.png")
