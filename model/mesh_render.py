import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, TexturesVertex, FoVOrthographicCameras, look_at_view_transform
)
import os

def render_and_save(vertices, faces, save_path="rendered_image.png"):


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ro = torch.tensor([1.0, -1.0, 1.0]).to(device)
    vertices = vertices.to(device) * ro
    faces = faces.to(device)
    
    
    verts_rgb = torch.full_like(vertices, 0.95)[None]  # (1, N, 3)
    textures = TexturesVertex(verts_features=verts_rgb)


    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    R, T = look_at_view_transform(eye=((-0.5, 0, 3),), at=((0, 0, 0),), up=((0, 1, 0),), device=device)


    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=((1.0, 1.0, 1.0),))


    lights = PointLights(device=device, location=[[0.0, 0.0, 5.0]])


    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )


    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )


    images = renderer(mesh)
    image = images[0, ..., :3].cpu().detach().numpy()


    plt.imsave(save_path, image)

