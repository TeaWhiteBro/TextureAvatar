import torch
from plyfile import PlyData, PlyElement
import numpy as np

def save_point_cloud_to_ply(points, save_path="point_cloud.ply"):


    points_np = points.squeeze(0).cpu().detach().numpy()
    

    vertices = np.array(
        [(point[0], point[1], point[2]) for point in points_np],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    

    ply_element = PlyElement.describe(vertices, 'vertex')
    

    PlyData([ply_element]).write(save_path)
    print(f"Point cloud saved to {save_path}")