import numpy as np
import json
import math
import os, sys
sys.path.append('../')
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch

import open3d as o3d
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from hydra.experimental import initialize_config_module, initialize_config_dir
from hydra import initialize, compose
from omegaconf import OmegaConf
import hydra

from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils.misc import sample_point_cloud

import trimesh
from src.utils.joint_estimation import aggregate_dense_prediction_r

from utils3d.mesh.utils import as_mesh
from utils3d.render.pyrender import get_pose, PyRenderer
import plotly.graph_objects as go


def trimesh_to_plotly(mesh, color='lightblue'):
    vertices = mesh.vertices
    faces = mesh.faces

    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        opacity=1.0,
        flatshading=True
    )

def vector_to_plotly_line(origin, direction, length=1.0, color='red'):
    end = origin + direction * length
    return go.Scatter3d(
        x=[origin[0], end[0]],
        y=[origin[1], end[1]],
        z=[origin[2]-0.25, end[2]],
        mode='lines',
        line=dict(color=color, width=10)
    )
def vector_to_plotly_line1(origin, direction, length=1.0, color='red'):
    end = origin + direction * length
    return go.Scatter3d(
        x=[origin[0], end[0]],
        y=[origin[1]-0.25, end[1]],
        z=[origin[2], end[2]],
        mode='lines',
        line=dict(color=color, width=10)
    )

def pointcloud_to_plotly(points, color='black'):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=color)
    )


def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=False,
                        show_axis=True,
                        in_u_sphere=False,
                        marker='.',
                        color=None,
                        s=8,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        lim=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        ax.set_title(title)

    sc = ax.scatter(x, y, z, marker=marker,c=color, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)
    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (min(np.min(x), np.min(y),
                   np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)


def vector_to_rotation(vector):
    z = np.array(vector)
    z = z / np.linalg.norm(z)
    x = np.array([1, 0, 0])
    x = x - z*(x.dot(z)/z.dot(z))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]

def add_r_joint_to_scene(scene,
                             axis,
                             pivot_point,
                             length,
                             radius=0.01,
                             joint_color=[200, 0, 0, 180],
                             recenter=False):
    if recenter:
        pivot_point = np.cross(axis, np.cross(pivot_point, axis))
    rotation_mat = vector_to_rotation(axis)
    screw_tran = np.eye(4)
    screw_tran[:3, :3] = rotation_mat
    screw_tran[:3, 3] = pivot_point
    
    axis_cylinder = trimesh.creation.cylinder(radius, height=length)
    axis_arrow = trimesh.creation.cone(radius * 2, radius * 4)
    arrow_trans = np.eye(4)
    arrow_trans[2, 3] = length / 2
    axis_arrow.apply_transform(arrow_trans)
    axis_obj = trimesh.Scene((axis_cylinder, axis_arrow))
    screw = as_mesh(axis_obj)
    
    # screw.apply_translation([0, 0, 0.1])
    screw.apply_transform(screw_tran)
    screw.visual.face_colors = np.array(joint_color, dtype=np.uint8)
    scene.add_geometry(screw)
    return screw


def subtract_pc(pc_subt_from,pc_subt,threshold):

    pc_subt_from_o3d = o3d.geometry.PointCloud()
    pc_subt_from_o3d.points = o3d.utility.Vector3dVector(pc_subt_from)

    pc_subt_o3d = o3d.geometry.PointCloud()
    pc_subt_o3d.points = o3d.utility.Vector3dVector(pc_subt)

    distances = pc_subt_from_o3d.compute_point_cloud_distance(pc_subt_o3d)
    distances = np.asarray(distances)


    indices = np.where(distances > threshold)[0]

    result = pc_subt_from_o3d.select_by_index(indices)
    num_subtracted = len(pc_subt_from) - len(indices)
    print(f"Points subtracted: {num_subtracted}")
    return result

def get_lower_left_corner_points(mesh, threshold=0.3):
    # Get bounding box
    bounds = mesh.bounds  # shape: (2, 3), [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    print(bounds)
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]

    # Compute thresholds to define "lower-left" corner
    x_thresh = min_x + (max_x - min_x) * threshold
    y_thresh = min_y + (max_y - min_y) * threshold
    print(x_thresh)
    print(y_thresh)
    # Filter points in the lower-left region (and optionally lowest z for bottom)
    mask = (
        (mesh.vertices[:, 0] <= x_thresh) &
        (mesh.vertices[:, 1] <= y_thresh)
    )

    corner_points = mesh.vertices[mask]
    return corner_points

def get_higher_left_corner_points(mesh, threshold=0.3):
    # Get bounding box
    bounds = mesh.bounds  # shape: (2, 3), [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    print(bounds)
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]

    # Compute thresholds to define "lower-left" corner
    x_thresh = max_x - (max_x - min_x) * threshold
    y_thresh = min_y + (max_y - min_y) * threshold
    print(x_thresh)
    print(y_thresh)
    # Filter points in the lower-left region (and optionally lowest z for bottom)
    mask = (
        (mesh.vertices[:, 0] >= x_thresh) &
        (mesh.vertices[:, 1] <= y_thresh)
    )

    corner_points = mesh.vertices[mask]
    return corner_points

def read_path(num_joint,num_plane):
    folder_path = f"{num_joint+1}L_{num_joint}J_{num_plane}P"
    pcs_start = []
    pcs_end = []
    for i in range(num_joint):
        pcs_start.append(np.load(os.path.join('./',folder_path,f'pc_start_j{i}.npy')))
        pcs_end.append(np.load(os.path.join('./',folder_path,f'pc_end_j{i}.npy')))
    return pcs_start, pcs_end

 
def find_origin(pc):
    minimum_x = pc.min(0)[0]
    points_min_x  = pc[pc[:,0]==minimum_x]
    minimum_y = points_min_x.min(0)[1]
    points_min_y_min_x = points_min_x[points_min_x[:,1]==minimum_y]
    return points_min_y_min_x
   

def get_color(seed_offset=0):
    np.random.seed(0 + seed_offset)  # Offset the seed for uniqueness
    return np.concatenate((np.random.randint(50, 255, size=3, dtype=np.uint8), [255]))


#   if i!=0:
#         pcs_to_subtract =[]
#         for j in range(i):
#             pcs_to_subtract.append(static_pcs[j]) 
#             pcs_to_subtract[j]+=(norm_center_oif[j]-norm_center_oif[i])/norm_scale_oif[i]
#             fig2 = plt.figure()
#             fig2.suptitle("Input Normalized with Static Part to Subtract", fontsize=14)
#             ax1 = fig2.add_subplot(1,1,1, projection='3d')

#             plot_3d_point_cloud(*pc_start.T,
#                                 axis=ax1,
#                                 azim=180,
#                                 elev=90,
#                                 lim=[(-0.1, 0.8)] * 3)


#             plot_3d_point_cloud(*pc_end.T,
#                                 axis=ax1,
#                                 azim=180,
#                                 elev=90,
#                                 lim=[(-0.5, 0.5)] * 3)
            

#             plot_3d_point_cloud(*pcs_to_subtract[j].T,
#                         axis=ax1,
#                         azim=180,
#                         elev=90,
#                         lim=[(-0.5, 0.5)] * 3)
            
#             plt.show()