import os, sys
sys.path.append('../')
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import json
import math

import torch
import plotly.graph_objects as go
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
from scipy.spatial.transform import Rotation as R
import pyvista as pv

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
        z=[origin[2], end[2]],
        mode='lines',
        line=dict(color=color, width=5)
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
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
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



with initialize(config_path='../configs/'):
    config = compose(
        config_name='config',
        overrides=[
            'experiment=Ditto_s2m.yaml',
        ], return_hydra_config=True)
config.datamodule.opt.train.data_dir = '../data/'
config.datamodule.opt.val.data_dir = '../data/'
config.datamodule.opt.test.data_dir = '../data/'


model = hydra.utils.instantiate(config.model)
ckpt = torch.load('../data/Ditto_s2m.ckpt')
device = torch.device(0)
model.load_state_dict(ckpt['state_dict'], strict=True)
model = model.eval().to(device)


generator = Generator3D(
    model.model,
    device=device,
    threshold=0.4,
    seg_threshold=0.5,
    input_type='pointcloud',
    refinement_step=0,
    padding=0.1,
    resolution0=32
)


pc_end = np.load("./pc_start4_j2.npy")

pc_start = np.load("./pc_end4_j2.npy")

#rotation = (R.from_euler('x', np.pi/2)*R.from_euler('y', np.pi/2)).as_matrix()
#rotation = R.from_euler('z', np.pi).as_matrix()

# Apply rotation
#pc_end = pc_end @ rotation.T
#pc_start = pc_start @ rotation.T 
bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
norm_center = (bound_max + bound_min) / 2
norm_scale = (bound_max - bound_min).max() * 1.1
pc_start = (pc_start - norm_center) / norm_scale
pc_end = (pc_end - norm_center) / norm_scale

pc_start, _ = sample_point_cloud(pc_start, 8192)
pc_end, _ = sample_point_cloud(pc_end, 8192)
  
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
ax2 = fig1.add_subplot(1,2,2, projection='3d')
plot_3d_point_cloud(*pc_start.T,
                    axis=ax1,
                    azim=0,
                    elev=90,
                    lim=[(-0.5, 0.5)] * 3)


plot_3d_point_cloud(*pc_end.T,
                    axis=ax2,
                    azim=0,
                    elev=90,
                    lim=[(-0.5, 0.5)] * 3)
plt.show()

sample = {
    'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
    'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
}

mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
with torch.no_grad():
    joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)

print(f"Mobile_points: {mobile_points_all.shape}")

renderer = PyRenderer(light_kwargs={'color': np.array([1., 1., 1.]), 'intensity': 9})


# compute articulation model
mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 20], dtype=np.uint8)
joint_type_prob = joint_type_logits.sigmoid().mean()
if joint_type_prob.item()< 0.5:
    print("Revolute")
    # axis voting
    joint_r_axis = (
        normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
    )
    joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
    joint_r_p2l_vec = (
        normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
    )
    joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
    p_seg = mobile_points_all[0].cpu().numpy()

    pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
    (
        joint_axis_pred,
        pivot_point_pred,
        config_pred,
    ) = aggregate_dense_prediction_r(
        joint_r_axis, pivot_point, joint_r_t, method="mean"
    )
    print(joint_axis_pred,pivot_point_pred,config_pred)
# prismatic
else:
    print("Prismatic")
    # axis voting
    joint_p_axis = (
        normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
    )
    joint_axis_pred = joint_p_axis.mean(0)
    joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
    config_pred = joint_p_t.mean()
    
    pivot_point_pred = mesh_dict[1].bounds.mean(0)
"""
"""
scene = trimesh.Scene()
static_part = mesh_dict[0].copy()
mobile_part = mesh_dict[1].copy()
scene.add_geometry(static_part)
scene.add_geometry(mobile_part)
#joint_axis_pred=np.array([0.03,0.95,0.03])
add_r_joint_to_scene(scene, joint_axis_pred, pivot_point_pred, 1.0, recenter=True)
scene.export('scene0.glb')
#scene.show()

pointcloud = np.random.rand(100, 3) * 0.2 + 0.5  # Adjust to fit the scene
# Add joint axis (if applicable)
joint_axis_line = vector_to_plotly_line(pivot_point_pred, joint_axis_pred, length=0.5)

# Convert all parts to plotly traces
fig = go.Figure(data=[
    trimesh_to_plotly(static_part, color='gray'),
    trimesh_to_plotly(mobile_part, color='lightgreen'),
    pointcloud_to_plotly(np.asarray(mobile_points_all.cpu()).squeeze()+1, color='blue'),
    joint_axis_line
])

# Layout settings
fig.update_layout(
scene=dict(
    xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=True),
    yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=True),
    zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=True),
    aspectmode='data'
),
margin=dict(l=0, r=0, t=0, b=0),
showlegend=False,
paper_bgcolor='rgba(250,250,250,250)',  # transparent background
plot_bgcolor='rgba(0,0,0,0)',
title="3D Scene with Meshes, Point Cloud, and Joint Axis"
)
fig.show()


camera_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
light_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

img = Image.fromarray(rgb)
img.show()


