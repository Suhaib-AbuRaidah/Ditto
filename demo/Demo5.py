import os, sys
sys.path.append('../')
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import json
import math

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

def pointcloud_to_plotly(points, color='black'):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=color)
    )


def trimesh_to_open3d(tri_mesh):
    vertices = np.asarray(tri_mesh.vertices)
    faces = np.asarray(tri_mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    if tri_mesh.visual.kind == 'face':
        face_colors = np.asarray(tri_mesh.visual.face_colors)[:, :3] / 255.0
        # Assign vertex colors by averaging face colors
        vert_colors = np.zeros((len(vertices), 3))
        count = np.zeros(len(vertices))
        for face, color in zip(faces, face_colors):
            for idx in face:
                vert_colors[idx] += color
                count[idx] += 1
        vert_colors[count > 0] /= count[count > 0][:, None]
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    return o3d_mesh


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

pc_start = np.load("./pc_start_j0.npy")
pc_end0 = np.load("./pc_end4_j0.npy")
pc_end1 = np.load("./pc_end4_j1.npy")
pc_end2 = np.load("./pc_end4_j2.npy")
pc_end_list = [pc_end0,pc_end1,pc_end2]

joint_axis_pred_list=[]
pivot_point_pred_list=[]
mesh_list = []

for i in [3,4,5]:
    print(i)
    if i<3:
        pc_start =np.load("./pc_start_j0.npy")
        pc_end=pc_end_list[i]
    else:
        pc_end = np.load("./pc_start_j0.npy")
        pc_start = pc_end_list[i-3]
    bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
    bound_min = np.minimum(pc_start[i].min(0), pc_end.min(0))
    norm_center = (bound_max + bound_min) / 2
    norm_scale = (bound_max - bound_min).max() * 1.1
    pc_start = (pc_start - norm_center) / norm_scale
    pc_end = (pc_end - norm_center) / norm_scale

    pc_start, _ = sample_point_cloud(pc_start, 12192)
    pc_end, _ = sample_point_cloud(pc_end, 12192)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig1.add_subplot(1,2,2, projection='3d')
    plot_3d_point_cloud(*pc_start.T,
                        axis=ax1,
                        azim=-45,
                        elev=45,
                        lim=[(-0.5, 0.5)] * 3)


    plot_3d_point_cloud(*pc_end.T,
                        axis=ax2,
                        azim=-30,
                        elev=-30,
                        lim=[(-0.5, 0.5)] * 3)
    #plt.show()

    sample = {
        'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
        'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
    }

    mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
    
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)



    renderer = PyRenderer(light_kwargs={'color': np.array([1., 1., 1.]), 'intensity': 9})


    # compute articulation model
    mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 0], dtype=np.uint8)
    mesh_list.append(mesh_dict)
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
        joint_axis_pred_list.append(joint_axis_pred)
        pivot_point_pred_list.append(pivot_point_pred)
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

scene = trimesh.Scene()
for i, mesh_dict in enumerate(mesh_list):
    if i==1 or i==2:
        continue
    mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 0], dtype=np.uint8)
    material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=[c / 255.0 for c in [84, 220, 83, 0]],  # Normalize to 0-1
    alphaMode="BLEND"
)
    mesh_dict[1].visual.material = material
    static_part = mesh_dict[0].copy()
    mobile_part = mesh_dict[1].copy()

    scene.add_geometry(static_part, node_name=f"static_part_{i}")
    scene.add_geometry(mobile_part, node_name=f"mobile_part_{i}")

    # Optional: add corresponding joint visuals for this pose
    if i==0:
        color =[250,0,0,250]
    elif i==1:
        color = [0,250,0,250]
    else:
        color = [0,0,250,250]
    add_r_joint_to_scene(
        scene,
        joint_axis_pred_list[i],      # Adjust indexing if needed
        pivot_point_pred_list[i],
        1.0,
        recenter=True,
        joint_color=color
    )

scene.export('scene1.glb')


fig = go.Figure()

for i, mesh_dict in enumerate(mesh_list):
    static_part = mesh_dict[0].copy()
    mobile_part = mesh_dict[1].copy()
    if i ==0:
        static_part.apply_translation([0, 0.21, 0])
        mobile_part.apply_translation([0, 0.21, 0])
    
    # Optional: differentiate colors per object
    static_color = 'rgb(150, 150, 150,1)'
    if i==0:
        mobile_color = 'lightgreen'
        axis_color = 'green'
        joint_axis_line = vector_to_plotly_line(pivot_point_pred_list[i]+np.array([0,0.21,0]), joint_axis_pred_list[i], length=0.5,color=axis_color)
    elif i==1:
        mobile_color = 'lightpink'
        axis_color= 'red'
        joint_axis_line = vector_to_plotly_line(pivot_point_pred_list[i], joint_axis_pred_list[i], length=0.5,color=axis_color)
    else:
        mobile_color = 'lightskyblue'
        axis_color= 'blue'
        joint_axis_line = vector_to_plotly_line(pivot_point_pred_list[i], joint_axis_pred_list[i], length=0.5,color=axis_color)

    
    fig.add_trace(trimesh_to_plotly(static_part, color=static_color))
    fig.add_trace(trimesh_to_plotly(mobile_part, color=mobile_color))

    # Add joint axis line (assuming it's defined outside the loop)
    fig.add_trace(joint_axis_line)

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    paper_bgcolor='rgba(250,250,250,250)',  # Light background
    plot_bgcolor='rgba(0,0,0,0)',
    title="3D Scene with Multiple Meshes and Joint Axis"
)

fig.show()


"""
scene = trimesh.Scene()
static_part = mesh_dict[0].copy()
mobile_part = mesh_dict[1].copy()
scene.add_geometry(static_part)
scene.add_geometry(mobile_part)
add_r_joint_to_scene(scene, joint_axis_pred_list[2], pivot_point_pred_list[2], 1.0, recenter=True,joint_color=[200, 0, 0, 180])
add_r_joint_to_scene(scene, joint_axis_pred_list[3], pivot_point_pred_list[3], 1.0, recenter=True,joint_color=[0, 200, 0, 180])

#scene.show()

o3d_meshes = []

for i, mesh_dict in enumerate(mesh_list):
    static_part = mesh_dict[0].copy()
    mobile_part = mesh_dict[1].copy()

    # Add material/visuals to mobile part
    mobile_part.visual.face_colors = np.array([84, 220, 83, 0], dtype=np.uint8)
    
    o3d_static = trimesh_to_open3d(static_part)
    o3d_mobile = trimesh_to_open3d(mobile_part)

    o3d_meshes.extend([o3d_static, o3d_mobile])
o3d.visualization.draw_geometries(o3d_meshes)

    # Optional: convert joint visuals (e.g., small cylinder to indicate axis)
    joint_geom = create_joint_o3d_visual(
        axis=joint_axis_pred_list[i],
        pivot=pivot_point_pred_list[i],
        length=1.0,
        radius=0.01,
        color=[(50*i+20)/255.0, (70*i+20)/255.0, 60/255.0]
    )
    o3d_meshes.append(joint_geom)
    """
""" 

 
camera_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
light_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

img = Image.fromarray(rgb)
img.show()
"""


