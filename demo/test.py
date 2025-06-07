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

def generate_random_mesh_pair():
    # Create a base mesh (e.g., a box or a sphere)
    mesh_type = np.random.choice(['box', 'sphere', 'cylinder'])
    if mesh_type == 'box':
        static = trimesh.creation.box(extents=np.random.rand(3) + 0.5)
    elif mesh_type == 'sphere':
        static = trimesh.creation.icosphere(radius=np.random.rand() + 0.3)
    elif mesh_type == 'cylinder':
        static = trimesh.creation.cylinder(radius=np.random.rand() + 0.2, height=np.random.rand() + 0.5)

    # Create mobile part by translating static one slightly
    mobile = static.copy()
    translation_vector = np.random.randn(3) * 0.1
    mobile.apply_translation(translation_vector)

    return [static, mobile]

# Generate a list of such mesh pairs
mesh_list = [generate_random_mesh_pair() for _ in range(5)]

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
"""
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