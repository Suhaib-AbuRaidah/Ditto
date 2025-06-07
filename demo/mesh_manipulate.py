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
from trimesh.intersections import slice_mesh_plane
from trimesh.transformations import rotation_matrix
from src.utils.joint_estimation import aggregate_dense_prediction_r

from utils3d.mesh.utils import as_mesh
from utils3d.render.pyrender import get_pose, PyRenderer
import plotly.graph_objects as go
from functions_for_demo import *

norm_center_oif=np.load('./norm_center_oif.npy',allow_pickle=True)
norm_scale_oif=np.load('./norm_scale_oif.npy',allow_pickle=True)
norm_center_iif=np.load('./norm_center_iif.npy',allow_pickle=True)
norm_scale_iif=np.load('./norm_scale_iif.npy',allow_pickle=True)


static_parts = []
mobile_parts = []
axes = []

renderer = PyRenderer(light_kwargs={'color': np.array([1., 1., 1.]), 'intensity': 9})
scene = trimesh.load("scene1.glb")

num_joints = int(len(scene.geometry.keys())/3)

for i in range(num_joints):
    static_parts.append(scene.geometry[f"geometry_{3*i}"]) 
    mobile_parts.append(scene.geometry[f"geometry_{3*i+1}"])    
    axes.append(scene.geometry[f"geometry_{3*i+2}"]) 
    if i != num_joints-1:
        scene.delete_geometry([f"geometry_{3*i+1}"])

    if i!=0:
        translation = (norm_scale_oif[i]-norm_center_oif[0])/norm_scale_oif[i]
        print(translation)
        static_parts[i].apply_translation(translation)
        mobile_parts[i].apply_translation(translation)
        axes[i].apply_translation(translation)
translation = np.array([0.16,-0.32,-0.22])
static_parts[1].apply_translation(translation)
mobile_parts[1].apply_translation(translation)
axes[1].apply_translation(translation)
static_parts[1].visual.face_colors = np.array([20, 100, 160, 170], dtype=np.uint8)
axes[1].visual.face_colors = np.array([20, 100, 160, 170], dtype=np.uint8)
# R=rotation_matrix(4*np.pi/10, [1,0,0], point=[0,0,0.35])
# axes[1].apply_transform(R)
# axes[1].apply_translation(np.array([-0.1,0,-0.38]))

    # translation = np.array([0,0,0]) - static_parts[i].bounds[0]+np.array([-0.5,0,0])
    # static_parts[i].apply_translation(translation)
    # mobile_parts[i].apply_translation(translation)
    # axes[i].apply_translation(translation)
    #if i != num_joints-1:
    #    scene.delete_geometry([f"geometry_{3*i+1}"])



#R=rotation_matrix(4*np.pi/10, [1,0,0], point=[0,0,0.35])
#print(R)
#axis2.apply_transform(R)
#

camera_pose = get_pose(1.5, ax=0, ay=0, az=np.pi/2)
light_pose = get_pose(1.5, ax=0, ay=0, az=np.pi/2)
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

img = Image.fromarray(rgb)
img.show()
scene.export('scene3.glb')

#scene.delete_geometry("geometry_1")
#scene.delete_geometry("geometry_2")
#part3 = part3.slice_plane(plane_origin=[0,0,0], plane_normal=[0,0,1])
#part3.visual.face_colors = np.array([255, 0, 83, 170], dtype=np.uint8)
#part4.visual.face_colors = np.array([155, 190, 83, 170], dtype=np.uint8)

#scene.add_geometry(part3, node_name="part3")

"""
corner3 = get_lower_left_corner_points(part4)
corner1 = get_higher_left_corner_points(part1)

# Compute centroids
centroid3 = corner3.mean(axis=0)
centroid1 = corner1.mean(axis=0)
"""