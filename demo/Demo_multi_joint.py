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
from functions_for_demo import *




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
    threshold=0.7,
    seg_threshold=0.5,
    input_type='pointcloud',
    refinement_step=0,
    padding=0.1,
    resolution0=32,
    upsampling_steps=3
)
num_joint = 2
num_plane =1
np.random.seed(0)
pc_start_list,pc_end_list = read_path(num_joint,num_plane)


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1, projection='3d')
# for i in range(1):
#     plot_3d_point_cloud(*pc_start_list[i].T,
#                     axis=ax1,
#                     azim=180,
#                     elev=90,
#                     lim=[(-0.1, 0.5)] * 3)
#     plot_3d_point_cloud(*pc_end_list[i].T,
#                         axis=ax1,
#                         azim=180,
#                         elev=90,
#                         lim=[(-0.5, 0.5)] * 3)
#     bound_max = np.maximum(pc_start_list[i].max(0), pc_end_list[i].max(0))
#     bound_min = np.minimum(pc_start_list[i].min(0), pc_end_list[i].min(0))
#     norm_center = (bound_max + bound_min) / 2
#     norm_scale = (bound_max - bound_min).max() * 1.1
#     pc_start_list[i] = (pc_start_list[i] - norm_center) / norm_scale
#     pc_end_list[i] = (pc_end_list[i] - norm_center) / norm_scale

#     plot_3d_point_cloud(*pc_start_list[i].T,
#                         axis=ax1,
#                         azim=180,
#                         elev=90,
#                         lim=[(-0.1, 0.5)] * 3)
#     plot_3d_point_cloud(*pc_end_list[i].T,
#                         axis=ax1,
#                         azim=180,
#                         elev=90,
#                         lim=[(-0.5, 0.5)] * 3)
    
# plt.show()



joint_axis_pred_list=[]
pivot_point_pred_list=[]
mesh_list = []
mobile_poits = []
mobile_points_arrs =[]
static_pcs= []
norm_center_oif = []
norm_scale_oif = []
norm_center_iif = []
norm_scale_iif = []
for i in range(num_joint):

    pc_start = pc_start_list[i]
    pc_end = pc_end_list[i]
    bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
    bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
    norm_center = (bound_max + bound_min) / 2
    norm_scale = (bound_max - bound_min).max() * 1.1
    norm_center_oif.append(norm_center)
    norm_scale_oif.append(norm_scale)
    pc_start = (pc_start-norm_center) / norm_scale
    pc_end = (pc_end-norm_center) / norm_scale
    print(f"Norm Center: {norm_center}\n\n Norm Scale: {norm_scale}")
 
    
    # origin = find_origin(pc_start)
    # translation = np.array([0,0,0])-origin
    # pc_start +=translation
    # pc_end += translation

    pc_start, idx = sample_point_cloud(pc_start, 20192)
    pc_end, _ = sample_point_cloud(pc_end, 20192)

    fig1 = plt.figure()
    fig1.suptitle("Input", fontsize=14)
    ax1 = fig1.add_subplot(1,1,1, projection='3d')

    plot_3d_point_cloud(*pc_start.T,
                        axis=ax1,
                        azim=180,
                        elev=90,
                        lim=[(-2, 2)] * 3)


    plot_3d_point_cloud(*pc_end.T,
                        axis=ax1,
                        azim=180,
                        elev=90,
                        lim=[(-2, 2)] * 3)
    plt.show()   


    if i!=0:
        if i ==1:
            static_pcs[i-1] +=(norm_center_oif[i-1]-norm_center_oif[i])/((norm_scale_oif[i-1]+norm_scale_oif[i])/2)
        else:
            norm_center_iif[i-2][2] = norm_center_oif[i][2]
            static_pcs[i-1] +=(norm_center_iif[i-2]-norm_center_oif[i])/(norm_scale_iif[i-2])

        pc_start=subtract_pc(pc_start,static_pcs[i-1],0.042)
        pc_start=np.asarray(pc_start.points)
        pc_end = subtract_pc(pc_end,static_pcs[i-1],0.042)  
        pc_end = np.asarray(pc_end.points)

        fig2 = plt.figure()
        fig2.suptitle("Input Normalized with Static Part to Subtract", fontsize=14)
        ax1 = fig2.add_subplot(1,1,1, projection='3d')

        plot_3d_point_cloud(*pc_start.T,
                            axis=ax1,
                            azim=180,
                            elev=90,
                            lim=[(-0.1, 0.8)] * 3)


        plot_3d_point_cloud(*pc_end.T,
                            axis=ax1,
                            azim=180,
                            elev=90,
                            lim=[(-0.5, 0.5)] * 3)
        

        plot_3d_point_cloud(*static_pcs[i-1].T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    lim=[(-0.5, 0.5)] * 3)
        
        plt.show()
        bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
        bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
        norm_center = (bound_max + bound_min) / 2
        norm_scale = (bound_max - bound_min).max() * 1.1
        norm_center_iif.append(norm_center)
        norm_scale_iif.append(norm_scale)
        pc_start = (pc_start - norm_center) / norm_scale
        pc_end = (pc_end - norm_center) / norm_scale
        print(f"Norm Center iif: {norm_center}\n\n Norm Scale iif: {norm_scale}")

        # origin = find_origin(pc_start)
        # translation = np.array([0,0,0])-origin
        # pc_start +=translation[0]
        # pc_end += translation[0]

        pc_start, _ = sample_point_cloud(pc_start, 20192)
        pc_end, _ = sample_point_cloud(pc_end, 20192)   


    
    sample = {
        'pc_start': torch.from_numpy(pc_end).unsqueeze(0).to(device).float(),
        'pc_end': torch.from_numpy(pc_start).unsqueeze(0).to(device).float()
    }


    mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
    
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)


    # compute articulation model
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


    mobile_points_arr=mobile_points_all.cpu().numpy().squeeze()
    static_pc = subtract_pc(pc_end,mobile_points_arr,0.025)
    static_pc = np.asarray(static_pc.points)
    mobile_points_arrs.append(mobile_points_arr)
    static_pcs.append(static_pc)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1, projection='3d')

    # plot_3d_point_cloud(*pc_end.T,
    #                     axis=ax1,
    #                     azim=180,
    #                     elev=90,
    #                     lim=[(-0.5, 0.5)] * 3)
    if i==1:
        a=mobile_points_arr+norm_center_iif[0]
        b=static_pc+norm_center_iif[0]
        np.save('./points_norms/a1.npy',a)
        np.save('./points_norms/b1.npy',b)
        np.save('./points_norms/norm1.npy',norm_center_iif[0])
        plot_3d_point_cloud(*(mobile_points_arr+norm_center_iif[0]).T,
                            axis=ax1,
                            azim=180,
                            elev=90,
                            lim=[(-0.5, 0.5)] * 3)
        
        plot_3d_point_cloud(*(static_pc+norm_center_iif[0]).T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    lim=[(-0.5, 0.5)] * 3)
        
        plot_3d_point_cloud(*norm_center_iif[0].T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    s=50,
                    color='red',
                    lim=[(-0.5, 0.5)] * 3)

    # Aggregate data
    mesh_list.append(mesh_dict)
    mobile_poits.append(mobile_points_all)
    joint_axis_pred_list.append(joint_axis_pred)
    pivot_point_pred_list.append(pivot_point_pred)


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,4,1, projection='3d')
    ax2 = fig1.add_subplot(1,4,2, projection='3d')
    ax3 = fig1.add_subplot(1,4,3, projection='3d')
    ax4 = fig1.add_subplot(1,4,4, projection='3d')

    plot_3d_point_cloud(*pc_start.T,
                        axis=ax1,
                        azim=180,
                        elev=90,
                        lim=[(-0.5, 0.5)] * 3)


    plot_3d_point_cloud(*pc_end.T,
                        axis=ax2,
                        azim=180,
                        elev=90,
                        lim=[(-0.5, 0.5)] * 3)
    
    plot_3d_point_cloud(*np.asarray(static_pc).T,
                        axis=ax3,
                        azim=180,
                        elev=90,
                        lim=[(-0.5, 0.5)] * 3)
    
    plot_3d_point_cloud(*mobile_points_arr.T,
                        axis=ax4,
                        azim=180,
                        elev=90,
                        lim=[(-0.5, 0.5)] * 3)
    
    plot_3d_point_cloud(*np.asarray(static_pc).T,
                        axis=ax4,
                        azim=180,
                        elev=90,
                        lim=[(-0.5, 0.5)] * 3)

# for i in range(len(static_pcs)):
#     np.save(f"static_pc{i}.npy",static_pcs[i])


fig2 = plt.figure()
fig2.suptitle("Input with Static Part to Subtract", fontsize=14)
ax1 = fig2.add_subplot(1,1,1, projection='3d')

plot_3d_point_cloud(*pc_start.T,
                    axis=ax1,
                    azim=-90,
                    elev=0,
                    lim=[(-0.1, 0.8)] * 3)


plot_3d_point_cloud(*pc_end.T,
                    axis=ax1,
                    azim=-90,
                    elev=0,
                    lim=[(-0.5, 0.5)] * 3)


plot_3d_point_cloud(*static_pcs[1].T,
            axis=ax1,
            azim=-90,
            elev=0,
            lim=[(-0.5, 0.5)] * 3)        
plt.show()

np.save('./points_norms/norm_center_oif.npy',norm_center_oif,allow_pickle=True)
np.save('./points_norms/norm_scale_oif.npy',norm_scale_oif,allow_pickle=True)
np.save('./points_norms/norm_center_iif.npy',norm_center_iif,allow_pickle=True)
np.save('./points_norms/norm_scale_iif.npy',norm_scale_iif,allow_pickle=True)
np.save('./points_norms/mobile_points_j0.npy',mobile_points_arrs[0])
np.save('./points_norms/mobile_points_j1.npy',mobile_points_arrs[1])
np.save('./points_norms/static_points_j0.npy',static_pcs[0])
np.save('./points_norms/static_points_j1.npy',static_pcs[1])
renderer = PyRenderer(light_kwargs={'color': np.array([1., 1., 1.]), 'intensity': 9})
scene = trimesh.Scene()
parts_static = []
parts_mobile = []

for i in range(num_joint):
    color_static = get_color(seed_offset=i^2+5*i)
    color_mobile = get_color(seed_offset=i^2+10*i+42) 
    mesh_list[i][0].visual.face_colors = color_static
    material = trimesh.visual.material.PBRMaterial(alphaMode="BLEND")
    mesh_list[i][0].visual.material = material

    mesh_list[i][1].visual.face_colors = color_mobile
    material = trimesh.visual.material.PBRMaterial(alphaMode="BLEND")
    mesh_list[i][1].visual.material = material

    parts_static.append(mesh_list[i][0].copy())
    parts_mobile .append(mesh_list[i][1].copy())


    scene.add_geometry(parts_static[i], node_name=f"part{2*i+1}")
    scene.add_geometry(parts_mobile[i], node_name=f"part{2*i+2}")

    # add corresponding joint visuals for this pose
    add_r_joint_to_scene(
    scene,
    joint_axis_pred_list[i],      
    pivot_point_pred_list[i],
    1.0,
    recenter=True,
    joint_color=color_static
    )

scene.export('scene1.glb')
camera_pose = get_pose(1.5, ax=0, ay=0, az=np.pi/2)
light_pose = get_pose(1.5, ax=0, ay=0, az=np.pi/2)
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

img = Image.fromarray(rgb)
img.show()


"""
fig = go.Figure()

for i, mesh_dict in enumerate(mesh_list):
    if i==0:
        static_part = mesh_dict[0].copy()
        mobile_part = mesh_dict[1].copy()
        static_color = 'rgb(150, 150, 150,1)'
        mobile_color = 'lightgreen'
        axis_color = 'red'
        joint_axis_line = vector_to_plotly_line(pivot_point_pred_list[i], joint_axis_pred_list[i], length=0.5,color=axis_color)
        fig.add_trace(trimesh_to_plotly(static_part, color=static_color))
        fig.add_trace(trimesh_to_plotly(mobile_part, color=mobile_color))
    if i==1:
        #static_part.apply_translation([0, 0, 0.125])
        #mobile_part.apply_translation([0, 0, 0.125])
        static_part= mobile_part
        mobile_part = mesh_dict[1].copy()
        static_color = 'lightgreen'
        mobile_color = 'lightpink'
        axis_color= 'blue'
        #joint_axis_line = vector_to_plotly_line1(pivot_point_pred_list[i], np.array([0.03,0.95,0.03]), length=0.5,color=axis_color)
        joint_axis_line = vector_to_plotly_line(pivot_point_pred_list[i], joint_axis_pred_list[i], length=0.5,color=axis_color)
        static_tip_y = static_part.vertices[:, 1].max()
        mobile_base_y = mobile_part.vertices[:, 1].min()
        translation_y = static_tip_y - mobile_base_y
        mobile_part.apply_translation([0, -translation_y, 0])
        # 1. Get the mobile base Y *after* translation
        mobile_base_y_translated = mobile_part.vertices[:, 1].max()
        print(mobile_base_y_translated)

        # 2. Crop static part: keep only region where vertex Y >= mobile_base_y_translated
        # Create a mask to select faces where all vertices are above the threshold
        keep_faces = []
        for face in static_part.faces:
            face_vertices = static_part.vertices[face]
            if (face_vertices[:, 1] >= mobile_base_y_translated).all():
                keep_faces.append(face)

        # 3. Create cropped mesh
        cropped_static_part = trimesh.Trimesh(
            vertices=static_part.vertices.copy(),
            faces=np.array(keep_faces),
            process=True
            )
        fig.add_trace(trimesh_to_plotly(cropped_static_part, color=static_color))
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





camera_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
light_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

img = Image.fromarray(rgb)
img.show()
"""



