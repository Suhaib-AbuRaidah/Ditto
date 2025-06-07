import open3d as o3d
import numpy as np


pc_file = np.load("./scenes/3e39c013a07d437ba31a473c6f00adc6.npz", allow_pickle=True)
print(pc_file.files)
#points = np.load("./sample_1_points.npy")
  
for key in pc_file.files:
    points = pc_file[key]
    if np.asarray(points).ndim !=2 or key=="start_occ_list" or key=="end_occ_list" or key=="start_mesh_pose_dict":
        print(f"{key} is skipped,        shape: {np.asarray(points).shape}")
        print(f"{pc_file[key]}\n\n")
        continue
    print(f"{key},        shape: {np.asarray(points).shape}")


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    o3d.visualization.draw_geometries([point_cloud])



