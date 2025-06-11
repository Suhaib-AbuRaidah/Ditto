import numpy as np
import open3d as o3d

pc_start=np.load("pc_start.npy")
pc_seg_start=np.load("pc_seg_start.npy")

num_points = pc_start.shape[0]

color = np.ones((num_points,3))

colored_points = np.hstack((pc_start,color))

colored_points[pc_seg_start, 3:] = [120, 255, 255]
colored_points[~pc_seg_start, 3:] = [255, 2500, 255]

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(colored_points[:,:3])
# point_cloud.colors = o3d.utility.Vector3dVector(colored_points[:,3:])
# o3d.visualization.draw_geometries([point_cloud])

np.save("colored_points.npy",colored_points)
