import numpy as np
import open3d as o3d
points = np.load("./PC/pc_end_j0.npy")
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])
