import trimesh
import numpy as np
import open3d as o3d

def mesh_from_points(points: np.ndarray, estimate_normals=True, method="poisson"):
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if estimate_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

    # Reconstruct mesh
    if method == "poisson":
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
    elif method == "bpa":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    else:
        raise ValueError("Unknown method")

    # Remove unreferenced vertices
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_unreferenced_vertices()

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)



def generate_mesh_dict_from_segments(static_points, mobile_points):
    mesh_dict = {}
    mesh_dict[0] = mesh_from_points(static_points, method="poisson")
    mesh_dict[1] = mesh_from_points(mobile_points, method="poisson")
    return mesh_dict
