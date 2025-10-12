import numpy as np
import open3d as o3d
import trimesh

def generate_colored_mesh_from_segmented_pointcloud(points, mask_static, mask_mobile, method="poisson"):
    """
    Generates a colored mesh from a full point cloud and two masks for static and mobile parts.
    Colors: Red = static, Blue = mobile

    Args:
        points (np.ndarray): (N, 3) full point cloud
        mask_static (np.ndarray): (N,) boolean or 0/1 array
        mask_mobile (np.ndarray): (N,) boolean or 0/1 array
        method (str): "poisson" or "bpa" (ball pivoting)

    Returns:
        mesh (trimesh.Trimesh): Colored mesh with unified geometry
    """
    assert points.shape[1] == 3
    assert len(mask_static) == len(points)
    assert len(mask_mobile) == len(points)

    # Step 1: Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Step 2: Surface reconstruction
    if method == "poisson":
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    elif method == "bpa":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    else:
        raise ValueError("Unsupported method: use 'poisson' or 'bpa'")

    # Clean up the mesh
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_unreferenced_vertices()

    # Step 3: Convert to Trimesh
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=False
    )

    # Step 4: Label vertices by nearest point in original cloud
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, idxs = tree.query(mesh.vertices)
    
    # Create color array
    colors = np.zeros((len(mesh.vertices), 3))  # RGB
    
    static_mask = mask_static[idxs]
    mobile_mask = mask_mobile[idxs]

    colors[static_mask] = [1.0, 0.0, 0.0]  # Red
    colors[mobile_mask] = [0.0, 0.0, 1.0]  # Blue

    mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)

    return mesh
def crop_mesh_to_input_bounds(mesh, points, padding=0.02):
    """
    Removes faces in the mesh that are far from the input point cloud.
    """
    min_bound = points.min(axis=0) - padding
    max_bound = points.max(axis=0) + padding

    in_box = np.all((mesh.vertices >= min_bound) & (mesh.vertices <= max_bound), axis=1)
    valid_faces = np.all(in_box[mesh.faces], axis=1)

    mesh.update_faces(valid_faces)
    mesh.remove_unreferenced_vertices()
    return mesh


def add_joint_axis_to_mesh(mesh, pivot_point_pred, joint_axis_pred, axis_length=0.5, color=[0, 255, 0]):
    """
    Add a visualized axis to a trimesh.Scene using a line representing the joint axis.

    Returns:
        scene: trimesh.Scene with mesh and axis line.
    """
    joint_axis_pred = joint_axis_pred / np.linalg.norm(joint_axis_pred)

    # Define start and end of the axis
    start = pivot_point_pred-np.array([0,0,0.5])
    end = pivot_point_pred + joint_axis_pred * axis_length

    # Create the axis as a Line
    axis_line = trimesh.load_path(np.vstack([start, end]))

    # Set color (green by default)
    axis_line.colors = np.array([color])

    # Build the scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(axis_line)

    return scene

data = np.load("./PointSam_res.npy", allow_pickle=True).item()
points = data['xyz']
masks = data['mask']
mask_static = masks[0]
mask_mobile = masks[1]
pivot_point_pred = np.array([-0.01077165,0.26204193,-0.0014641])
joint_axis_pred = np.array([0.0108894,-0.00532914,0.99984896])

mesh = generate_colored_mesh_from_segmented_pointcloud(points, mask_static, mask_mobile)
mesh = crop_mesh_to_input_bounds(mesh, points)
# Save or show
mesh.export("segmented_colored_mesh.ply")  # or .glb
scene = add_joint_axis_to_mesh(mesh, pivot_point_pred, joint_axis_pred)
scene.show()
