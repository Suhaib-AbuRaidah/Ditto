import numpy as np
import open3d as o3d
import eulerangles as euang
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("raw_path_src", help="Path of the raw file containing depth data for source pose")
parser.add_argument("raw_path_des", help="Path of the raw file containing depth data for destination pose")
parser.add_argument("save_path_src", help="Path of the file to save the output point cloud of the source pose")
parser.add_argument("save_path_des", help="Path of the file to save the output point cloud of the destination pose")
args = parser.parse_args()


def depth_image_to_point_cloud(depth_image, intrinsic_matrix, min_bound, max_bound):
    """
    Converts a depth image to a point cloud using the camera intrinsic parameters.

    Args:
        depth_image (np.ndarray): The depth image (H x W).
        intrinsic_matrix (np.ndarray): The 3x3 intrinsic matrix.

    Returns:
        o3d.geometry.PointCloud: The resulting point cloud.
    """
    h, w = depth_image.shape

    # Generate a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image

    # Flatten the arrays
    x = x.flatten()
    
    y = y.flatten()
    
    z = z.flatten()

    # Create homogeneous pixel coordinates
    pixels = np.stack((x, y, np.ones_like(x)), axis=1)
    
    # Convert to camera coordinates
    camera_coords = np.linalg.inv(intrinsic_matrix) @ pixels.T * z

    # Transpose to get Nx3 array of points
    points = camera_coords.T

    # Remove points with zero depth
    points = points[z > 0]
    points = points*-1
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    point_cloud = point_cloud.crop(bbox)
    return point_cloud

# Example usage
if __name__ == "__main__":
    # Example intrinsic matrix (replace with actual values)
    fx, fy = 321.9, 321.9
    cx, cy = 318.1, 177.6
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    width, height = 640, 360
    raw_path1 = args.raw_path_src
    raw_path2 = args.raw_path_des


# Load raw Z16 depth image
    depth_raw1 = np.fromfile(raw_path1, dtype=np.uint16).reshape((height, width))
    depth_image1 = depth_raw1.astype(np.float32) / 1000.0
    depth_raw2 = np.fromfile(raw_path2, dtype=np.uint16).reshape((height, width))
    depth_image2 = depth_raw2.astype(np.float32) / 1000.0
    # Example depth image (replace with actual depth data)
    #depth_image = np.random.uniform(0, 5, (480, 640)).astype(np.float32)
    #bbx = [[-0.5,-0.16,-1],[0.15,0.2,-0.5]]
    bbxs = [[-0.4,-2,-1.6],[0.12,0.42,-0.5]]
    bbxd = [[-0.35,-2,-1.6],[0.12,0.42,-0.5]]
    # Convert to point cloud
    pcd1 = depth_image_to_point_cloud(depth_image1, intrinsic_matrix
                                     ,bbxs[0],bbxs[1])
    pcd2 = depth_image_to_point_cloud(depth_image2, intrinsic_matrix
                                     ,bbxd[0],bbxd[1])
    
    trans = np.zeros((4,4))
    rot = euang.euler2matrix([90,90,0],axes='zxz',
                                intrinsic=True,right_handed_rotation=True)
    trans[:3,:3] = rot
    trans[3,3] = 1
    
    pcd1.transform(trans)
    pcd2.transform(trans)
    pcd_arr1 = np.asarray(pcd1.points)
    pcd_arr2 = np.asarray(pcd2.points)
    print(pcd_arr1.shape)
    np.save(args.save_path_src,pcd_arr1)
    np.save(args.save_path_des,pcd_arr2)
    # Visualize
    o3d.visualization.draw_geometries([pcd1])
    o3d.visualization.draw_geometries([pcd2])

