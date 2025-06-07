import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from functions_for_demo import *

# norm_center_oif=np.load('./norm_center_oif.npy',allow_pickle=True)
# norm_scale_oif=np.load('./norm_scale_oif.npy',allow_pickle=True)
# norm_center_iif=np.load('./norm_center_iif.npy',allow_pickle=True)
# norm_scale_iif=np.load('./norm_scale_iif.npy',allow_pickle=True)
# mobile_points_j0=np.load('./points_norms/mobile_points_j0.npy')
# mobile_points_j1=np.load('./points_norms/mobile_points_j1.npy')
# static_points_j0=np.load('./points_norms/static_points_j0.npy')
# static_points_j1=np.load('./points_norms/static_points_j1.npy')
# static_pcs = [static_points_j0,static_points_j1]
# for i in range(2):
#     static_pcs[i-1] -=(norm_center_oif[i-1]-norm_center_oif[i])/((norm_scale_oif[i-1]+norm_scale_oif[i])/2)


a0=np.load('./points_norms/a0.npy')
a1=np.load('./points_norms/a1.npy')
b0=np.load('./points_norms/b0.npy')
b1=np.load('./points_norms/b1.npy')
norm0=np.load('./points_norms/norm0.npy')
norm1=np.load('./points_norms/norm1.npy')

fig1 = plt.figure()
#fig1.suptitle("Input", fontsize=14)
ax1 = fig1.add_subplot(1,1,1, projection='3d')
#ax2 = fig1.add_subplot(1,2,2, projection='3d')
#ax3 = fig1.add_subplot(1,3,3, projection='3d')

plot_3d_point_cloud(*a0.T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    color ='blue',
                    alpha=0.1,
                    title = 'Normalization Center Difference',                    
                    lim=[(-0.5, 0.5)] * 3)

plot_3d_point_cloud(*b0.T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    alpha=0.1,
                    color='blue',
                    lim=[(-0.5, 0.5)] * 3)

plot_3d_point_cloud(*a1.T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    alpha=0.1,                    
                    color='purple',
                    lim=[(-0.5, 0.5)] * 3)



plot_3d_point_cloud(*b1.T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    alpha=0.1,
                    color='purple',
                    lim=[(-0.5, 0.5)] * 3)

plot_3d_point_cloud(*norm0.T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    s=200,
                    alpha=1,
                    color='green',
                    lim=[(-0.5, 0.5)] * 3)



plot_3d_point_cloud(*norm1.T,
                    axis=ax1,
                    azim=180,
                    elev=90,
                    s=200,
                    alpha=1,
                    color='red',
                    lim=[(-0.5, 0.5)] * 3)

p0 = norm0
p1 = norm1

# Compute direction vector
direction = p1 - p0

# Plot arrow
ax1.quiver(p0[0], p0[1], p0[2],
           direction[0], direction[1], direction[2],
           color='black', arrow_length_ratio=0.1, linewidth=2)
# #Compute direction
# dx = p1[0] - p0[0]  # X difference
# dy = p1[1] - p0[1]  # Y difference
# z = p0[2]           # Keep Z fixed

# # Horizontal arrow (X axis)
# ax1.quiver(p0[0], p0[1], z,
#            dx, 0, 0,
#            color='black', arrow_length_ratio=0.1, linewidth=2)

# # Vertical arrow (Y axis), starting where the horizontal arrow ends
# intermediate_x = p0[0] + dx
# intermediate_y = p0[1]
# ax1.quiver(intermediate_x, intermediate_y, z,
#            0, dy, 0,
#            color='black', arrow_length_ratio=0.1, linewidth=2)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_xy_box(ax, corners, z=0.0, color=None, alpha=0.0, edge_color='k'):
    """
    Plots a flat box (rectangle) on the XY plane at a specified Z level.
    
    Parameters:
        ax (Axes3D): The 3D axis to plot on.
        corners (array-like): A list or array of four (x, y) tuples or shape (4, 2) array.
        z (float): The Z level where the box lies. Default is 0.
        color (str): Fill color of the box.
        alpha (float): Transparency.
        edge_color (str): Color of the box edges.
    """
    # Ensure corners is a NumPy array
    corners = np.asarray(corners)
    
    if corners.shape != (4, 2):
        raise ValueError("corners must be a (4, 2) array-like object representing four (x, y) points.")
    
    # Create 3D points with fixed Z
    verts = [[(x, y, z) for x, y in corners]]

    # Create and add polygon
    poly = Poly3DCollection(verts, facecolors=color, edgecolors=edge_color, alpha=alpha)
    ax.add_collection3d(poly)


corners_xy = [
    [0.261, -0.2],
    [-0.269,-0.2],
    [-0.269, 0.438],
    [0.261, 0.438]
]

#plot_xy_box(ax1, corners_xy, z=0.2, color='yellow', alpha=0.5)
plt.show()   
