"""
CS 4391 Homework 1 Programming
Backprojection
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def backproject(depth, intrinsic_matrix):
    H, W = depth.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Stack pixel coordinates and convert to homogeneous coordinates
    homogeneous_coords = np.stack((u, v, np.ones_like(u)), axis=-1)  

    # Reshape to (H*W, 3) for matrix multiplication
    homogeneous_coords = homogeneous_coords.reshape(-1, 3)

    # Compute the inverse of the intrinsic matrix
    intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

    # Apply the inverse intrinsic matrix to the homogeneous coordinates
    camera_coords = homogeneous_coords @ intrinsic_matrix_inv.T 

    # Multiply by the depth values
    depth_flat = depth.flatten()[:, np.newaxis] 
    point_cloud = camera_coords * depth_flat  

    # Reshape back to (H, W, 3)
    point_cloud = point_cloud.reshape(H, W, 3)

    return point_cloud


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)
    
    # depth image
    depth_filename = 'data/000006-depth.png'
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    # convert from mm to m
    depth = depth / 1000.0
    
    # read the mask image
    mask_filename = 'data/000006-label-binary.png'
    mask = cv2.imread(mask_filename)
    mask = mask[:, :, 0]
    
    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    # load matedata
    meta_filename = 'data/000006-meta.mat'
    meta = scipy.io.loadmat(meta_filename)
    
    # intrinsic matrix
    intrinsic_matrix = meta['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
    
    # backprojection
    pcloud = backproject(depth, intrinsic_matrix)
        
    # get the points on the box
    pbox = pcloud[mask > 0, :]
    index = pbox[:, 2] > 0
    pbox = pbox[index]
    print(pbox.shape)
        
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show depth image
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(depth)
    ax.set_title('depth image')
        
    # show segmentation mask
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(mask)
    ax.set_title('segmentation mask')
        
    # up to now, suppose you get the points box as pbox
    # then you can use the following code to visualize the points in pbox
    # You shall see the figure in the homework assignment
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(pbox[:, 0], pbox[:, 1], pbox[:, 2], marker='.', color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D ploud cloud of the box')
                  
    plt.show()
