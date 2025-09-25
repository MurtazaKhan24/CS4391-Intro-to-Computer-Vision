"""
CS 4391 Homework 2 Programming: Part 4 - non-local means filter
Implement the nlm_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def nlm_filtering(
    img: np.uint8,
    intensity_variance: float,
    patch_size: int,
    window_size: int,
) -> np.uint8:
    """
    Homework 2 Part 4
    Compute the filtered image given an input image, kernel size of image patch, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")  # input image
    img_filtered = np.zeros(img.shape)  # Placeholder of the filtered image

    sizeX, sizeY = img.shape
    patch_radius = patch_size // 2
    window_radius = window_size // 2

    # zero-padding for boundary handling
    pad_size = window_radius + patch_radius
    padded = np.pad(img, pad_size, mode="reflect")

    # filtering for each pixel
    for i in range(sizeX):
        for j in range(sizeY):
            # reference patch centered at (i, j)
            ref_patch = padded[
                i + pad_size - patch_radius : i + pad_size + patch_radius + 1,
                j + pad_size - patch_radius : j + pad_size + patch_radius + 1,
            ]

            weights = []
            neighbors = []

            # iterate over search window
            for wi in range(-window_radius, window_radius + 1):
                for wj in range(-window_radius, window_radius + 1):
                    ni = i + wi
                    nj = j + wj

                    # neighbor patch
                    neigh_patch = padded[
                        ni + pad_size - patch_radius : ni + pad_size + patch_radius + 1,
                        nj + pad_size - patch_radius : nj + pad_size + patch_radius + 1,
                    ]

                    # squared distance between patches
                    dist2 = np.sum((ref_patch - neigh_patch) ** 2)

                    # compute weight
                    w = np.exp(-dist2 / intensity_variance)
                    weights.append(w)
                    neighbors.append(padded[ni + pad_size, nj + pad_size])

            # normalize weights
            weights = np.array(weights)
            neighbors = np.array(neighbors)
            weights /= np.sum(weights)

            # compute filtered pixel
            img_filtered[i, j] = np.sum(weights * neighbors)

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    intensity_variance = 1
    patch_size = 5 # small image patch size
    window_size = 15 # serach window size
    img_bi = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_bi)