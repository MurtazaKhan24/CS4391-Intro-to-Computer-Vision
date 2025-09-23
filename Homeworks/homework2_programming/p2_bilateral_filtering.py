"""
CS 4391 Homework 2 Programming: Part 3 - bilateral filter
Implement the bilateral_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance
    """

    img = img / 255.0
    img = img.astype("float32")
    img_filtered = np.zeros_like(img)
    xsize, ysize = img.shape

    padding = kernel_size // 2
    img_padded = np.pad(img, ((padding, padding), (padding, padding)), mode="reflect")

    for i in range(xsize):
        for j in range(ysize):
            Wp = 0.0
            filtered_pixel = 0.0

            center_val = img_padded[i + padding, j + padding]

            for k in range(-padding, padding + 1):
                for l in range(-padding, padding + 1):
                    neighbor_val = img_padded[i + k + padding, j + l + padding]

                    # Spatial Gaussian
                    spatial_weight = np.exp(-(k**2 + l**2) / (2 * spatial_variance))

                    # Intensity Gaussian
                    intensity_weight = np.exp(-((center_val - neighbor_val) ** 2) / (2 * intensity_variance))

                    weight = spatial_weight * intensity_weight
                    Wp += weight
                    filtered_pixel += weight * neighbor_val

            img_filtered[i, j] = filtered_pixel / Wp

    img_filtered = (img_filtered * 255).astype(np.uint8)
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
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)