import numpy as np
import sys
import pdb

def knnRemoval(image, k, kernelSize, distance_thr):
    halfSize = int(kernelSize/2)
    angle_width_per_pixel = np.pi*2/image.shape[1]
    angle_height_per_pixel = np.pi*2/image.shape[0]
    x_grid, y_grid = np.meshgrid(list(range(kernelSize)), list(range(kernelSize)))
    cnt = np.zeros((1),dtype=np.uint32)
    count = 0

    def knnRemovalValue(image, y, x, k, kernelSize, distance_thr):
        center_depth = image[y,x]   
        depth_squared = center_depth**2
        width_per_pixel = np.sqrt( 2*depth_squared - (np.cos(angle_width_per_pixel)*2*depth_squared) )
        height_per_pixel = np.sqrt( 2*depth_squared - (np.cos(angle_height_per_pixel)*2*depth_squared) )
        kernel = np.abs(image[y-halfSize:y+halfSize+1, x-halfSize:x+halfSize+1] - center_depth) \
               + np.abs(width_per_pixel * (x_grid-halfSize)) \
               + np.abs(height_per_pixel * (y_grid-halfSize))
        sorted_kernel = np.sort(kernel.reshape(-1))
        sorted_kernel = sorted_kernel[np.where(sorted_kernel!=0)]
        mean = np.mean(sorted_kernel[:k])
        if(mean > distance_thr):
            cnt[0] += 1
            return 0.
        return image[y,x]
    
    filtered_image = np.zeros(image.shape, dtype=image.dtype)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(i >= halfSize and j >= halfSize and i < image.shape[0] - halfSize and j < image.shape[1] - halfSize):
                if(image[i,j] != 0):
                    filtered_image[i,j] = knnRemovalValue(image, i, j, k, kernelSize, distance_thr)
                    count += 1;
        sys.stdout.write("\rrow %d/%d of image"%(i, image.shape[0]))
        sys.stdout.flush()
    sys.stdout.write("\n")
    return filtered_image, cnt[0]
