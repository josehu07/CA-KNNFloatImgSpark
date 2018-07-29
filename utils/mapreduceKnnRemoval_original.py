import numpy as np
from pyspark import SparkContext
import sys

# KNNRemove Function
def knnRemoval(image, k, kernelSize, distance_thr):
    # Setting Spark context
    sc = SparkContext()
    # Image and counter
    filtered_image = np.zeros(image.shape, dtype = image.dtype)
    cnt = 0
    # Constants
    halfSize = int(kernelSize / 2)
    sqrt_reverse_cos_angle_width_per_pixel = np.sqrt(2 * (1 - np.cos(np.pi * 2 / image.shape[1])))
    sqrt_reverse_cos_angle_height_per_pixel = np.sqrt(2 * (1 - np.cos(np.pi * 2 / image.shape[0])))
    x_grid, y_grid = np.meshgrid(list(range(kernelSize)), list(range(kernelSize)))
    # Generate RDD and calculate
    for x in range(halfSize, image.shape[0] - halfSize):
        for y in range(halfSize, image.shape[1] - halfSize):
            if (image[x, y] != 0):
                center_depth = image[y,x]   
                width_per_pixel = sqrt_reverse_cos_angle_width_per_pixel * center_depth
                height_per_pixel = sqrt_reverse_cos_angle_height_per_pixel * center_depth
                meanList = sc.parallelize([(i, j, image[i, j]) for i in range(x - halfSize, x + halfSize + 1) \
                                                               for j in range(y - halfSize, y + halfSize + 1)], 16) \
                             .map(lambda tup: - np.abs(tup[2] - center_depth) \
                                              - np.abs(tup[0] - halfSize) * width_per_pixel \
                                              - np.abs(tup[1] - halfSize) * height_per_pixel) \
                             .filter(lambda x: x != 0) \
                             .top(k)
                if (-np.mean(meanList) > distance_thr):
                    cnt += 1
                    filtered_image[x, y] = 0
                else:
                    filtered_image[x, y] = image[x, y]
            else:
                filtered_image[x, y] = 0
        sys.stdout.write("\rrow %d/%d of image"%(y, image.shape[0]))
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Return
    return filtered_image, cnt
