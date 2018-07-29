import numpy as np
import sys
from pyspark import SparkContext

def knnRemoval(image, k, kernelSize, distance_thr):
    halfSize = kernelSize // 2
    cos_angle_width_per_pixel = np.cos(np.pi * 2 / image.shape[1])
    cos_angle_height_per_pixel = np.cos(np.pi * 2 / image.shape[0])
    x_grid, y_grid = np.meshgrid(list(range(kernelSize)),
                                 list(range(kernelSize)))
    x_grid = x_grid - halfSize
    y_grid = y_grid - halfSize

    sc = SparkContext()
    image_RDD = (sc.parallelize(range(halfSize, image.shape[0] - halfSize))
      .cartesian(sc.parallelize(range(halfSize, image.shape[1] - halfSize)))
      .map(lambda coordinate: (coordinate + (image[coordinate[0], coordinate[1]],))))

    def knnRemovalValue(y, x, center_depth):
        twice_depth_squared = 2 * center_depth ** 2
        width_per_pixel = np.sqrt(twice_depth_squared - (cos_angle_width_per_pixel * twice_depth_squared))
        height_per_pixel = np.sqrt(twice_depth_squared - (cos_angle_height_per_pixel * twice_depth_squared))
        kernel = np.abs(image[y-halfSize:y+halfSize+1, x-halfSize:x+halfSize+1] - center_depth)\
                 + np.abs(width_per_pixel * x_grid)\
                + np.abs(height_per_pixel * y_grid)
        sorted_kernel = np.sort(kernel.reshape(-1))
        sorted_kernel = sorted_kernel[np.where(sorted_kernel != 0)]
        mean = np.mean(sorted_kernel[:k])
        return mean > distance_thr

    known_RDD = image_RDD.filter(lambda pixel: pixel[2] != 0)
    known_RDD.cache()
    remain_RDD = known_RDD.filter(lambda pixel: not knnRemovalValue(*pixel))
    remain_RDD.cache()
    cnt = known_RDD.count() - remain_RDD.count()
    filtered_image = np.zeros(image.shape, dtype=image.dtype)
    pixels = remain_RDD.collect()
    for pixel in pixels:
        filtered_image[pixel[0], pixel[1]] = pixel[2]

    return filtered_image, cnt
