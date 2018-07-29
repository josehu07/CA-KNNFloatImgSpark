import numpy as np
from pyspark import SparkContext

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
	x_grid = np.abs(x_grid - halfSize)
	y_grid = np.abs(y_grid - halfSize)
	# Generate RDD and calculate
	def calcDist(tup):
		calc_kernel = np.abs(image[tup[0] - halfSize:tup[0] + halfSize + 1, \
								   tup[1] - halfSize:tup[1] + halfSize + 1] - image[tup]) \
					  + x_grid * sqrt_reverse_cos_angle_width_per_pixel * image[tup] \
					  + y_grid * sqrt_reverse_cos_angle_height_per_pixel * image[tup]
		calc_kernel = calc_kernel[np.where(calc_kernel != 0)]
		return np.mean(calc_kernel[np.argpartition(calc_kernel, k)[:k]])
	mean_list = sc.parallelize([(i, j) for i in range(halfSize, image.shape[0] - halfSize) \
									   for j in range(halfSize, image.shape[1] - halfSize)], 20) \
				  .filter(lambda tup: image[tup] != 0) \
				  .map(lambda tup: (tup, calcDist(tup))) \
				  .collect()
	# Judge and write
	for tup in mean_list:
		if (tup[1] > distance_thr):
			cnt += 1
		else:
			filtered_image[tup[0]] = image[tup[0]]
	# Return
	return filtered_image, cnt
