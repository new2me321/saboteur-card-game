#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dip.machine_learning.k_means import k_means
import numpy as np
from dip.thresholding.single_point_greyscale import threshold, ThresType


def otsu_thresholding(image: np.ndarray) -> np.ndarray:

    # flatten image
    img_flat = image.flatten()

    # get center values from Kmeans
    centroids = k_means(img_flat, K=2)
    
    # get pivot value from the mean of the centers
    centroids_mean = int(np.mean(centroids))

    # threshold image.
    img_thresh = threshold(image, centroids_mean, ThresType.BLACK_WHITE)


    return img_thresh
