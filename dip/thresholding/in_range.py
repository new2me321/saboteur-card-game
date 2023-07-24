#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from dip.thresholding.single_point_greyscale import ThresType, threshold
from dip.colourspaces.convert import rgb_to_grey


def in_range(img: np.ndarray, lowerBounds: np.ndarray, upperBounds: np.ndarray) -> np.ndarray:

    img_thresh = np.zeros_like(img)

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    R_threshU = np.where(R <= upperBounds[0], 255, 0)
    R_threshL = np.where(R < lowerBounds[0], 0, 255)

    G_threshU = np.where(G <= upperBounds[1], 255, 0)
    G_threshL = np.where(G < lowerBounds[1], 0, 255)

    B_threshU = np.where(B <= upperBounds[2], 255, 0)
    B_threshL = np.where(B < lowerBounds[2], 0, 255)

    R_thresh = np.logical_and(R_threshL, R_threshU)
    G_thresh = np.logical_and(G_threshL, G_threshU)
    B_thresh = np.logical_and(B_threshL, B_threshU)

    # in case lowerbounds are greater than upper bounds
    if lowerBounds[0] > upperBounds[0]:
        R_thresh = np.logical_or(R_threshL, R_threshU)
    if lowerBounds[1] > upperBounds[1]:
        G_thresh = np.logical_or(G_threshL, G_threshU)
    if lowerBounds[2] > upperBounds[2]:
        B_thresh = np.logical_or(B_threshL, B_threshU)

    # BW
    img_thresh[np.logical_and(
        B_thresh, np.logical_and(G_thresh, R_thresh))] = 255

    # colored
    img_thresh_color = np.bitwise_and(img_thresh, img)

    return img_thresh_color.astype(np.uint8), rgb_to_grey(img_thresh).astype(np.uint8)
