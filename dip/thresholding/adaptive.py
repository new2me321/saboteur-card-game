#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def mean_adaptive_thresholding(img: np.ndarray, kernel_size: int, C: int) -> np.ndarray:
    img_thresh = np.zeros_like(img, dtype=np.uint8)
  
    h_pad = kernel_size // 2
    w_pad = kernel_size // 2

    # creating a padded array to allow convolution / sliding window to work
    img_padded = np.pad(img, (h_pad, w_pad), mode='edge')
    
    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        img_padded, (kernel_size, kernel_size))

    # find the mean of each sliding window
    img_mean = np.mean(sliding_windows, axis=(2, 3))
    # print(sliding_windows)
    # print(f"img: {img.shape},  img_mean: {img_mean.shape}, img_padded: {img_padded.shape} ")

    # thresholding step. C
    img_thresh = np.where(img > img_mean - C, 255, 0)
    

    # print(f"img_thresh{img_thresh}")
    return img_thresh.astype(np.uint8)


def gaussian_adaptive_thresholding(img: np.ndarray, kernel_size: int, sigma: float, C: int) -> np.ndarray:
    img_thresh = np.zeros_like(img, dtype=np.uint8)
    kernel = cv2.getGaussianKernel(kernel_size, sigma=sigma)
    kernel2d = kernel * kernel.T
    # print(f"kernel2d{kernel2d.shape}, kern {kernel.shape}, kern.T {kernel.T.shape},")

    h_pad = kernel_size // 2
    w_pad = kernel_size // 2

    # creating a padded array to allow convolution / sliding window to work
    img_padded = np.pad(img, (h_pad, w_pad), mode='edge')

    img_conv = cv2.filter2D(img_padded, -1, kernel2d)

    # img_mean = np.divide(img_conv, np.sum(kernel))
    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        img_conv, (kernel_size, kernel_size))

    # find the mean of each sliding window
    img_mean = np.mean(sliding_windows, axis=(2, 3))

    # thresholding step. C
    img_thresh = np.where(img > img_mean - C, 255, 0)

    # print(f"img_thresh{img_thresh}")
    return img_thresh.astype(np.uint8)
