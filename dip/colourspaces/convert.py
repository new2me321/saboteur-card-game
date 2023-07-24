#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    M = np.maximum(np.maximum(R, G), B)
    m = np.minimum(np.minimum(R, G), B)

    # print(M, m)
    num = R-(G/2)-(B/2)

    denom = np.sqrt((R**2) + (G**2) + (B**2) - (R*G) - (R*B) - (G*B))
    # print(num[0][0], denom[0][0])

    # take care of division-by-zero with 1e-9
    d = np.round(num/(denom + 1e-12), 4)
    theta = np.arccos(d)/2
    # print(cos_inv[0], "\n\n")

    ###### H ######
    H = np.zeros_like(G)
    H = np.where(G <= B, H, theta)
    H = np.where(G > B, H, np.pi - theta)

    H = np.degrees(H)
    H = np.where(H < 179, H, 0)

    print(H.max())

    ###### S ######
    S = np.zeros_like(G)
    S = np.where(M == 0, S, 1-(m/(M+1e-15)))

    ###### V ######
    V = M

    print("RGB to HSV conversion complete!")

    return np.dstack([(np.round(H)).astype(np.uint8), (np.round(255*S)).astype(np.uint8), (255*V).astype(np.uint8)])


def rgb_to_grey(image: np.ndarray) -> np.ndarray:
    # print("RGB to Grey conversion complete!")
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    grey = 0.299*R + 0.587*G + 0.114*B
    # print(grey.shape)
    return grey


def rgb_to_cmyk(image: np.ndarray) -> np.ndarray:

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    if np.max(R) >= 1.5:
        # assume ranges are [0,255]
        R, G, B = R/255, G/255, B/255

    K = 1-np.maximum(np.maximum(R, G), B)

    denom = 1-K  # common denom
    C = np.divide(1-R-K, denom)

    M = np.divide(1-G-K, denom)

    Y = np.divide(1-B-K, denom)

    print("RGB to CMYK conversion complete!", np.max(C), np.max(M), np.max(Y), np.max(K))
    return np.dstack([C, M, Y, K])
