#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum
import numpy as np
import cv2


class ThresType(enum.Enum):
    BLACK_WHITE = 0  # opencv binary
    BLACK_PIVOT = 1
    BLACK_SAME = 2  # opencv to_zero
    WHITE_BLACK = 3  # opencv bin_inv
    WHITE_PIVOT = 4
    PIVOT_BLACK = 5
    PIVOT_WHITE = 6
    PIVOT_SAME = 7
    SAME_WHITE = 8
    SAME_PIVOT = 9  # cv trunc
    # WHITE_SAME = 10


def threshold(img: np.ndarray, pivot: int, type: ThresType) -> np.ndarray:

    img_thresh = np.zeros_like(img)

    if type == ThresType.BLACK_WHITE:
        _, img_thresh = cv2.threshold(
            img, pivot, 255, type=cv2.THRESH_BINARY)

    elif type == ThresType.BLACK_PIVOT:
        img_thresh = np.where(img >= pivot, img, 0)  # black
        img_thresh = np.where(img < pivot, img_thresh, pivot)  # pivot

    elif type == ThresType.BLACK_SAME:
        img_thresh = np.where(img >= pivot, img, 0)

    elif type == ThresType.WHITE_BLACK:
        _, img_thresh = cv2.threshold(
            img, pivot, 255, type=cv2.THRESH_BINARY_INV)

    elif type == ThresType.WHITE_PIVOT:
        img_thresh = np.where(img >= pivot, img, 255)  # white
        img_thresh = np.where(img < pivot, img_thresh, pivot)  # pivot

    elif type == ThresType.PIVOT_BLACK:
        img_thresh = np.where(img >= pivot, img, pivot)  # pivot
        img_thresh = np.where(img < pivot, img_thresh, 0)  # black

    elif type == ThresType.PIVOT_WHITE:
        img_thresh = np.where(img >= pivot, img, pivot)  # pivot
        img_thresh = np.where(img < pivot, img_thresh, 255)  # white

    elif type == ThresType.PIVOT_SAME:
        img_thresh = np.where(img >= pivot, img, pivot)  # pivot

    elif type == ThresType.SAME_WHITE:
        img_thresh = np.where(img < pivot, img, 255)  # white

    elif type == ThresType.SAME_PIVOT:
        _, img_thresh = cv2.threshold(
            img, pivot, 255, type=cv2.THRESH_TRUNC)
        
    # elif type == ThresType.WHITE_SAME:
    #     img_thresh = np.where(img > pivot, img, 255)  # white

    return img_thresh
