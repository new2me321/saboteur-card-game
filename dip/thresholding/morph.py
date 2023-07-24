import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view


def dilation(image: np.ndarray, kernel_size: int) -> np.ndarray:
    res = np.zeros_like(image)

    img_padded = np.pad(image, (kernel_size//2, kernel_size//2))

    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         res[i, j] = np.max(img_padded[i:i+kernel_size, j:j+kernel_size])

    # vectorized version is faster
    image_windows = sliding_window_view(img_padded, (kernel_size, kernel_size))
    res = np.max(image_windows, axis=(2, 3))

    return res


def erosion(image: np.ndarray, kernel_size: int) -> np.ndarray:
    res = np.zeros_like(image)

    img_padded = np.pad(image, (kernel_size//2, kernel_size//2))

    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         res[i, j] = np.min(img_padded[i:i+kernel_size, j:j+kernel_size])

    # vectorized version is faster
    image_windows = sliding_window_view(img_padded, (kernel_size, kernel_size))
    res = np.min(image_windows, axis=(2, 3))

    return res


def opening(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return dilation(erosion(image, kernel_size), kernel_size)


def closing(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return erosion(dilation(image, kernel_size), kernel_size)


def skeletonize(image: np.ndarray, kernel_size: int) -> np.ndarray:
    img_curr = image.copy()
    img_prev = img_curr
    for i in range(1000):
        print(f"Iteration: {i}")
        img_open = opening(img_curr, kernel_size)

        img_temp = img_curr - img_open

        img_erode = erosion(img_curr, kernel_size)
        img_curr = np.bitwise_or(img_temp, img_erode)

        if np.array_equal(img_curr, img_prev):

            print("Converged! at iteration", i)
            break

        img_prev = img_curr

    return img_curr
