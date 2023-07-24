import numpy as np
import cv2


def flatten(nested_list):
    return [element for sublist in nested_list for element in sublist]


def calc_dist(A, B):
    return np.linalg.norm(A - B)


def drawContour(image: np.ndarray, contour,  color_rect=(0, 255, 0), color_text=(0, 0, 255), pos=None, target=False, label = None):
    '''
    Draws bounding box around the detected object on the given image.
    '''

    H, W, = image.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    if pos == 1:
        x = W//2 + x

    center_x = x + w // 2
    center_y = y + h // 2
    text = str((center_x, center_y))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    font_thickness = 1
    text_size, _ = cv2.getTextSize(
        text, font, font_scale, font_thickness)
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    # UNCOMMENT THIS TO PRINT TEXT
    # cv2.putText(image, text, (text_x, text_y), font, font_scale,
    #             color_text, font_thickness, cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y + h), color_rect, 2)
    if target:
        cv2.putText(image, "Target", (x+10, y+h-10), font, 0.5,
                    color_text, 2, cv2.LINE_AA)
    if label is not None:
        cv2.putText(image, label, (x+2, y+15), font, 0.5,
                    color_text, 2, cv2.LINE_AA)

    return image


def sort_corners(corners):
    """
    sorts corners in the order of top left, top right, bottom left, bottom right
    """
    corners = np.array(corners, dtype=np.float32)

    y_max = np.max(corners[:, 1])
    y_min = np.min(corners[:, 1])
    x_max = np.max(corners[:, 0])
    x_min = np.min(corners[:, 0])

    center = (x_min + x_max) / 2, (y_max + y_min) / 2

    above_center_corners = corners[corners[:, 1] < center[1]]
    below_center_corners = corners[corners[:, 1] > center[1]]

    # We sort by x-axis
    sorted_arr_above = above_center_corners[np.argsort(
        above_center_corners[:, 0])]
    sorted_arr_below = below_center_corners[np.argsort(
        below_center_corners[:, 0])]

    top_left = sorted_arr_above[0]
    top_right = sorted_arr_above[1]
    bottom_left = sorted_arr_below[0]
    bottom_right = sorted_arr_below[1]

    sorted_corners = np.array(
        [top_left, top_right, bottom_left, bottom_right], np.float32)
    return sorted_corners


def get_card_corners(contour):

    epsilon = 0.05 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    # print(polygon)

    if len(polygon) != 4:
        # print("Polygon does not have four vertices")
        return []

    corner_points = np.squeeze(polygon)
    sorted_corners = sort_corners(corner_points)
    return sorted_corners
