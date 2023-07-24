#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### To be used for basic segmentation ####

import ast
import numpy as np
import cv2
import os.path as path

good_cards_outline = {'lH': 0, 'lS': 89, 'lV': 191, 'hH': 242, 'hS': 255, 'hV': 255}
path_ = {'lH': 0, 'lS': 0, 'lV': 209, 'hH': 255, 'hS': 255, 'hV': 255}



def update_value(new_value):
    global trackbar_value
    trackbar_value = new_value


HSV = {'lH': 0, 'lS': 0, 'lV': 0, 'hH': 255, 'hS': 255, 'hV': 255}
if path.exists("trackbar_defaults.txt"):
    with open("trackbar_defaults.txt", 'r') as f:
        data = f.read()
        HSV = ast.literal_eval(data)
        # print(HSV)

# trackbar_value = 0

# create trackbar
cv2.namedWindow("Threshold")
cv2.createTrackbar("lH", "Threshold", HSV['lH'], 255, update_value)
cv2.createTrackbar("lS", "Threshold", HSV['lS'], 255, update_value)
cv2.createTrackbar("lV", "Threshold", HSV['lV'], 255, update_value)
cv2.createTrackbar("hH", "Threshold", HSV['hH'], 255, update_value)
cv2.createTrackbar("hS", "Threshold", HSV['hS'], 255, update_value)
cv2.createTrackbar("hV", "Threshold", HSV['hV'], 255, update_value)

frame = cv2.imread("solutions/data/saboteur_test_images/002.jpg")
# Open the camera

resize_scale = 1
player_rect = (80, 380, 300, 260)
board_rect = (440, 57, 740, 600) # x, y, w, h
# extract region from image
board_area = frame[board_rect[1]:board_rect[1] +
                    board_rect[3], board_rect[0]:board_rect[0]+board_rect[2]]
player_area = frame[player_rect[1]:player_rect[1] +
                    player_rect[3], player_rect[0]:player_rect[0]+player_rect[2]]
board_area = cv2.resize(board_area, (board_area.shape[1]//resize_scale, board_area.shape[0]//resize_scale))
player_area = cv2.resize(player_area, (player_area.shape[1]//resize_scale, player_area.shape[0]//resize_scale))
while True:
      # Read the image from the camera
    # frame = cv2.cvtColor(board_area, cv2.COLOR_RGB2BGR)
    frame = board_area
    # You will need this later
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Colour detection limits
    HSV['lH'] = cv2.getTrackbarPos('lH', "Threshold")
    HSV['lS'] = cv2.getTrackbarPos('lS', "Threshold")
    HSV['lV'] = cv2.getTrackbarPos('lV', "Threshold")
    HSV['hH'] = cv2.getTrackbarPos('hH', "Threshold")
    HSV['hS'] = cv2.getTrackbarPos('hS', "Threshold")
    HSV['hV'] = cv2.getTrackbarPos('hV', "Threshold")

    lowerLimits = np.array([HSV['lH'], HSV['lS'], HSV['lV']])
    upperLimits = np.array([HSV['hH'], HSV['hS'], HSV['hV']])

    # Our operations on the frame come here
    thresholded = cv2.inRange(frame_hsv, lowerLimits, upperLimits)
    outimage = cv2.bitwise_and(frame, frame, mask=thresholded)

    count = np.sum(outimage >= 1)

    print("White pixels count:", count)
    cv2.imshow('Original', frame)

    # Display the resulting frame
    cv2.imshow('Processed', outimage)
    cv2.imshow("Threshold", thresholded)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
try:
    with open("trackbar_defaults.txt", 'w') as f:
        f.write(str(HSV))
except FileNotFoundError:
    pass

# When everything done, release the capture
print('closing program')

cv2.destroyAllWindows()