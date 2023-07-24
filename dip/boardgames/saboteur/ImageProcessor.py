#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dip.thresholding.morph import dilation, erosion, closing
from dip.thresholding.single_point_greyscale import threshold,  ThresType
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        self.HSV_cards = {'lH': 97, 'lS': 0, 'lV': 0,
                          'hH': 156, 'hS': 255, 'hV': 255}  # cards HSV
        

    def threshold(self, image, pivot):
        return threshold(image, pivot, ThresType.BLACK_WHITE)

    def erode(self, image, kernel_size):
        return erosion(image, kernel_size)

    def dilate(self, image, kernel_size):
        return dilation(image, kernel_size)

    def processMap(self, path_image, thresh_val=100) -> tuple[np.ndarray, np.ndarray]:
        '''
        Extracts a map from the image. 
        Input:
            path_image: The path to the image
            thresh_val: The threshold value
        Output:
            map_bw, map_rgb : The thresholded map, also in RGB
        '''
        
        # blurred = cv2.GaussianBlur(path_image, (3, 3), 0)
        blurred = cv2.medianBlur(path_image, 3)
        # cv2.imshow('blurred', blurred)

        if path_image.ndim == 3:
            frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            HSV_ladder = {'lH': 0, 'lS': 130, 'lV': 56, 'hH': 255, 'hS': 255, 'hV': 255}
            lowerLimits = np.array([HSV_ladder['lH'], HSV_ladder['lS'], HSV_ladder['lV']])
            upperLimits = np.array([HSV_ladder['hH'], HSV_ladder['hS'], HSV_ladder['hV']])
            thresholded0 = cv2.inRange(frame_hsv, lowerLimits, upperLimits)
            thresholded0 = cv2.erode(thresholded0, cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)))
            thresholded0 = self.dilate(thresholded0, 13)
            thresholded0 = closing(thresholded0, 7)

            HSV_path = {'lH': 0, 'lS': 0, 'lV': 172, 'hH': 255, 'hS': 94, 'hV': 255} #{'lH': 108, 'lS': 0, 'lV': 165, 'hH': 255, 'hS': 56, 'hV': 255} #{'lH': 0, 'lS': 0, 'lV': 101, 'hH': 255, 'hS': 50, 'hV': 255}
            lowerLimits = np.array([HSV_path['lH'], HSV_path['lS'], HSV_path['lV']])
            upperLimits = np.array([HSV_path['hH'], HSV_path['hS'], HSV_path['hV']])
            thresholded1 = cv2.inRange(frame_hsv, lowerLimits, upperLimits)
            thresholded1 = self.dilate(thresholded1, 5)

            HSV_gold = {'lH': 0, 'lS': 115, 'lV': 56, 'hH': 255, 'hS': 255, 'hV': 255}
            lowerLimits = np.array([HSV_gold['lH'], HSV_gold['lS'], HSV_gold['lV']])
            upperLimits = np.array([HSV_gold['hH'], HSV_gold['hS'], HSV_gold['hV']])
            thresholded2 = cv2.inRange(frame_hsv, lowerLimits, upperLimits)
            thresholded2 = self.dilate(thresholded2, 3)
            thresholded = cv2.bitwise_or(cv2.bitwise_or(thresholded1, thresholded2), thresholded0)
        else:
            thresholded = self.threshold(blurred, thresh_val)


        res = cv2.erode(thresholded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # cv2.imshow('threshed_erode', res)

        map_bw = cv2.morphologyEx(res, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        map_rgb = cv2.cvtColor(map_bw, cv2.COLOR_GRAY2RGB)
        return map_bw, map_rgb

    def processCard(self, card_img, thresh_val=220, k_erode=7, k_dil=7):
        '''
        Takes in a card image(bounding box of the card) and returns the image with only the paths. 
        Also returns the card for validation
        Input:
            card_img: The image to process
            thresh_val: The threshold value
            k_erode: The kernel size for erosion
            k_dil: The kernel size for dilation
        Output:
            blurred: The blurred card image
            thresh: The thresholded card image     
        '''
        frame = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        # frame = cv2.GaussianBlur(histeq, (3, 3), 0)
        thresh = self.threshold(frame, thresh_val)
        thresh = self.dilate(thresh, k_dil)
        return frame, thresh

    def processImage(self, frame, HSV={'lH': 0, 'lS': 0, 'lV': 0, 'hH': 255, 'hS': 88, 'hV': 255}):#HSV = {'lH': 59, 'lS': 0, 'lV': 23, 'hH': 127, 'hS': 255, 'hV': 255}):#HSV = {'lH': 60, 'lS': 0, 'lV': 16, 'hH': 153, 'hS': 255, 'hV': 255}):
        '''
        Processes the RAW image.

        Returns:
            tuple: A thresholded image, masked image
        '''
        # frame = cv2.GaussianBlur(image, (3, 3), 0)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowerLimits = np.array([HSV['lH'], HSV['lS'], HSV['lV']])
        upperLimits = np.array([HSV['hH'], HSV['hS'], HSV['hV']])
        thresholded = cv2.inRange(frame_hsv, lowerLimits, upperLimits)

        # Threshold Goal cards
        # Goal cards covered
        HSV = {'lH': 61, 'lS': 0, 'lV': 149, 'hH': 255, 'hS': 255, 'hV': 255}
        lowerLimits = np.array([HSV['lH'], HSV['lS'], HSV['lV']])
        upperLimits = np.array([HSV['hH'], HSV['hS'], HSV['hV']])
        thresholded_goal_covered = cv2.inRange(frame_hsv, lowerLimits, upperLimits)
        thresholded = cv2.bitwise_or(thresholded_goal_covered, thresholded)
        thresholded = cv2.dilate(thresholded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        thresholded = cv2.erode(thresholded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        masked = cv2.bitwise_and(frame, frame, mask=thresholded)
        return thresholded, masked

    def processEdges(self, frame):
        if frame.ndim == 3:
            HSV = {'lH': 0, 'lS': 0, 'lV': 0, 'hH': 255, 'hS': 255, 'hV': 15}
            lowerLimits = np.array([HSV['lH'], HSV['lS'], HSV['lV']])
            upperLimits = np.array([HSV['hH'], HSV['hS'], HSV['hV']])
            thresholded = cv2.inRange(frame, lowerLimits, upperLimits)
        else:
            thresholded = cv2.threshold(frame, 15, 255, cv2.THRESH_BINARY_INV)[1]

        thresholded = self.erode(thresholded, 3)
        thresholded = self.dilate(thresholded, 5)
        return thresholded

    def findGoldCard(self, card_img):
        '''
        Takes in a card image and returns True if it is a gold goal card
        '''
        # frame = cv2.GaussianBlur(image, (3, 3), 0)
        card_img = card_img[2:-2, 3:-3]
        HSV = {'lH': 0, 'lS': 0, 'lV': 0, 'hH': 61, 'hS': 255, 'hV': 255}
        lowerLimits = np.array([HSV['lH'], HSV['lS'], HSV['lV']])
        upperLimits = np.array([HSV['hH'], HSV['hS'], HSV['hV']])
        thresholded_goal = cv2.inRange(card_img, lowerLimits, upperLimits)
        thresholded_goal = self.erode(thresholded_goal, 5)
        thresholded_goal = self.dilate(thresholded_goal, 3)
        contours, _ = cv2.findContours(thresholded_goal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) > 0:
            cont = contours[0]
            print("Area", cv2.contourArea(cont))
            if cv2.contourArea(cont) > 300 and cv2.contourArea(cont) < 1000:
                return True

        return False

    def fixBrightness(self, image):
        '''
        Takes in an image and returns a beautiful image
        '''
        average_intensity = np.mean(image)
        # print(f"Average intensity: {average_intensity}")
        # Define the brightness threshold values

        dark_threshold = 105  # Adjust as needed
        bright_threshold = 130  # Adjust as needed
        lower_dark = 70
        upper_bright = 165

        # Set the gamma value based on average intensity
        if average_intensity < dark_threshold:
            gamma = 0.85
            if average_intensity < lower_dark:
                gamma = 0.5
            # print("Increasing brightness, gamma = ", gamma)
        elif average_intensity > bright_threshold:
            gamma = 1.2  # Adjust as needed for bright images
            if average_intensity > upper_bright:
                gamma = 1.95
            # print("Decreasing brightness")
        else:
            gamma = 1.0  # Default gamma value
            # print("Brightness is OK!")

        # Apply gamma correction
        corrected_image = np.power(image / 255.0, gamma)
        corrected_image = (corrected_image * 255).astype(np.uint8)

        average_intensity = np.mean(corrected_image)
        # print(f"After avg intensity: {average_intensity}")
        return corrected_image

    def extract_image(self, image, card_corners,target_width=70, target_height=100):
        card_corners = np.array(card_corners, dtype=np.float32)

        top_left =  0, 0
        top_right = target_width, 0
        bottom_left = 0, target_height
        bottom_right = target_width, target_height

        out_points = np.array([top_left, top_right, bottom_left, bottom_right], np.float32)

        # Perspective transformation
        transformation_matrix = cv2.getPerspectiveTransform(card_corners, out_points)
        transformed_image = cv2.warpPerspective(image, transformation_matrix, (target_width, target_height))

        return transformed_image