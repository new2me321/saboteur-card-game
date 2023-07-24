#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dip.boardgames.saboteur.ImageProcessor import ImageProcessor
import cv2
import time

class Camera:
    def __init__(self, device=0) -> None:
        self.calib_pos = list({"left_top": (0, 0), "right_top": (
            0, 0), "left_bottom": (0, 0), "right_bottom": (0, 0)}.items())
        self.counter = 0
        self.calib_pos_is_done = False
        self.imageProcessor = ImageProcessor()
        self.start_time = time.time()
        self.device = device


    def setup(self, ) -> None:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise Exception("Camera not opened")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # turn off auto exposure
            cap.set(cv2.CAP_PROP_AUTO_WB, 0) # turn off white balance
            cap.set(cv2.CAP_PROP_EXPOSURE, 0.5) # set exposure
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn off auto focus
            print("Camera is running")
            self.running = True

        while True:
            _, self.frame = cap.read()
            cv2.imshow("Camera", self.frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite("saboteur.jpg", self.frame)
                break
        
        cv2.destroyAllWindows()
        cap.release()
