#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class GUI:
    def __init__(self):
        self.game_area = None
        self.player_area = None
        self.text_area = None
        
    def set_player_area(self, player_area):
        self.player_area = player_area

    def set_game_area(self, game_area):
        self.game_area = game_area
    
    def display_text(self, gui_text = {"FPS": "", "Player Cards": "", "Board Cards": "", "Playable Cells": ""}):
        image_width = self.text_area.shape[1]
        offset = 300  # Number of pixels to offset from the right edge
        
        # Define the size of the rectangles
        rectangle_height = 50
        rectangle_width = 50
        rectangle_offset = 10
        
        rectangles = [
            ((image_width - offset - rectangle_width, rectangle_offset), (image_width - offset, rectangle_offset + rectangle_height), (0, 0, 255)),  # Wrong Card (Red)
            ((image_width - offset - rectangle_width, 2*rectangle_height), (image_width - offset,  3*rectangle_height), (0, 255, 0)),  # Suggested Cards (Green)
            ((image_width - offset - rectangle_width, 4*rectangle_height), (image_width - offset, 5* rectangle_height), (255, 0, 0))  # Best Card (Blue)
        ]
        
        # Draw rectangles and add text below
        for (start, end, color), text in zip(rectangles, ["Wrong Card", "Suggested Cards", "Best Card"]):
            cv2.rectangle(self.text_area, start, end, color, 2)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = end[0] + 10  # Right side of the rectangle with a small padding
            text_y = start[1] + (end[1] - start[1]) // 2 + text_height // 2
            cv2.putText(self.text_area, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        fps = gui_text["FPS"]
        player_cards_no = gui_text["Player Cards"]
        board_cards_no = gui_text["Board Cards"]
        playable_cells_no = gui_text["Playable Cells"]
        
        text_offset = 20
        # Write FPS
        cv2.putText(self.text_area, f"FPS: {int(fps)}", (10, text_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Game status update
        cv2.putText(self.text_area, f"Player Cards:  {player_cards_no}", (10, 7*text_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.text_area, f"Board Cards:  {board_cards_no}", (10, 8*text_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.text_area, f"Playable Cells:  {playable_cells_no}", (10, 9*text_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

    def draw(self, height=960, width=1024, offset_H=10, offset_W=10, gui_text = {"FPS": "", "Round": "", "Player Cards": "", "Board Cards": "", "Playable Cells": ""}):
        
        # img = np.zeros((height, width, 3), dtype=np.uint8)
        img = np.tile(np.array([65, 65, 65]), (height, width, 1)).astype(np.uint8)
        
        # ROI parameters
        g_W = width-(2*offset_W)
        g_H =  int(0.7*height)
        
        p_W = int(0.6*width)
        p_H = height-g_H

        # get shapes
        _game_area = img[offset_H: g_H - offset_H, offset_W:offset_W + g_W]
        _player_area = img[g_H: g_H + p_H - offset_H, p_W: int(width - 2*offset_W) + offset_W]

        # Draw images on the screen
        # game_area
        img[offset_H:g_H - offset_H, offset_W:offset_W + g_W] = cv2.resize(
            self.game_area, (_game_area.shape[1], _game_area.shape[0])) if self.game_area is not None else [0, 120, 35]

        # player area
        img[g_H: g_H + p_H - offset_H, p_W: int(width - 2*offset_W) + offset_W] = cv2.resize(
            self.player_area, (_player_area.shape[1], _player_area.shape[0])) if self.player_area is not None else [255, 20, 0]

        # Text area
        img[g_H: g_H + p_H - offset_H, offset_W: p_W - offset_W] = [0, 0, 0]
        self.text_area = img[g_H: g_H + p_H - offset_H, offset_W: p_W - offset_W]
        self.display_text(gui_text)
        cv2.imshow('GUI', img)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0
        
  