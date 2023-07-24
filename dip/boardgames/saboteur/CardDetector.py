#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dip.boardgames.saboteur.Card import Card
from dip.boardgames.saboteur.ImageProcessor import ImageProcessor
from dip.boardgames.saboteur.utils import *
import cv2
import numpy as np
import time


class CardDetector:
    def __init__(self):
        self.img_processor = ImageProcessor()
        self.board_cards_count = 0
        self.player_cards_count = 0
        self.board_cards: list[Card] = []
        self.player_cards: list[Card] = []
        self.gameStarted = False
        self.start_time = 0
        self.detected_board_img = None
        self.detected_player_img = None
        self.goal_loc = 0 # which half to find the goal cards
        

    def detect(self, image, dist_threshold=10, scale=1, mode="board"):
        """
        Detect cards from given image and return them as cards

        Input:
            image: the image to be processed
        Return:
            cards: the list of cards detected
        """
        
        H, W = image.shape[:2]
        max_ref_contour_area = 8000 
        min_ref_contour_area = 4000
        min_contour_area = (W / (W*scale)) * (H / (H*scale)) * min_ref_contour_area
        max_contour_area = (W / (W*scale)) * (H / (H*scale)) * max_ref_contour_area

        # get copy of image
        if mode == 'player':
            self.detected_player_img = image.copy()
        else:
            self.detected_board_img = image.copy()
        
        image_binary, _= self.img_processor.processImage(image)

        if mode == 'board':  
            # cv2.imshow("Image_binary", image_binary)  
            if len(self.board_cards) == 0 and not self.gameStarted:
                left_half_img = image[:, :W//2]
                right_half_img = image[:, W//2:]
                two_halfs = [left_half_img, right_half_img]
                left_half_img_binary = image_binary[:, :W//2]
                right_half_img_binary = image_binary[:, W//2:]
                for idx, img in enumerate([left_half_img_binary, right_half_img_binary]):

                    conts = cv2.findContours(
                        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    conts = [contour for contour in conts if cv2.contourArea(contour) > min_contour_area and cv2.contourArea(contour) < max_contour_area]
                    
                    if len(conts) == 1:  # Start card
                 
                        cont = conts[0]
                        x, y, w, h = cv2.boundingRect(cont)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        # card_img = two_halfs[idx][y:y+h, x:x+w]
                        card_corners = get_card_corners(cont)
                        card_img = self.img_processor.extract_image(two_halfs[idx], card_corners, target_width=w, target_height=h)
                        start_card = Card("start", card_img, (center_x, center_y))
                        start_card.contour = cont
                        self.board_cards.append(start_card)
                        

                    elif len(conts) == 3:  # Goal cards
                        self.goal_loc = idx 
                        for cont in conts:
                            x, y, w, h = cv2.boundingRect(cont)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            # card_img = two_halfs[idx][y:y+h, x:x+w]
                            card_corners = get_card_corners(cont)
                            card_img = self.img_processor.extract_image(two_halfs[idx], card_corners, target_width=w, target_height=h)
                            # cv2.imshow("img", card_img)
                            # cv2.waitKey(3000)
                            goal_card = Card("goal", card_img,
                                            (center_x+W//2, center_y))
                            goal_card.contour = cont
                            self.board_cards.append(goal_card)
                   
                if len(self.board_cards) == 4:
                    self.gameStarted = True
                    self.start_time = time.time()
        
        if self.gameStarted: 
            start_goal_cardsC:list[Card] = []
            if time.time() - self.start_time >= 3:
                # reset the cards to only contain start and goal cards
                self.board_cards = self.board_cards[:4]
                self.player_cards = []
                self.start_time = time.time()
                
                
                if mode == 'board':
                    start_goal_cardsC = self.board_cards.copy()
                    for idx, card in enumerate(start_goal_cardsC):
                        # update card image
                        h, w = card.image.shape[:2]
                        cont = card.contour
                        contC= cont.copy()
                        
                        if card.type == "goal":
                            contC[:, 0, 0] += W//2
                    
                        card_corners = get_card_corners(contC)
                        card.image = self.img_processor.extract_image(image, card_corners, target_width=w, target_height=h)

            conts = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            conts = [contour for contour in conts if cv2.contourArea(contour) > min_contour_area and cv2.contourArea(contour) < max_contour_area]
            if mode == 'board':
                centers = [(x + (w // 2), y + (h // 2)) for contour in conts for x, y, w, h in [cv2.boundingRect(contour)]]
                start_goal_centers = [(card.x, card.y) for card in self.board_cards if card.type == "start" or card.type == "goal"]
                
                # remove the start and goal contours from the detected contours
                conts = [contour for contour, center in zip(conts, centers) if min(calc_dist(np.array(center), np.array(point)) for point in start_goal_centers) > dist_threshold]
                
            for cont in conts:
                x, y, w, h = cv2.boundingRect(cont)
                center_x = x + w // 2
                center_y = y + h // 2

                
                try:
                    # no two cards will have the same center point
                    if mode == 'player':
                        same = False
                        for card in self.player_cards:
                            if calc_dist(np.array([card.get_cx(), card.get_cy()]), np.array([center_x, center_y])) < dist_threshold:
                                same = True
                                break
                        if not same:
                            card_corners = get_card_corners(cont)
                            card_img = self.img_processor.extract_image(image, card_corners, target_width=w, target_height=h)
                            new_card = Card("path", card_img, (center_x, center_y))
                            new_card.contour = cont
                            self.player_cards.append(new_card)
                    
                    else:
                        #### Board cards ####
                        same = False
                        for card in self.board_cards:
                            if calc_dist(np.array([card.get_cx(), card.get_cy()]), np.array([center_x, center_y])) < dist_threshold:
                                same = True
                                break
                        if not same:
                            card_corners = get_card_corners(cont)
                            card_img = self.img_processor.extract_image(image, card_corners, target_width=w, target_height=h)
                            new_card = Card("path", card_img, (center_x, center_y))
                            new_card.contour = cont
                            self.board_cards.append(new_card)
                except Exception as e:
                    pass
                        
        else:
            print("No cards detected!")
        
        # cv2.waitKey(1)
        return self.board_cards, self.player_cards

    def is_cardFit(self, neighbors:dict, src, offset=10):
        """
        Check if the source image fits into the 4 neighbouring areas.
        The neighbouring areas could be empty or filled with a card.

        Source: The newly detected card image (can be player's or board's)
        Neighbors: The 4 neighbouring cards around the source card.
        They should sorted in the following order:
            [top, bottom, left, right]

        Return:
        True if the the source fits into it's neighbors, False otherwise    
        """
        height, width = src.shape[:2]

        # Works in two iterations. Because we need to rotate card maybe it might fit other way round
        truth_list = []
        truth_dict = {}
        rotate = False
        mean_thresh = 20
        for _ in range(2):
            if rotate:
                src = cv2.rotate(src, cv2.ROTATE_180)
                truth_list = []
                truth_dict = {}

            # edge partition for source image
            src_top = src[0:offset, :]
            src_bottom = src[height - offset:, :]
            src_left = src[:, 0:offset]
            src_right = src[:, width - offset:]
            

            if len(neighbors) > 0:
                for edge, neighbor in neighbors.items():
                    # edge partition for neighbor image
                    if neighbor is not None:
                        if edge == 'top':
                            neighbor_top = neighbor[0:offset, :]
                            truth_list.append(src_bottom.mean() > mean_thresh and neighbor_top.mean() > mean_thresh)
                            truth_dict[edge] = src_bottom.mean() > mean_thresh and neighbor_top.mean() > mean_thresh
                        elif edge == 'bottom':
                            neighbor_bottom = neighbor[height - offset:, :]
                            truth_list.append(src_top.mean() > mean_thresh and neighbor_bottom.mean() > mean_thresh)
                            truth_dict[edge] = src_top.mean() > mean_thresh and neighbor_bottom.mean() > mean_thresh
                        elif edge == 'left':
                            neighbor_left = neighbor[:, width - offset:]
                            truth_list.append(src_right.mean() > mean_thresh and neighbor_left.mean() > mean_thresh)
                            truth_dict[edge] = src_right.mean() > mean_thresh and neighbor_left.mean() > mean_thresh
                        elif edge == 'right':
                            neighbor_right = neighbor[:, 0:offset]
                            truth_list.append(src_left.mean() > mean_thresh and neighbor_right.mean() > mean_thresh)
                            truth_dict[edge] = src_left.mean() > mean_thresh and neighbor_right.mean() > mean_thresh
                    
                if all(truth_list):
                    break
                else:
                    rotate = True
            else:
                return True, 0             

        return all(truth_list), 1 if rotate else 0



    def get_boardCards(self, image_roi: np.ndarray, scale=1) -> list[Card]:
        detected,_ = self.detect(image_roi, scale=scale, mode='board')
        self.board_cards_count = len(detected)
        return detected

    def get_playerCards(self,  image_roi: np.ndarray, scale=1) -> list[Card]:
        """Detects and returns the card from the AI Assistant viewbox."""
        _, detected = self.detect(image_roi, scale=scale, mode='player')
        self.player_cards_count = len(detected)
        return detected

    def get_cardsCount(self,):
        """Returns the number of cards on the board and the number of cards in the player's area"""
        return self.board_cards_count, self.player_cards_count

    def check_orientation(self, card: Card):
        """Checks if the card is placed horizontal or vertical"""

        card_img = card.get_image()
        card_img_thresh, _ = self.img_processor.processImage(card_img)
        card_img_padded = np.pad(card_img_thresh, (2, 2), 'constant')
        contours = cv2.findContours(
            card_img_padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c for c in contours if cv2.contourArea(c) > 500]
        is_Horizontal = None  # None if no contour is found

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if aspect_ratio > 1:
                # contour is "Horizontal"
                is_Horizontal = True
            else:
                # contour is "Vertical"
                is_Horizontal = False

        return is_Horizontal

    def get_startCard(self,) -> Card:
        return [card for card in self.board_cards if card.get_type() == 'start'][0]

    def get_goalCards(self,) -> list[Card]:
        return [card for card in self.board_cards if card.get_type() == 'goal']

    def visualize(self, mode=None, selected_goal:tuple = None):
        """Visualizes the board and the cards on the board"""
        goal_coords = [[(0,0),(2,0), (4, 0)], [(0, 8), (2, 8), (4, 8)]] # left side is not implemented for now. make sure self.goal_loc = 1
        c = 0
        if mode == 'board':
            if self.detected_board_img is not None:   
                for card in self.board_cards:
                    cont = card.contour
                    if card.type == 'goal':
                        self.detected_board_img=drawContour(self.detected_board_img, cont, color_rect=(0, 255, 255) ,pos=self.goal_loc, label=card.get_type())
                        for idx, goal in enumerate(sorted(goal_coords[self.goal_loc], reverse=True)):
                            if selected_goal ==  goal and idx == c:
                                self.detected_board_img=drawContour(self.detected_board_img, cont, color_rect=(235, 255, 0), pos=self.goal_loc, target=True)
                        c+=1
                    elif card.type == 'start':
                        self.detected_board_img = drawContour(self.detected_board_img, cont, color_rect=(0, 255, 255), label=card.get_type())
                    else:
                        self.detected_board_img = drawContour(self.detected_board_img, cont, label=card.get_type())
                        
                    # check orientation
                    if self.check_orientation(card):
                        self.detected_board_img = drawContour(self.detected_board_img, cont, color_rect=(0, 0, 255))
                    

                # cv2.putText(self.detected_board_img, 'Cards No: {}'.format(len(self.board_cards)), (10, 20),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("BoardCards Detection", self.detected_board_img)
                
        elif mode == 'player':
            if self.detected_player_img is not None:
                for card in self.player_cards:
                    cont = card.contour
                    self.detected_player_img = drawContour(self.detected_player_img, cont)
                cv2.putText(self.detected_player_img, 'Cards No: {}'.format(len(self.player_cards)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("PlayerCards Detection", self.detected_player_img)
        else:
            raise Exception("Something went wrong. Invalid mode selected")
        