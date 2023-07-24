#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dip.boardgames.saboteur.Card import Card
from dip.boardgames.saboteur.CardDetector import CardDetector
from dip.boardgames.saboteur.PathPlanner import run_astar
from dip.boardgames.saboteur.Grid import Grid
from dip.boardgames.saboteur.Camera import Camera
from dip.boardgames.saboteur.ImageProcessor import ImageProcessor
from dip.boardgames.saboteur.GUI import GUI


import numpy as np
import cv2
import time
import random


class Game:
    def __init__(self, loop_rate=5 ) -> None:

        # self.n_players = n_players
        # self.n_rounds = n_rounds
        self.camera = Camera()
        self.imageProcessor = ImageProcessor()
        self.gui = GUI()
        self.detector = CardDetector()
        self.grid = Grid(dim={'rows': 5, 'cols': 9}, cell={
                         'cell_width': 40, 'cell_height': 65})
        self.best_move = None
        self.best_idx = None
        self.frame_cv2 = None
        self.suggested_cells = []
        self.start_pos_pix = None
        self.loop_rate = loop_rate

    def validate_move(self, map_binary:np.ndarray, player_card_img:np.ndarray, coord:tuple) -> bool:
        '''
        Validate the supposed move. This checks if the player's card connects to the start card. If not, False is returned.
        '''
        # insert player_card_img into the grid at the given coordinates
        r, c = coord
        map_binaryC = map_binary.copy()
        start_x = c * self.grid.cell_width
        start_y = r * self.grid.cell_height

        map_binaryC[start_y:start_y+self.grid.cell_height, start_x:start_x+self.grid.cell_width] = player_card_img
        # cv2.imshow("Before map with playercard", map_binaryC)

        map_binaryC = self.imageProcessor.dilate(map_binaryC, 3)
        map_binaryC = self.imageProcessor.erode(map_binaryC, 3)
        goal_pos_pix = self.grid.getGoal(image=map_binaryC, coord = coord)


        path_found = run_astar(None, map_binaryC, self.start_pos_pix, goal_pos_pix )

        if not path_found:
            print("Card does not connect to start card")
        if path_found is not None:
            return path_found
        else:
            return False


    def suggest_move(self, map_binary: np.ndarray, playable_cells: list[tuple], player_cards: list[Card]) -> tuple[list[Card], list[tuple]]:
        """
        Suggest a move for the player.

        Input:
            map_binary: the map image 
            playable_cells: list of cells (center coordinates) where a player can play a suggested card
            player_cards: list of cards that the player has
        Output:
            a list of cards that the player can play, empty list if no move can be made
        """
        imageC = cv2.cvtColor(map_binary, cv2.COLOR_GRAY2BGR)
        suggested_cards : list[tuple[Card, int, bool]]= []
        # loc = []
        is_card_suitable = False
        offset = 3  # remove some pixels from the edges of the player card
        # Loop through the board cards
        self.suggested_cells = []
        temp = []
        for empty_cell_idx, empty_cell in enumerate(playable_cells):
            x = empty_cell[1] * self.grid.cell_width
            y = empty_cell[0] * self.grid.cell_height
            imageC[y:y+self.grid.cell_height, x:x+self.grid.cell_width] = (0, 0, 255)

            # loop through the player cards and find all suitable cards
            for idx, player_card in enumerate(player_cards):
                # quick image processing for the player card
                player_card_img = player_card.get_image()
                player_card_img = player_card_img[offset:-offset, offset:-offset]
                player_card_img = cv2.resize(player_card_img, (self.grid.cell_width, self.grid.cell_height))
                player_card_img = self.imageProcessor.fixBrightness(player_card_img)
                player_card_img, _ = self.imageProcessor.processMap(player_card_img)

                # find the neighbors of the empty cell
                empty_cell_neighbors, empty_cell_neighbor_coords = self.grid.get_neighbors(empty_cell[0], empty_cell[1])

                # extract empty cell neighbor images
                cell_neighbor_images = {}
                for side, neighbor in empty_cell_neighbors.items():
                    coord = empty_cell_neighbor_coords[side]
                    # filter out out of bounds cells.
                    if coord[0] < 0 or coord[1] < 0 or coord[0] >= self.grid.rows or coord[1] >= self.grid.cols:
                        continue

                    if neighbor is not None:
                        cell_neighbor_images[side] = self.grid.extract_cell(
                            map_binary, self.grid.cell_centers[coord[0]][coord[1]])


                    x = coord[1] * self.grid.cell_width
                    y = coord[0] * self.grid.cell_height
                    imageC[y:y+self.grid.cell_height , x:x+self.grid.cell_width] = (0, 255, 0)

                is_card_suitable, is_rotate = self.detector.is_cardFit(cell_neighbor_images, player_card_img)
                if not is_card_suitable:
                    #     cv2.imshow("notsuitable", player_card_img)
                    #     print("notsuitable @", empty_cell_idx)
                    # cv2.waitKey(0)
                    continue
                else:

                    # Validate the card using Astar
                    isValid = self.validate_move(map_binary=map_binary, player_card_img=player_card_img, coord=empty_cell)

                    if isValid:
                        # append the player card to the suggested cards
                        card_info : tuple[Card, int, bool] = (player_card, empty_cell_idx, is_rotate)

                        suggested_cards.append(card_info)
                        # temp.append([idx, empty_cell_idx])

                        # append the coordinates of the empty cell to the suggested cells
                        if empty_cell not in self.suggested_cells:
                            self.suggested_cells.append(empty_cell)

        return suggested_cards , self.suggested_cells

    def findBestMove(self, suggested_cells: list[tuple], goal_coord: tuple) -> tuple:
        """
        Get the best move for the player by computing the Manhattan distance between the suggested cards and goal card
        Input:
            suggested_cards: list of cards that the player can play
        Output:
            a card that the player can play
        """

        dists = self.grid.computeDistances(goal_coord)
        # print(f"dists {len(dists)}")
        shortest_dist = np.inf
        best_move = None
        # best_idx = None
        for idx, cell in enumerate(suggested_cells):
            # print(cell)
            dist = dists[cell[0]][cell[1]]
            if dist < shortest_dist:
                shortest_dist = dist
                best_move = cell
                # best_idx = idx

        # print(f"Best move: {best_move}, dist: {shortest_dist}")
        # print(f"Best idx: {best_idx}")

        self.best_move = best_move

    def get_playable_cells(self, grid_arr: list[list[Card]], grid_img: np.ndarray) -> list[tuple]:
        """
        Get the playable cells for the player.
        Input:
            grid_arr: list of lists of cards
            grid_img: numpy array of the grid image
        Output:
            a list of cells (center coordinates) where a player can play a suggested card
        """

        playable_cells = []

        # loop through the cells
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):

                # check if the cell is empty
                if grid_arr[row][col] is not None:

                    # skip 'goal' cards from grid array
                    if grid_arr[row][col].type == 'goal':
                        continue

                    # print(f"row: {row}, col: {col}")
                    neighbors = self.grid.get_neighbors(row, col)

                    # Get the neighbors
                    neighbors, neighbors_idx = self.grid.get_neighbors(
                        row, col)
                    for side, neighbor in neighbors.items():
                        is_empty = False
                        coord = neighbors_idx[side]

                        # filter out out-of-bounds cells. Because, the grid is created with padded cells
                        if coord[0] < 0 or coord[1] < 0 or coord[0] >= self.grid.rows or coord[1] >= self.grid.cols:
                            continue

                        if neighbor is not None:
                            neighbor_card_img = self.grid.extract_cell(
                                grid_img, self.grid.cell_centers[coord[0]][coord[1]])
                            is_empty = np.mean(neighbor_card_img) == 0
                            # cv2.imshow(side, neighbor_card_img)
                            # if cv2.waitKey(100) == 27:
                            #     break
                        else:
                            is_empty = True
                        # print(f"row: {row}, col: {col}, side: {side}, coord: {coord}, is_empty: {is_empty}")

                        if is_empty and (coord not in playable_cells):
                            playable_cells.append(coord)
        # sort it
        playable_cells.sort(key=lambda cell: (cell[0], cell[1]))
        return playable_cells

    def visualize_moves(self, image, playable_cells: list[tuple], suggested_cards : list[tuple[Card, int, bool]], mode="board"):
        """
        Visualize the moves that can be made using a number of colors and labels.

        Input:
            playable_cells: list of cells (center coordinates) where a player can play a suggested card
            suggested_cards: list of cards that the player can play from player deck
        Output:
            None
        """

        # highlight the playable cells

        cell_width = self.grid.cell_width
        cell_height = self.grid.cell_height

        imageC = image.copy()

        if mode == "board":
            # Add playable cells to the resized output
            for c_idx, cell in enumerate(playable_cells):
                x = cell[1] * cell_width
                y = cell[0] * cell_height
                # imageC[y:y+cell_height, x:x+cell_width] = (0, 255, 0)  # Mark the cell with green color (RGB)
                cv2.rectangle(imageC, (x, y), (x + int(cell_width), y + int(cell_height)), (0, 255, 0), 2)  # Green bounding box with thickness 2
                # cv2.line(imageC, (x, y), (x + cell_width -30, y + cell_height-30), (0, 255, 0), 2)
                # cv2.line(imageC, (x + cell_width-30, y),(x, y + cell_height-30), (0, 255, 0), 2)

                # Write index at the center of the cell
                text = str(c_idx)
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = x + (cell_width - text_size[0]) // 2
                text_y = y + (cell_height + text_size[1]) // 2
                # if c_idx == 12:
                # print("look here.................................\n\n\n")
                if self.best_move == (cell[0], cell[1]):
                    # cv2.putText(imageC, "BM", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(imageC, (x, y), (x + int(cell_width), y + int(cell_height)), (255, 0, 0), 2)
                    self.best_idx = c_idx
                # else:
                cv2.putText(imageC, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('game_MOVES', imageC)
            self.grid

        if mode == "player":

            # # show the card label
            temp_dict = {}
            temp_rotate_dict = {}
            for card_info in suggested_cards:
                card = card_info[0]
                cell_no = card_info[1]
                is_rotate = card_info[2]
                center_x = card.get_cx()
                center_y = card.get_cy()
                card_img = card.get_image()

                if (center_x, center_y) in temp_dict:
                    # Append the cell_no to the existing list
                    temp_dict[(center_x, center_y)].append(cell_no)
                    selected_choice = random.sample(list(temp_dict[(center_x, center_y)]), min(len(temp_dict[(center_x, center_y)]), 3))
                    temp_dict[(center_x, center_y)] = selected_choice
                    temp_rotate_dict[(center_x, center_y)] = is_rotate
                else:
                    # Create a new list with the cell_no
                    temp_dict[(center_x, center_y)] = [cell_no]
                    temp_rotate_dict[(center_x, center_y)] = is_rotate

            # Draw bounding boxes and write numbers inside the boxes for dictionary centers
            for center, numbers in temp_dict.items():
                x = center[0]
                y = center[1]

                c_start_x = x - card_img.shape[1] // 2
                c_start_y = y - card_img.shape[0] // 2
                cv2.rectangle(imageC, (c_start_x, c_start_y), (c_start_x +card_img.shape[1], c_start_y + card_img.shape[0]), (0, 255, 0), thickness=2)
                if temp_rotate_dict[(x, y)]:
                    cv2.putText(imageC, " R", (c_start_x-5, c_start_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                if self.best_idx is not None :
                    if self.best_idx in numbers:
                        cv2.rectangle(imageC, (c_start_x, c_start_y), (c_start_x +card_img.shape[1], c_start_y + card_img.shape[0]), (255, 0, 0), thickness=2)
                        # cv2.putText(imageC, "Best Move", (c_start_x, c_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Write the numbers in the current cell with the desired spacing
                for number in numbers:
                    cv2.putText(imageC, str(number), (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    y += 30  # Increase the y-coordinate for the next number within the cell


            imageC = cv2.resize(imageC, (imageC.shape[1] * 2, imageC.shape[0] * 2))


            cv2.imshow('player_MOVES', imageC)



    def check_gameStatus(self, grid_arr:list[list[Card]],  grid_img, map ) -> bool:
        # TODO: Checks the current status of the game. if the game is over, return a string. Currently not working
        """If there is a connected path to the goal return True, else return False"""
        # astar = run_astar(self.frame_cv2, map, start_pos, goal_pos)
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # check if the cell is not empty
                if grid_arr[row][col] is not None:
                    # Find card with gold
                    if grid_arr[row][col].type == 'goal':
                        card = grid_arr[row][col]
                        card_img = card.get_image()
                        isGold = self.imageProcessor.findGoldCard(card_img)

                        print("Gold: ", isGold)
        return False



    def start(self, cam_device_id=0):
        """
        Starts the game
        Input:
            cam_device_id: The id of the camera. To be used by cv2.VideoCapture. 
        """
        global lock

        global map_binary, board_area, start_pos
        gui = GUI()

        # Camera
        cap = cv2.VideoCapture(cam_device_id)
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

        _, image = cap.read()
        # Assuming this is the image
        # Get game regions
        print("Getting board area...")
        player_rect = (80, 380, 300, 280)
        board_rect = (440, 50, 750, 650) # x, y, w, h
        # extract region from image
        board_area = image[board_rect[1]:board_rect[1] +
                           board_rect[3], board_rect[0]:board_rect[0]+board_rect[2]]
        player_area = image[player_rect[1]:player_rect[1] +
                            player_rect[3], player_rect[0]:player_rect[0]+player_rect[2]]
        cv2.imshow('player',player_area)
        cv2.imshow('board', board_area)
        cv2.waitKey(0)
        # Scale to resize the frames
        resize_scale = 1

        board_area = cv2.resize(
            board_area, (board_area.shape[1]//resize_scale, board_area.shape[0]//resize_scale))
        player_area = cv2.resize(
            player_area, (player_area.shape[1]//resize_scale, player_area.shape[0]//resize_scale))


        grid_img = np.zeros((self.grid.cell_width*self.grid.rows,
                            self.grid.cell_height*self.grid.cols, 3), dtype=np.uint8)
        start_time = time.time()
        start_time_FPS = time.time()
        frame_count = 0
        isGameStart = False
        showDebug = True




        gui_text = {"FPS": 0, "Player Cards": 0,
                    "Board Cards": 0, "Playable Cells": 0}
        playable_cells = []

        print("Entering game loop")
        lock = False
        start_pos = None
        goal_pos = None
        self.start_pos_pix = None
        goal_pos_pix = None
        map_binary = None


        while True:
            gui.set_game_area(self.detector.detected_board_img)
            gui.set_player_area(player_area)
            gui.draw(gui_text=gui_text)

            # get the frame
            ret, frame = cap.read()
            board_area = frame[board_rect[1]:board_rect[1] +
                               board_rect[3], board_rect[0]:board_rect[0]+board_rect[2]]
            player_area = frame[player_rect[1]:player_rect[1] +
                                player_rect[3], player_rect[0]:player_rect[0]+player_rect[2]]

            # resize the frames
            board_area = cv2.resize(
                board_area, (board_area.shape[1]//resize_scale, board_area.shape[0]//resize_scale))
            player_area = cv2.resize(
                player_area, (player_area.shape[1]//resize_scale, player_area.shape[0]//resize_scale))

            # detect the cards
            board_cards = self.detector.get_boardCards(board_area, resize_scale)
            player_cards = self.detector.get_playerCards(player_area, resize_scale)

            # generate grid array
            grid_arr = self.grid.getArray(board_area, board_cards)

            # update frame
            self.frame_cv2 = board_area

            if not isGameStart:
                print("Starting game")
                start_card = self.detector.get_startCard()
                start_pos = self.grid.getCardCoord(start_card)

                print(f"start_card: {start_pos}")

                # get goal cards positions
                goal_card = self.detector.get_goalCards()[0]
                goal_pos = self.grid.getCardCoord(goal_card)
                print(f"Selected goal_card: {goal_pos}")



                isGameStart = True

            # Now We will be ready to suggest moves
            if isGameStart:

                if time.time() - start_time > self.loop_rate:
                    start_time = time.time()
                    grid_img = self.grid.drawCards(grid_arr)
                    cv2.imshow('Grid', grid_img)

                    # suggest moves to reach the goal card
                    playable_cells = self.get_playable_cells(
                        grid_arr=grid_arr, grid_img=grid_img)

                    # get map for path planning
                    map_binary, map_rgb = self.imageProcessor.processMap(
                        grid_img, thresh_val=50)
                    cv2.imshow('Map', map_rgb)

                    # Get the start coordinates in pixel location. Will be used for path planning
                    if not lock:
                        self.start_pos_pix = self.grid.getGoal(image=map_binary, coord=start_pos)
                        lock = True

                    # get the moves
                    selected_cards, suggested_cells = self.suggest_move(
                        map_binary, playable_cells, player_cards)

                    # find the best move
                    self.findBestMove(suggested_cells, goal_pos)

                    # visualize the moves
                    self.visualize_moves(grid_img, playable_cells,  selected_cards, mode='board')
                    self.visualize_moves(player_area, playable_cells, selected_cards, mode='player')

                    # Path planner verification. This is for every card generally placed on the board while game is in session
                    coords = self.grid.getGridCoords()
                    for cell in coords:
                        if cell == start_pos:
                            continue
                        goal_pos_pix = self.grid.getGoal(image=map_binary, coord = cell)
                        run_astar(grid_img, map_binary, self.start_pos_pix, goal_pos_pix)

                    # self.check_gameStatus(grid_arr=grid_arr, grid_img=grid_img, map=map_binary)

            # Show FPS
            frame_count += 1
            end_time_FPS = time.time()
            total_time = end_time_FPS - start_time_FPS
            fps = frame_count / total_time

            # Update GUI
            gui_text["FPS"] = fps
            gui_text["Player Cards"] = len(player_cards)
            gui_text["Board Cards"] = len(board_cards)
            gui_text["Playable Cells"] = len(playable_cells)

            if showDebug:
                self.detector.visualize(mode='board', selected_goal=goal_pos)
                self.detector.visualize(mode='player')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cv2.destroyAllWindows()
        cap.release()