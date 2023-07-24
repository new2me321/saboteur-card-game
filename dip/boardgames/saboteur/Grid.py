#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dip.boardgames.saboteur.ImageProcessor import ImageProcessor
from dip.boardgames.saboteur.Card import Card
from dip.boardgames.saboteur.utils import *
import numpy as np
import cv2


class Grid:
    def __init__(self, dim={'rows': 5, 'cols': 9}, cell={'cell_width': 60, 'cell_height': 90}, region={'top': 0, 'left': 0}):
        self.rows = dim['rows']
        self.cols = dim['cols']
        self.cell_width = cell['cell_width']
        self.cell_height = cell['cell_height']
        self.top, self.left = region['top'], region['left']
        self.cell_centers = [[None for j in range(
            self.cols)] for i in range(self.rows)]
        self.imageProcessor = ImageProcessor()
        self.image = None
        self.array: list[list[Card]] = [[]]
        self.coordinates = []

    def getArray(self, image: np.ndarray, cards: list[Card]) -> list[list[Card]]:
        '''
        Generates a grid of 5x9 from the list of cards 
        '''
        H, W = image.shape[:2]

        # print(f"Height: {H}, Width: {W}")
        # print(f"Rows: {self.rows}, Cols: {self.cols}")
        grid = [[None for j in range(self.cols)] for i in range(self.rows)]
        for card in cards:
            cell_row = int(card.y / (H / self.rows))
            cell_col = int(card.x / (W / self.cols))

            # print(f"Card {card.get_type()} at {cx, cy}")
            # print(f"Grid {card.get_type(), cell_row, cell_col}")
            grid[cell_row][cell_col] = card
        # print(grid)
        self.array = grid
        return grid

    def draw(self, ):
        """Draws a grid on the image frame"""

        # Calculate total image dimensions
        image_width = self.cols * self.cell_width
        image_height = self.rows * self.cell_height
        res = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):

                # calc start coord of a cell
                x = self.left + col * self.cell_width
                y = self.top + row * self.cell_height

                # calc the center coordinates of the cell
                center_x = (x + x + self.cell_width) // 2
                center_y = (y + y + self.cell_height) // 2

                # add to list of cell centers
                self.cell_centers[row][col] = (center_x, center_y)


        return res

    def drawCards(self, grid_arr: list[list[Card]]) -> np.ndarray:
        res = self.draw()
        offset = 1  # remove some pixels from the edges
        for row in range(self.rows):
            for col in range(self.cols):
                card = grid_arr[row][col]
                if card is not None:
                    x = self.left + col * self.cell_width
                    y = self.top + row * self.cell_height
                    card_img = card.get_image()
                    card_img = card_img[offset:-offset, offset:-offset]
                    card_img = self.imageProcessor.fixBrightness(card_img)

                    # Resize the card image
                    res[y:y+self.cell_height, x:x+self.cell_width] = cv2.resize(
                        card_img, (self.cell_width, self.cell_height))

                    # cv2.imshow("grid", res)
                    # cv2.waitKey(300)
        return res

    def get_neighbors(self, cell_row, cell_col) -> tuple[dict[str, Card], dict[str, tuple]]:
        """
        Returns the 4-neighbors surrounding a card 
        Input:
            cell_row: row of the cell
            cell_col: column of the cell
        Output:
            neighbors: dict of the 4 neighbors surrounding the cell
            neighbors_idx: dict of the indexes of the neighbors

        """
        if self.array is not None:
            pad_width = 1
            grid = np.pad(self.array, pad_width=pad_width,
                          constant_values=None)
            row_index = cell_row + pad_width
            col_index = cell_col + pad_width

            neighbors = {
                'top': grid[row_index - 1][col_index],
                'left': grid[row_index][col_index - 1],
                'right': grid[row_index][col_index + 1],
                'bottom': grid[row_index + 1][col_index],
            }
            neighbors_idx = {'top': (cell_row - 1, cell_col), 'left': (cell_row, cell_col - 1),
                             'right': (cell_row, cell_col + 1), 'bottom': (cell_row + 1, cell_col)}
            return neighbors, neighbors_idx

        else:
            raise Exception("Grid is not initialized")

    def get_cellCenters(self):
        """Returns the center of the grid cells as a list of coordinate points (x,y)"""
        return self.cell_centers

    def getGridCoords(self):
        """returns the r, c coordinates of a cell in the grid"""
        return [(row, col) for row in range(self.rows) for col in range(self.cols)]

    def extract_cell(self, image, coord: tuple) -> np.ndarray:
        """ Extracts the image at the cell given the center coordinates of that cell"""

        center_x, center_y = coord
        # Calculate the top-left corner of the cell
        x1 = center_x - self.cell_width // 2
        y1 = center_y - self.cell_height // 2

        # Calculate the bottom-right corner of the cell
        x2 = x1 + self.cell_width
        y2 = y1 + self.cell_height

        # Extract the cell from the image
        cell = image[y1:y2, x1:x2]

        return cell

    def getGoal(self, image: np.ndarray, coord: tuple) -> np.ndarray | None:

        card_cx, card_cy = self.cell_centers[coord[0]][coord[1]]

        # Define the kernel window dimensions
        w_size_x = 30
        w_size_y = int(1.1*(self.cell_height/self.cell_width)*w_size_x)

        # Generate random points within the cell image
        num_points = 50  # Number of random points to generate
        random_x = np.random.rand(num_points) * (w_size_x - 2)
        random_y = np.random.rand(num_points) * (w_size_y-2)
        random_points = np.column_stack((random_x, random_y)).astype(int)

        # Calculate the kernel start coordinates in the grid image
        kernel_start_x_wrt_grid = card_cx - (w_size_x + self.cell_width) // 4
        kernel_start_y_wrt_grid = card_cy - \
            (w_size_y + self.cell_height//2) // 4
        # Calculate the points' coordinates in reference to the grid image
        gpoints_x = random_points[:, 0] + kernel_start_x_wrt_grid
        gpoints_y = random_points[:, 1] + kernel_start_y_wrt_grid
        grandom_points = np.column_stack((gpoints_x, gpoints_y)).astype(int)
        # print(random_points)
        imageC = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(num_points):
            cv2.circle(
                imageC, (grandom_points[i, 0], grandom_points[i, 1]), 2, (0, 0, 255), -1)

        mask = image[grandom_points[:, 1], grandom_points[:, 0]] >= 1

        indices = np.where(mask)[0]
        positive_points = np.array(grandom_points[indices])
        # print("Len of positive hits", len(positive_points))
        if len(positive_points) == 0:
            return None

        selected = positive_points[np.random.randint(len(
            positive_points))]  # positive_points[np.argmin(positive_points[:, 1])]

        return selected

    def getCardCoord(self, card: Card) -> tuple:
        """
        Returns the coordinates of the cell that the card is in. Only for Start and Goal Cards
        """

        if card.get_type() == 'path':
            raise Exception("Path Cards are not supported")

        goals = []

        for row in range(self.rows):
            for col in range(self.cols):
                if self.array[row][col] is not None:
                    if card.get_type() == 'start' and card.get_type() == self.array[row][col].get_type():
                        return (row, col)
                    elif card.get_type() == 'goal' and card.get_type() == self.array[row][col].get_type():
                        goals.append((row, col))

        if len(goals) == 0:
            return None
        elif len(goals) == 3:
            return goals[np.random.randint(len(goals))]  # pick a random goal
        elif len(goals) < 3:
            raise Exception("Not enough goal cards detected!")

        return None

    def computeDistances(self, goal: tuple) -> list:
        """
        Computes the manhattan distances from cells to the given goal"""
        return [[abs(col - goal[1]) + abs(row - goal[0]) for col in range(self.cols)] for row in range(self.rows)]
