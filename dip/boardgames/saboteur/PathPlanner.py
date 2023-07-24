#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import math
import cv2


class Astar:
    """
    A-star algorithm for finding the shortest path from start to goal.

    Parameters
    ----------
    start : array_like
        The starting node of the path.
    goal : array_like
        The goal node of the path.
    grid : array_like
        The map image of the grid
    """

    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.alpha = 5  # the heuristic term
        self.img_grid = grid
        self.path_found = None
        self.path = None

    def calc_distance(self, pt1, pt2):
        # print("Calc distance", pt1, pt2)
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def compute_heuristic_cost(self, node, goal):
        """
        Compute the cost from the current node to the goal
        """
        heuristic = self.alpha * math.sqrt(
            (node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2
        )
        return heuristic

    def get_8_neighbourhood(self, x: int, y: int):
        """
        Get the 8 neighbours of the current node
        """

        p_0 = (x - 1, y - 1)
        p_1 = (x - 1, y)
        p_2 = (x - 1, y + 1)
        p_3 = (x, y - 1)
        p_4 = (x, y + 1)
        p_5 = (x + 1, y - 1)
        p_6 = (x + 1, y)
        p_7 = (x + 1, y + 1)

        points = [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7]

        return points

    def get_path(
        self,
    ):
        """
        The main function of A* algorithm.

            F = G + H

            F:= total cost

            G:= the cost from the current node to the start

            H:= the estimate cost from the current node to the goal
            
        Returns:
            path : The list of nodes from the start node to the goal node
            bool: True if a path is found, False otherwise
        """
        start_time = time.time()

        # initialize both open and closed lists
        open_list = [tuple(self.start)]
        closed_list = []  # contains all the points in the graph that were visited

        # dictionaries to store F and G scores
        F_scores = {}  # total cost from start to goal
        G_scores = {}  # total cost from start node to current node

        # store scores for start node
        G_scores[tuple(self.start)] = 0
        F_scores[tuple(self.start)] = self.compute_heuristic_cost(
            self.start, self.goal)
        # print("Initial distance", self.calc_distance(self.start, self.goal))
        # path from goal to start
        path = {}
        path[tuple(self.start)] = None

        shortest_path = []
        c = 0
        while len(open_list) != 0:
            c += 1
            # find the highest priority node i.e node with lowest F score
            current_idx = np.argmin([F_scores[node] for node in open_list])
            current_node = open_list[current_idx]

            # add the current node closed list (i.e visited_nodes)
            closed_list.append(current_node)

            # UNCOMMENT TO VISUALIZE
            # if c != 1:
            #     visualize(self.img_grid, current_node)

            # remove the current node from the open list
            open_list = list(filter(lambda x: x != current_node, open_list))

            if current_node == tuple(self.goal) or self.calc_distance(current_node, self.goal) <= 1.5:
                # print("Goal found!")

                if self.calc_distance(current_node, self.goal) != 0:
                    # add the goal node to the path
                    path[tuple(self.goal)] = current_node

                # print("Distance", self.calc_distance(current_node, self.goal))
                shortest_path = self.reconstruct_path(path)

                break
            try:
                next_nodes = self.get_8_neighbourhood(
                    current_node[0], current_node[1])
            except KeyError:
                continue
            else:
                try:
                    for neighbor in next_nodes:
                        if neighbor in closed_list:
                            continue

                        if self.grid[neighbor[1], neighbor[0]] == 0:
                            # print("black")
                            continue

                        new_g_score = G_scores[current_node] + self.calc_distance(
                            current_node, neighbor
                        )

                        if neighbor not in open_list:
                            open_list.append(neighbor)

                        if neighbor not in F_scores or new_g_score < G_scores[neighbor]:
                            G_scores[neighbor] = new_g_score
                            F_scores[neighbor] = G_scores[neighbor] + \
                                self.compute_heuristic_cost(
                                    neighbor, self.goal)
                            path[tuple(neighbor)] = current_node
                except:
                    return False
        end_time = time.time()
        elapsed_time = end_time - start_time

        if len(shortest_path) > 1:
            # print("Shortest path found!")
            # print(
            #     "A-star algorithm for finding the shortest path took {} seconds".format(
            #         elapsed_time
            #     )
            # )
            self.path_found = True
            self.path = shortest_path
            return shortest_path
        self.path_found = False
        return False

    def reconstruct_path(self, visited):
        shortest_path = []
        current_node = tuple(self.goal)

        while current_node is not None:
            shortest_path.append(current_node)
            current_node = visited[tuple(current_node)]

        # Reverse the list to get the shortest path from the goal to the start node
        shortest_path = shortest_path[::-1]
        shortest_path = [np.array(point) for point in shortest_path]

        return shortest_path


def visualize(img, current_node):
    cv2.circle(img, (current_node[0], current_node[1]), 1, (0, 255, 0), -1,)
    cv2.imshow("image", img)
    cv2.waitKey(1)


def smooth_path(path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
    '''
    Smooths the path
    '''
    smoothed_path = path.copy()
    change = tolerance

    while change >= tolerance:
        change = 0.0

        for i in range(1, len(path) - 1):
            for j in range(len(path[i])):
                previous_value = smoothed_path[i][j]
                smoothed_path[i][j] += weight_data * \
                    (path[i][j] - smoothed_path[i][j])
                smoothed_path[i][j] += weight_smooth * (
                    smoothed_path[i - 1][j]
                    + smoothed_path[i + 1][j]
                    - (2.0 * smoothed_path[i][j])
                )
                change += abs(previous_value - smoothed_path[i][j])

    return np.array(smoothed_path).astype(int)


def resize_path(path, original_size, resized_size):
    '''
    Resizes the path to fit the original size such that the path fits the original image. 
    '''
    
    scale_factor_x = original_size[1] / resized_size[1]
    scale_factor_y = original_size[0] / resized_size[0]

    resized_x = np.array(path)[:, 0] * scale_factor_x
    resized_y = np.array(path)[:, 1] * scale_factor_y

    frac_x = resized_x - np.floor(resized_x)
    frac_y = resized_y - np.floor(resized_y)

    top_left = np.floor(resized_x).astype(int), np.floor(resized_y).astype(int)
    top_right = np.ceil(resized_x).astype(int), np.floor(resized_y).astype(int)
    bottom_left = np.floor(resized_x).astype(
        int), np.ceil(resized_y).astype(int)
    bottom_right = np.ceil(resized_x).astype(
        int), np.ceil(resized_y).astype(int)

    interpolated_x = (1 - frac_y) * (
        (1 - frac_x) * top_left[0] + frac_x * top_right[0]
    ) + frac_y * ((1 - frac_x) * bottom_left[0] + frac_x * bottom_right[0])
    interpolated_y = (1 - frac_y) * (
        (1 - frac_x) * top_left[1] + frac_x * top_right[1]
    ) + frac_y * ((1 - frac_x) * bottom_left[1] + frac_x * bottom_right[1])

    resized_path = np.column_stack((interpolated_x, interpolated_y))

    return resized_path


def dist_to_all(points1, points2):
    '''
    Computes the distance between all points in points1 and points2.
    '''
    squared_diff = np.square(points1[:, np.newaxis] - points2)
    squared_distances = np.sum(squared_diff, axis=2)
    distances = np.sqrt(squared_distances)
    return distances


def draw_path(path, frame_cv2):
    global img_resize
    # Plot the path
    if path is not False:
        path_resized = resize_path(path, frame_cv2.shape, img_resize.shape)
        path_smooth = smooth_path(path_resized,
                                  weight_data=2.5,
                                  weight_smooth=0.25,
                                  tolerance=0.500001)

        # print(f"resized_path {path_resized.shape}")
        # print(f"thinned_path {path_smooth.shape}")

        # Plot the path
        cv2.polylines(frame_cv2, [path_smooth],
                      isClosed=False,
                      color=(255, 0, 0),
                      thickness=2)
        cv2.circle(
            frame_cv2, (path_smooth[0][0], path_smooth[0][1]), 4, (0, 0, 255), -1)
        cv2.circle(
            frame_cv2, (path_smooth[-1][0], path_smooth[-1][1]), 4, (0, 255, 0), -1)
        cv2.imshow('Path Results', frame_cv2)
        # cv2.waitKey(20)

        # Release memory
        del path_resized
        del path_smooth


def run_astar(frame_cv2, thresh_image, start, goal):
    global img_resize
    astar = Astar(None, None, None)
    if goal is not None:
        # print("goal", goal)

        if thresh_image.ndim == 3:
            H, W, _ = thresh_image.shape
        else:
            H, W = thresh_image.shape

        img_resize = cv2.resize(thresh_image, ((W // 6, H//3)))
        # img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        grid = img_resize.copy()

        points = np.argwhere(grid == 255)
        points = np.array([[x, y] for y, x in points])

        scale_factor_h = W / img_resize.shape[1]  # - 0.1
        scale_factor_w = H / img_resize.shape[0]  # + 0.2

        start_resize = np.array([(start[0] / scale_factor_h),
                                 (start[1] / scale_factor_w)]).astype(int)

        # start_resize = np.array([179, 45])
        goal_resize = np.array(
            [goal[0] / scale_factor_h,
             goal[1] / scale_factor_w]).astype(int)
        # print(f"after resize start {start_resize}, goal {goal_resize}")

        distances = dist_to_all(np.array([goal_resize]), points)
        # print(distances)
        # k = 3  # Select the k-th minimum distance
        # nearest_index = np.argpartition(distances, k - 1)[0, k - 1]

        nearest_index = np.argmin(distances)
        nearest_point = points[nearest_index]
        goal_resize = nearest_point

        cv2.circle(img_resize, (start_resize[0], start_resize[1]), 2,
                   (0, 0, 255), -1)
        cv2.circle(img_resize, (goal_resize[0], goal_resize[1]), 2,
                   (0, 255, 0), -1)

        # #### Get the path
        astar = Astar(start_resize, goal_resize, grid)
        astar.get_path()

        # path = astar.get_path()
        if astar.path_found is False:
            # print("Couldn't find a path to goal", goal)
            pass

        if astar.path is not None:
            if frame_cv2 is not None:
                draw_path(astar.path, frame_cv2)

    return astar.path_found
