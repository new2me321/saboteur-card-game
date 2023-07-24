#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Card:
    def __init__(self, type: str, img: np.ndarray,   center: tuple) -> None:
        self.type = type
        self.x, self.y = center
        self.image = img
        self.contour = []

    def get_cx(self) -> int:
        return self.x

    def get_cy(self) -> int:
        return self.y

    def get_image(self) -> np.ndarray:
        return self.image

    def get_type(self) -> str:
        return self.type


if __name__ == "__main__":
    img = np.zeros((10, 10))
    card = Card("test", img, (5, 5))
    card.image[card.get_cx(), card.get_cy()] = 255

    print(card.get_image())
    print("x:", card.get_cx())
    print("y:", card.get_cy())
    print(card.get_type())
