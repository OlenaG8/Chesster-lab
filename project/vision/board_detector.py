import cv2
from project.config import *

FILES = "abcdefgh"
RANKS = '87654321'

CLASS_MAP = {
    0: "B", 1: "K", 2: "N", 3: "P", 4: "Q", 5: "R",
    6: "b", 7: "empty", 8: "k", 9: "n", 10: "p", 11: "q", 12: "r"
}

CNN_SIZE = 64
OFFSET = 10

def crop_square(board, row, col):
    y1 = (row * CELL_SIZE) + OFFSET
    y2 = ((row + 1) * CELL_SIZE) - OFFSET

    x1 = (col * CELL_SIZE) + OFFSET
    x2 = ((col + 1) * CELL_SIZE) - OFFSET

    return board[y1:y2, x1:x2]

def split_board(board_img):
    squares = {}

    for row in range(8):
        for col in range(8):
            name = FILES[col] + RANKS[row]

            square = crop_square(board_img, row, col)

            square = cv2.resize(
                square,
                (CNN_SIZE, CNN_SIZE),
                interpolation=cv2.INTER_AREA
            )

            squares[name] = square

    return squares