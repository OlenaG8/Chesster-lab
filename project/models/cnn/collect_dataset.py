import cv2
import os
import random
import numpy as np

from project.config import *

from project.vision.detect_aruco import ArucoDetector
from project.vision.homography import warp_board
from project.vision.board_detector import split_board

from project.vision.camera_calibration.undistort import undistort_camera
from project.vision.camera import Camera

FILES = 'abcdefgh'
RANKS = '12345678'

CNN_SIZE = 64
OFFSET = 10

TARGET = 100 # target amount of pictures of each piece
PIECE_INDEX = 0 # piece index from PIECES dict
CURR_INDEX = 36 # the current index in the dataset

PIECES = [
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

def get_random_square():
    return random.choice(FILES) + random.choice(RANKS)

def save_square(img, label):
    path = f"dataset/raw/{label}"

    os.makedirs(path, exist_ok=True)

    index = CURR_INDEX
    if label != "empty" and index < TARGET:
        filename = f"{path}/{index}.png"
        cv2.imwrite(filename, img)

def crop_square(board, row, col):
    y1 = (row * CELL_SIZE) + OFFSET
    y2 = ((row + 1) * CELL_SIZE) - OFFSET

    x1 = (col * CELL_SIZE) + OFFSET
    x2 = ((col + 1) * CELL_SIZE) - OFFSET

    return board[y1:y2, x1:x2]

def square_to_index(square):
    file = FILES.index(square[0])
    rank = int(square[1])

    row = 8 - rank
    col = file
    return row, col

def save_board(board_img, piece_name, piece_square):
    piece_row, piece_col = square_to_index(piece_square)

    for row in range(8):
        for col in range(8):
            square = crop_square(board_img, row, col)
            square = cv2.resize(square, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_AREA)

            if row == piece_row and col == piece_col:
                save_square(square, piece_name)
            else:
                save_square(square, "empty")

def add_info_panel(board_img, piece, square, counter, target):
    height = board_img.shape[0]

    panel_width = 300
    panel = np.full((height, panel_width, 3), 80, dtype=np.uint8)

    cv2.putText(panel, f"Piece: {piece}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, f"Place on: {square}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, f"{counter}/{target}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    result = np.hstack((board_img, panel))
    return result


if __name__ == '__main__':
    map1, map2 = undistort_camera(CAM_INDEX)
    camera = Camera(CAM_INDEX, map1, map2)

    aruco = ArucoDetector()

    square = get_random_square()

    while True:
        frame = camera.read()

        if frame is None:
            continue

        cv2.imshow("Board", frame)
        corners = aruco.detect(frame)
        if corners is None:
            continue

        board_img = warp_board(frame, corners)
        display_board = add_info_panel(board_img, PIECES[PIECE_INDEX], square, CURR_INDEX, TARGET)
        cv2.imshow("Board", display_board)

        key = cv2.waitKey(1)
        if key == ord("s"):
            save_board(board_img, PIECES[PIECE_INDEX], square)
            square = get_random_square()
            CURR_INDEX += 1
        elif key == ord("q"):
            break

        if CURR_INDEX == TARGET:
            PIECE_INDEX += 1
            CURR_INDEX = 0
            if PIECE_INDEX >= len(PIECES):
                break
