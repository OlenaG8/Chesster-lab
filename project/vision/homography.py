import cv2
import numpy as np
from project.config import *

def warp_board(frame, corners, size=BOARD_SIZE):
    src = np.array([
        c[0][0] for c in corners
    ], dtype="float32")

    dst = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ], dtype="float32")

    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, H, (size, size))