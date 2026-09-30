import chess
import cv2
import numpy as np

from project.vision.detect_aruco import get_square_size


class AugmentFrame:
    def __init__(self, frame, corners):
        self.frame = frame.copy()
        self.corners = corners
        self.matrix = None
        self.square_size = None

    def highlight_move(self, move: chess.Move, size=800):
        self.matrix, self.square_size = get_square_size(self.corners, size, 8)

        self.frame = self.highlight_square(move.from_square, (50, 100, 255))
        self.frame = self.highlight_square(move.to_square, (50, 255, 100))

        return self.frame

    def highlight_square(self, square: chess.Square, color):
        points = self.square_to_points(square)
        transformed_points = cv2.perspectiveTransform(points, self.matrix)

        img_copy = self.frame.copy()
        overlay = img_copy.copy()

        pt1 = tuple(transformed_points[0][0].astype(int))
        pt2 = tuple(transformed_points[1][0].astype(int))
        cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)

        alpha = 0.6
        img_new = cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0)
        return img_new

    def square_to_points(self, square: chess.Square):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        x1 = file * self.square_size
        y1 = (7 - rank) * self.square_size

        x2 = x1 + self.square_size
        y2 = y1 + self.square_size

        return np.array([[[x1, y1]], [[x2, y2]]], dtype="float32")