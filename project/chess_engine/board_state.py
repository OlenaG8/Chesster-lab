import io

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
from PIL import Image

from project.chess_engine.fen import board_to_fen


class BoardState:
    def __init__(self, init_state=None):
        self.board = chess.Board()
        self.state = init_state or [ 5,  2,  0,  4,  1,  0,  2,  5,
                                     3,  3,  3,  3,  3,  3,  3,  3,
                                     6,  6,  6,  6,  6,  6,  6,  6,
                                     6,  6,  6,  6,  6,  6,  6,  6,
                                     6,  6,  6,  6,  6,  6,  6,  6,
                                     6,  6,  6,  6,  6,  6,  6,  6,
                                    10, 10, 10, 10, 10, 10, 10, 10,
                                    12,  9,  7, 11,  8,  7,  9, 12]

    def apply_map(self, map):
        result = self.state.copy()

        disappeared = []
        appeared = []

        for i in range(64):
            old_state = self.state[i]
            new_state = map[i]

            old_occupied = old_state != 6
            new_occupied = new_state != 1

            if old_occupied and not new_occupied:
                disappeared.append((i, old_state)) # A piece disappeared from this square.
            elif (old_state < 6 and new_state == 2) or (old_state > 6 and new_state == 0):
                appeared.append(i) # a piece is captured
            elif not old_occupied and new_occupied:
                appeared.append(i)   # A piece appeared on this square.

        if len(disappeared) == 2 and len(appeared) == 1:
                if appeared[0] in range(16, 24) and disappeared[0][0] in range(24, 32):
                    for square, _ in disappeared:
                        result[square] = 6
                    result[appeared[0]] = 10
                    self.state = result
                    self.update_fen(board_to_fen(self.state))
                    return
                elif appeared[0] in range(40, 48) and disappeared[0][0] in range(32, 40):
                    for square, _ in disappeared:
                        result[square] = 6
                    result[appeared[0]] = 3
                    self.state = result
                    self.update_fen(board_to_fen(self.state))
                    return
                else:
                    raise Exception("Incorrect move.")

        if len(disappeared) != len(appeared):
            raise Exception("Mismatch between disappeared and appeared.")
        elif len(disappeared) > 2 or len(appeared) > 2:
            raise Exception("Too many changes.")

        if len(disappeared) == 2 and len(appeared) == 2:
            if ((disappeared == [(0, 5), (4, 1)] and appeared == [2, 3]) or
                (disappeared == [(4, 1), (7, 5)] and appeared == [5, 6]) or
                (disappeared == [(56, 12), (60, 8)] and appeared == [58, 59]) or
                (disappeared == [(60, 8), (63, 12)] and appeared == [61, 62])):
                disappeared.reverse()
            else:
                raise Exception("Too many changes.")

        # Remove pieces that disappeared.
        for square, _ in disappeared:
            result[square] = 6

        # Put the disappeared pieces on newly occupied squares.
        for square, (_, piece) in zip(appeared, disappeared):
            result[square] = piece
        self.state = result
        self.update_fen(board_to_fen(self.state))

    def update_fen(self, fen):
        self.board = chess.Board(fen)

    def legal_move(self, uci):
        move = chess.Move.from_uci(uci)
        return move in self.board.legal_moves

    def push(self, uci):
        move = chess.Move.from_uci(uci)
        self.board.push(move)

    def show_board(self, last_move=None):
        svg = chess.svg.board(self.board, lastmove=last_move, coordinates=True, size=450)
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Board State", img_cv)
        cv2.waitKey(1)

    def display(self):
        print(self.board.unicode())