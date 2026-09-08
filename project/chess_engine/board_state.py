import io

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
from PIL import Image


class BoardState:
    def __init__(self):
        self.board = chess.Board()

    def update_fen(self,fen):
        self.board = chess.Board(fen)

    def legal_move(self,uci):
        move = chess.Move.from_uci(uci)
        return move in self.board.legal_moves

    def push(self,uci):
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