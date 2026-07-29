"""
Step 1 - Initialization:
- configure camera
- calibrate camera

Step 2 - Get Frame and undistort (undistort.py)
while game is running:
    frame <- camera.read()
    undistort(frame)

Step 3 - detect AruCo markers (detect_aruco.py)
ID0 -------- ID1
|              |
|  Chessboard  |
|              |
ID3 -------- ID4

corners <- detector.detect(undistorted_frame)
if ids.count < 4:
    continue

Step 4 - straightening the board
H <- find_homography(detected_corners, destination_corners)
topView <- warpPerspective(frame, H)
+ ---------- +
| a8, b7 ... |
| a7, b6 ... |
| ...        |
| a1, b1 ... |
+ ---------- +

Step 5 - Divide the board to get 64 images

Step 6 - piece recognition (ChessCNN)
    for every square:
        class <- CNN.predict(square)
        board[row][col] <- class

Step 7 - FEN + move verification

Step 8 - get the best move(Stockfish Engine)

Step 9 - send the move to the robot and then verify (SO-ARM101)
"""

from config import *

from vision.detect_aruco import ArucoDetector
from vision.homography import warp_board
from vision.board_detector import split_board
from vision.cnn_classifier import ChessCNN

from chess_engine.stockfish import Stockfish
from chess_engine.board_state import BoardState
from chess_engine.fen import board_to_fen

from vision.camera_calibration.undistort import undistort_camera
from vision.camera import Camera


def main():
    map1,map2 = (undistort_camera(CAM_INDEX))
    camera = Camera(CAM_INDEX, map1, map2)
    aruco = ArucoDetector()

    #cnn = ChessCNN(CNN_MODEL_PATH)
    cnn = None

    chess_board = BoardState()
    stockfish = Stockfish(ENGINE_PATH)


    while True:
        frame = camera.read()

        if frame is None:
            continue

        corners = aruco.detect(frame)

        if corners is None:
            continue

        board_img = warp_board(frame, corners)

        squares = split_board(board_img)

        state=[]
        #for sq,img in squares.items():
            #cls = cnn.predict(img)
            #state.append(cls)


        # fen = board_to_fen(state)
        #
        # chess_board.update_fen(fen)

        move = stockfish.get_move(chess_board.board)

        print("Stockfish:", move)



if __name__=="__main__":
    main()