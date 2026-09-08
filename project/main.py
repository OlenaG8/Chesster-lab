"""
Step 1 - Get Frame and undistort (undistort.py)
while game is running:
    frame <- camera.read()
    undistort(frame)

Step 2 - detect AruCo markers (detect_aruco.py)
ID0 -------- ID1
|              |
|  Chessboard  |
|              |
ID3 -------- ID4

corners <- detector.detect(undistorted_frame)
if ids.count < 4:
    continue

Step 3 - straightening the board
H <- find_homography(detected_corners, destination_corners)
topView <- warpPerspective(frame, H)
+ ---------- +
| a8, b7 ... |
| a7, b6 ... |
| ...        |
| a1, b1 ... |
+ ---------- +

Step 4 - Divide the board to get 64 images

Step 5 - piece recognition (ChessCNN)
    for every square:
        class <- CNN.predict(square)
        board[row][col] <- class

Step 6 - FEN + move verification

Step 7 - get the best move(Stockfish Engine)

Step 8 - send the move to the robot and then verify (SO-ARM101)
"""
import cv2

from config import *

from vision.detect_aruco import ArucoDetector
from vision.homography import warp_board
from vision.board_detector import split_board, CLASS_MAP
from vision.cnn_classifier import ChessCNNClassifier

from chess_engine.stockfish import Stockfish
from chess_engine.board_state import BoardState
from chess_engine.fen import board_to_fen

from vision.camera_calibration.undistort import getDistortionMaps
from vision.camera import Camera


def main():
    map1, map2 = getDistortionMaps(CAM_INDEX)
    camera = Camera(CAM_INDEX, map1, map2)

    aruco = ArucoDetector()
    cnn = ChessCNNClassifier(CNN_MODEL_PATH)

    chess_board = BoardState()
    stockfish = Stockfish(ENGINE_PATH)

    while True:
        frame = camera.read()

        if frame is None:
            continue

        corners, ids = aruco.detect(frame)

        if corners is None:
            continue

        board_img = warp_board(frame, corners)
        cv2.imshow("Preview", board_img)

        squares = split_board(board_img)
        state = []

        for sq, img in squares.items():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cls = cnn.predict(img)

            state.append(cls)

        fen = board_to_fen(state)

        chess_board.update_fen(fen)
        #chess_board.show_board()
        chess_board.display()

        move = stockfish.get_move(chess_board.board)
        print("Stockfish:", move)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()