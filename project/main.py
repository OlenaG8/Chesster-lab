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

Step 5 - piece recognition (ChessCNNClassifier)
    for every square:
        class <- CNN.predict(square)
        board[row][col] <- class

Step 6 - FEN + move verification

Step 7 - get the best move(Stockfish Engine)

Step 8 - send the move to the robot and then verify (SO-ARM101)
"""
import time
import cv2
from config import *
from vision.detect_aruco import ArucoDetector
from vision.homography import warp_board
from vision.board_detector import split_board, CLASS_MAP
from vision.cnn_classifier import ChessCNNClassifier
from chess_engine.stockfish import Stockfish
from chess_engine.board_state import BoardState
from chess_engine.fen import board_to_fen, update_board
from vision.camera_calibration.undistort import getDistortionMaps
from vision.camera import Camera


def main():
    is_moving = False
    last_motion_time = time.time()
    comp_turn = False

    map1, map2 = getDistortionMaps(CAM_INDEX)
    camera = Camera(CAM_INDEX, map1, map2)
    aruco = ArucoDetector()
    cnn = ChessCNNClassifier(CNN_MODEL_PATH)
    chess_board = BoardState()
    stockfish = Stockfish(ENGINE_PATH)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    print("[INFO] Press 'u' to undo last move, 'U' = undo 2 moves, 'q' = quit")

    cached_corners = None

    try:
        while not chess_board.board.is_game_over():
            frame = camera.read()

            if frame is None:
                continue

            if cached_corners is None:
                corners, ids = aruco.detect(frame)
                if corners is not None and len(corners) >= 4:
                    cached_corners = corners
                else:
                    continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            board_img = warp_board(frame, cached_corners)
            cv2.imshow("Preview", board_img)
            chess_board.show_board()

            fg_mask = bg_subtractor.apply(board_img)
            motion_level = cv2.countNonZero(fg_mask)

            if motion_level > MOTION_THRESHOLD:
                is_moving = True
                last_motion_time = time.time()
            else:
                if is_moving and (time.time() - last_motion_time) > DELAY:
                    print("[INFO] Analyzing movement ...")

                    squares = split_board(board_img)
                    cnn_state = []

                    for sq, img in squares.items():
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cls = cnn.predict(img)
                        cnn_state.append(cls)

                    #old_fen = board_to_fen(old_board_state)
                    #new_state = update_board(old_board_state, cnn_state)
                    print(cnn_state)
                    fen = board_to_fen(cnn_state)

                    try:
                        chess_board.update_fen(fen)
                        if chess_board.board.is_valid():
                            print("ok")
                            # old_board_state = new_state
                            # comp_turn = True
                        else:
                            print("[ERROR] Invalid board position.")
                    except ValueError:
                        print("[ERROR] Malformed FEN.")

                    is_moving = False

            if comp_turn:
                move = stockfish.get_move(chess_board.board)
                print("[Stockfish] move:", move)
                comp_turn = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cached_corners = None

        print("[INFO] Game over.")
    finally:
        camera.cap.release()
        cv2.destroyAllWindows()
        stockfish.close()


if __name__ == "__main__":
    main()