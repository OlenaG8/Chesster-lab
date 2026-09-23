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

import chess
import cv2
from config import *
from vision.detect_aruco import ArucoDetector
from vision.homography import warp_board
from vision.board_detector import split_board
from vision.cnn_classifier import ChessCNNClassifier
from chess_engine.stockfish import Stockfish
from chess_engine.board_state import BoardState
from vision.camera_calibration.undistort import getDistortionMaps
from vision.camera import Camera


def read_board_from_camera(camera, aruco, cnn, chess_board):
    frame = camera.read()

    if frame is None:
        raise RuntimeError("Could not read frame from camera.")

    corners, ids = aruco.detect(frame)

    if corners is None or len(corners) < 4:
        raise RuntimeError("[WARN] Could not detect ArUco markers.")

    board_img = warp_board(frame, corners)

    squares = split_board(board_img)

    cnn_state = []

    for sq, img in squares.items():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cls = cnn.predict(img)
        cnn_state.append(cls // 2)

    print("[CNN]", cnn_state)

    test_board = chess_board.copy()

    try:
        test_board.apply_map(cnn_state)
    except Exception as e:
        raise RuntimeError(f"Board recognition rejected: {e}") from e

    if not test_board.board.is_valid():
        raise RuntimeError("Recognized board is not a valid chess position.")

    return test_board


def main():
    is_moving = False
    last_motion_time = time.time()
    comp_turn = False
    last_move = None

    cached_corners = None
    frames_since_detection = 0

    map1, map2 = getDistortionMaps(CAM_INDEX)
    camera = Camera(CAM_INDEX, map1, map2)
    aruco = ArucoDetector()
    cnn = ChessCNNClassifier(CNN_MODEL_PATH)
    chess_board = BoardState()
    stockfish = Stockfish(ENGINE_PATH)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    print("[INFO] Press 'q' = quit")

    try:
        while not chess_board.board.is_game_over():
            frame = camera.read()
            if frame is None:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            corners, ids = aruco.detect(frame)
            if corners is not None and len(corners) >= 4:
                cached_corners = corners
                frames_since_detection = 0
            elif cached_corners is not None and frames_since_detection < MAX_CACHED_FRAMES:
                corners = cached_corners
                frames_since_detection += 1
            else:
                continue

            frame_copy = frame.copy()
            aruco.draw_grid(frame_copy, corners)
            cv2.imshow("Preview", frame_copy)

            board_img = warp_board(frame, corners)
            cv2.imshow("Chessboard", board_img)
            chess_board.show_board(last_move=last_move)

            fg_mask = bg_subtractor.apply(board_img)
            motion_level = cv2.countNonZero(fg_mask)

            if motion_level > MOTION_THRESHOLD:
                is_moving = True
                last_motion_time = time.time()
            else:
                if is_moving and (time.time() - last_motion_time) > DELAY:
                    print("[INFO] Analyzing movement ...")

                    recognized_board = None

                    for trial in range(MAX_BOARD_READ_TRIALS):
                        print(f"[INFO] Board recognition attempt {trial + 1}/{MAX_BOARD_READ_TRIALS}")

                        try:
                            recognized_board = read_board_from_camera(camera, aruco, cnn, chess_board)
                            print("[INFO] Board recognized successfully.")
                            break
                        except Exception as e:
                            print(f"[WARN] {e}")

                    if recognized_board is None:
                        print("[ERROR] Failed to recognize a valid chessboard after 100 attempts.")
                        break

                    chess_board = recognized_board

                    if chess_board.board.turn == chess.BLACK:
                        comp_turn = True

                    is_moving = False

            if comp_turn:
                move = stockfish.get_move(chess_board.board)
                print("[Stockfish] move:", move)
                chess_board.show_board(last_move=move)

                if move is not None:
                    expected_board = chess_board.copy()
                    expected_board.push(move)

                    '''
                    print("[INFO] Waiting for a robot to make a move.")
            
                    robot.move(move)
                    '''
                    time.sleep(4)
                    detected_board = None

                    for trial in range(MAX_BOARD_READ_TRIALS):
                        print(f"[INFO] Board recognition attempt {trial + 1}/{MAX_BOARD_READ_TRIALS}")

                        try:
                            detected_board = read_board_from_camera(camera, aruco, cnn, chess_board)
                            print("[INFO] Board recognized successfully.")
                            break
                        except Exception as e:
                            print(f"[WARN] {e}")

                    if detected_board is None:
                        print("[ERROR] Failed to recognize a valid chessboard after 100 attempts.")
                        break

                    if detected_board.board == expected_board.board:
                        chess_board = expected_board
                        print("[INFO] Robot move verified.")
                    else:
                        print("[ERROR] Physical board does not match expected move.")

                    last_move = move

                comp_turn = False

        game_over, result, reason = chess_board.check_game_over()
        print("[GAME OVER]")
        print(reason)
        print(f"[RESULT] {result}")
    finally:
        camera.cap.release()
        cv2.destroyAllWindows()
        stockfish.close()


if __name__ == "__main__":
    main()