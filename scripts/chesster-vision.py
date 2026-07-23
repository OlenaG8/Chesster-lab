import cv2
import numpy as np
import chess
import chess.engine
import chess.svg
from PIL import Image
import io
import random
import os
import sys
import cairosvg
import time

from detect_aruco import get_aruco_corners
from camera_calibration.undistort import undistort_camera

ENGINE_PATH = r"/home/olena/Downloads/stockfish-ubuntu-x86-64/stockfish/stockfish-ubuntu-x86-64"
MOVE_THRESHOLD = 30
MIN_CONTOUR_AREA = 250
CAM_INDEX = 4
BOARD_ORIENTATION = "TOP"

MOTION_THRESHOLD = 500
DELAY = 2
DEBUG_MODE = False

if not os.path.exists(ENGINE_PATH):
    print(f"[ERROR] Engine file not found: {ENGINE_PATH}")
    sys.exit(1)

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
print(f"[INFO] Stockfish running from {ENGINE_PATH}")

files = 'abcdefgh'
ranks = '12345678'


def get_normalized_sq_points(size=400):
    step = size // 8
    sq_pts = {}
    board_files = 'abcdefgh'
    board_ranks = '87654321'
    for r in range(8):
        for c in range(8):
            tl = [c * step, r * step]
            tr = [(c + 1) * step, r * step]
            br = [(c + 1) * step, (r + 1) * step]
            bl = [c * step, (r + 1) * step]
            sq_pts[f"{board_files[c]}{board_ranks[r]}"] = [tl, tr, br, bl]
    return sq_pts


# Logic points (normalized 400x400)
sq_points = get_normalized_sq_points(400)

def get_original_sq_points(corners):
    src = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype="float32")
    dst = np.array([c[0][0] for c in corners], dtype="float32")
    H = cv2.getPerspectiveTransform(src, dst)

    src_grid_list = []
    for y in range(9):
        row = []
        for x in range(9):
            row.append([x, y])
        src_grid_list.append(row)

    src_grid = np.array(src_grid_list, dtype=np.float32)
    dst_grid = cv2.perspectiveTransform(src_grid.reshape(-1, 1, 2), H).reshape(9, 9, 2)

    chessboard_state = {}
    for r in range(8):
        for c in range(8):
            tl = dst_grid[r, c].tolist()
            tr = dst_grid[r, c + 1].tolist()
            br = dst_grid[r + 1, c + 1].tolist()
            bl = dst_grid[r + 1, c].tolist()
            chessboard_state[(r, c)] = [tl, tr, br, bl]

    board_files = 'abcdefgh'
    board_ranks = '87654321'
    original_squares = {}
    for (r_disp, c_disp), poly in chessboard_state.items():
        r_std, c_std = r_disp, c_disp
        file_letter = board_files[c_std]
        rank_char = board_ranks[r_std]
        original_squares[f"{file_letter}{rank_char}"] = poly
    return original_squares


def remap_square(square_name: str) -> str:
    f = square_name[0]
    r = square_name[1]
    fi = files.index(f)
    ri = ranks.index(r)
    if BOARD_ORIENTATION == "TOP":
        return square_name
    elif BOARD_ORIENTATION == "BOTTOM":
        return f"{files[7 - fi]}{ranks[7 - ri]}"
    elif BOARD_ORIENTATION == "SIDE_L":
        return f"{files[ri]}{ranks[7 - fi]}"
    elif BOARD_ORIENTATION == "SIDE_R":
        return f"{files[7 - ri]}{ranks[fi]}"
    else:
        return square_name


def poly_center(pts):
    a = np.array(pts, np.int32)
    M = cv2.moments(a)
    if M["m00"] == 0:
        return int(a[:, 0].mean()), int(a[:, 1].mean())
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def find_square(x, y):
    """Return square name if point (x,y) is inside the polygon for that square."""
    pt = (float(x), float(y))
    for sq, pts in sq_points.items():
        poly = np.array(pts, np.int32)
        # pointPolygonTest >= 0 means inside or on the edge of the polygon
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return sq
    return None


def overlay_poly(frame, poly_pts, color, alpha=0.45):
    overlay = frame.copy()
    pts = np.array(poly_pts, np.int32)
    cv2.fillPoly(overlay, [pts], color)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_board_labels(base_frame, pts_dict):
    overlay = base_frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for sq, pts in pts_dict.items():
        p = np.array(pts, np.int32)
        cv2.polylines(overlay, [p], True, (0, 255, 0), 2)
        if sq == "a1":
            cx, cy = poly_center(pts)
            mapped = remap_square(sq)
            cv2.putText(overlay, mapped, (cx - 12, cy + 5), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return overlay


def show_board(board, last_move=None):
    svg = chess.svg.board(board=board, lastmove=last_move, coordinates=True, size=450)
    png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(png_data))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Board State", img_cv)
    cv2.waitKey(1)


def get_normalized_view(frame, corners, size=400):
    src = np.array([c[0][0] for c in corners], dtype="float32")
    dst = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ], dtype="float32")
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, H, (size, size))


def process_frame_diff(ref_frame, frame_raw):
    g1 = 0.5 * ref_frame[:, :, 2] + 0.4 * ref_frame[:, :, 1] + 0.1 * ref_frame[:, :, 0]
    g2 = 0.5 * frame_raw[:, :, 2] + 0.4 * frame_raw[:, :, 1] + 0.1 * frame_raw[:, :, 0]
    g1 = g1.astype(np.uint8)
    g2 = g2.astype(np.uint8)

    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    g2 = cv2.GaussianBlur(g2, (5, 5), 0)
    diff = cv2.absdiff(g1, g2)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    diff = cv2.convertScaleAbs(diff, alpha=1.3, beta=0)
    _, diff_thresh = cv2.threshold(diff, MOVE_THRESHOLD, 255, cv2.THRESH_BINARY)
    diff_m = cv2.dilate(diff_thresh, None, iterations=4)
    diff_m = cv2.erode(diff_m, None, iterations=2)

    mask_board = np.zeros_like(diff_m)
    for pts in sq_points.values():
        cv2.fillPoly(mask_board, [np.array(pts, np.int32)], 255)
    diff_m = cv2.bitwise_and(diff_m, mask_board)

    if DEBUG_MODE:
        cv2.imshow("Diff", diff_m)

    kernel = np.ones((3, 3), np.uint8)
    diff_m = cv2.morphologyEx(diff_m, cv2.MORPH_OPEN, kernel)
    diff_m = cv2.morphologyEx(diff_m, cv2.MORPH_CLOSE, kernel)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = MIN_CONTOUR_AREA

    params.filterByCircularity = True
    params.minCircularity = 0.65

    params.filterByConvexity = True
    params.minConvexity = 0.7

    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(diff_m)

    return keypoints


def get_detected_squares(keypoints):
    detected = set()
    chosen_mapping = []

    for kp in keypoints:
        x, y = kp.pt
        sq = find_square(x, y)
        if sq:
            detected.add(sq)
            chosen_mapping.append((sq, x, y))

    return detected, chosen_mapping


def interpret_move(detected, board):
    # === move interpretation ===
    from_sq, to_sq = None, None
    if len(detected) == 2:
        a, b = list(detected)

        # Use previous board snapshot for more accurate detection
        # (copy state before move)
        prev_board = board.copy()

        piece_a = prev_board.piece_at(chess.parse_square(a))
        piece_b = prev_board.piece_at(chess.parse_square(b))

        # If only one has a piece at the initial position -> that is from_sq
        if piece_a and not piece_b:
            from_sq, to_sq = a, b
        elif piece_b and not piece_a:
            from_sq, to_sq = b, a
            from_sq, to_sq = b, a
        else:
            # If both are empty or both are occupied (difficult), use rank direction heuristic
            def rank_idx(s):
                return int(s[1])

            if board.turn == chess.WHITE:
                from_sq, to_sq = sorted([a, b], key=rank_idx)
            else:
                from_sq, to_sq = sorted([a, b], key=rank_idx, reverse=True)

    elif len(detected) == 1:
        # only one square changed — try a more reliable way to find from_sq
        to_sq = list(detected)[0]
        prev_board = board.copy()
        piece_now = board.piece_at(chess.parse_square(to_sq))

        # 1) If the current square is occupied, try to find a legal move ending here
        if piece_now:
            # filter candidates that originally had a piece on prev_board
            candidates = [m for m in board.legal_moves if m.uci()[2:] == to_sq]
            chosen = None
            for m in candidates:
                src = m.uci()[:2]
                if prev_board.piece_at(chess.parse_square(src)):
                    chosen = m
                    break
            # if none found that were originally occupied, fallback to first candidate
            if not chosen and candidates:
                chosen = candidates[0]
            if chosen:
                from_sq = chosen.uci()[:2]
                to_sq = chosen.uci()[2:]
        else:
            # 2) if the final square is empty -> possibility the piece moved from surrounding squares
            file = to_sq[0]
            rank = int(to_sq[1])
            fi = files.index(file)

            # create priority search order: vertical (according to turn), horizontal, diagonal, 2-step
            search_offsets = []

            if board.turn == chess.WHITE:
                # prefer datang dari bawah (rank-1), lalu left/right, lalu diagonals, then two-step from rank-2
                search_offsets += [(0, -1), (-1, 0), (1, 0), (-1, -1), (1, -1), (0, -2)]
            else:
                # black moves downward in rank numbers (from higher rank to lower)
                search_offsets += [(0, 1), (-1, 0), (1, 0), (-1, 1), (1, 1), (0, 2)]

            # ensure we also consider all orthogonals/diagonals if needed
            search_offsets += [(-1, 1), (1, 1), (-1, -1), (1, -1)]

            found = False
            for df, dr in search_offsets:
                f_idx = fi + df
                r_idx = rank + dr
                if 0 <= f_idx < 8 and 1 <= r_idx <= 8:
                    adj = f"{files[f_idx]}{r_idx}"
                    if prev_board.piece_at(chess.parse_square(adj)):
                        # verify move adj -> to_sq is legal
                        try_mv = chess.Move.from_uci(adj + to_sq)
                        if try_mv in board.legal_moves:
                            from_sq = adj
                            found = True
                            break
            # 3) if still not found, fallback: search for *any* neighbor that has a piece (without legal check)
            if not found:
                for df in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        if df == 0 and dr == 0:
                            continue
                        f_idx = fi + df
                        r_idx = rank + dr
                        if 0 <= f_idx < 8 and 1 <= r_idx <= 8:
                            adj = f"{files[f_idx]}{r_idx}"
                            if prev_board.piece_at(chess.parse_square(adj)):
                                # if move adj->to_sq is legal use it, otherwise keep adj as last-resort
                                try_mv = None
                                try:
                                    try_mv = chess.Move.from_uci(adj + to_sq)
                                except Exception:
                                    try_mv = None
                                if try_mv and try_mv in board.legal_moves:
                                    from_sq = adj
                                    found = True
                                    break
                                if not from_sq:
                                    from_sq = adj
                    if found:
                        break
    return from_sq, to_sq


def toggle_debug():
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    state = "ON" if DEBUG_MODE else "OFF"
    print(f"[INFO] Debug mode: {state}")

    if not DEBUG_MODE:
        try:
            if cv2.getWindowProperty("Diff", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Diff")
        except cv2.error:
            pass
        try:
            if cv2.getWindowProperty("Contours", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Contours")
        except cv2.error:
            pass


def main():
    global DEBUG_MODE

    is_moving = False
    prev_gray = None
    last_motion_time = time.time()

    display_sq_points = {}

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        engine.quit()
        sys.exit(1)

    map1, map2 = undistort_camera(CAM_INDEX)

    board = chess.Board()
    ref_frame = None
    last_move = None
    comp_turn = False
    engine_move = None
    move_history = []

    print("[INFO] Press 'u' to undo last move, 'U'=undo 2 moves, 'd'=toggle debug, 'q'=quit.")
    show_board(board)

    try:
        while not board.is_game_over():
            ret, raw_frame = cap.read()
            if not ret:
                continue

            undistorted_frame = cv2.remap(raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            corners, ids = get_aruco_corners(undistorted_frame)

            if corners is None or len(corners) != 4:
                if not display_sq_points:
                    cv2.imshow("Chess Tracker", undistorted_frame)
                else:
                    display = draw_board_labels(undistorted_frame.copy(), display_sq_points)
                    cv2.imshow("Chess Tracker", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Update coordinates dynamically for visual drawing on original frame
            display_sq_points = get_original_sq_points(corners)

            # Map the board to standard 400x400 for logic and motion
            norm_frame = get_normalized_view(undistorted_frame, corners, size=400)

            # Draw labels on the raw frame using display_sq_points
            display = draw_board_labels(undistorted_frame.copy(), display_sq_points)
            cv2.imshow("Chess Tracker", display)

            gray = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is None:
                prev_gray = gray
                ref_frame = norm_frame.copy()
                continue

            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_level = cv2.countNonZero(thresh)

            if motion_level > MOTION_THRESHOLD:
                is_moving = True
                last_motion_time = time.time()
            else:
                if is_moving and (time.time() - last_motion_time) > DELAY:
                    print(f"[INFO] Analyzing movement...")

                    keypoints = process_frame_diff(ref_frame, norm_frame)
                    detected, chosen_mapping = get_detected_squares(keypoints)

                    if DEBUG_MODE:
                        print(f"[DEBUG] Chosen mapping: {chosen_mapping}")
                        print(f"[DEBUG] Detected squares: {detected}")

                        if chosen_mapping:
                            dbg = norm_frame.copy()
                            for sq, cx, cy in chosen_mapping:
                                poly = np.array(sq_points[sq], np.int32)
                                cv2.polylines(dbg, [poly], True, (0, 255, 0), 2)
                                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                                cv2.putText(dbg, sq, (int(cx) + 6, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                            1)
                            cv2.imshow("Contours", dbg)

                    print(f"[DEBUG] Detected squares: {detected}")

                    from_sq, to_sq = interpret_move(detected, board)

                    if not from_sq or not to_sq:
                        print("[WARN] Invalid detection.")
                    else:
                        detected_move_uci = from_sq + to_sq
                        if engine_move and not comp_turn:
                            if detected_move_uci == engine_move.uci():
                                print(f"[SYNC] AI move verified: {detected_move_uci}")
                                board.push(engine_move)
                                move_history.append(engine_move)
                                engine_move = None
                                show_board(board, engine_move)
                                ref_frame = norm_frame.copy()
                            else:
                                print(f"[!] SYNC ERROR! Detected: {detected_move_uci}, Expected: {engine_move.uci()}.")
                        else:
                            try:
                                mv = chess.Move.from_uci(detected_move_uci)
                                if mv in board.legal_moves:
                                    print(f"[YOU] You played: {detected_move_uci}")
                                    board.push(mv)
                                    move_history.append(mv)
                                    show_board(board, mv)

                                    if display_sq_points:
                                        frame_high = overlay_poly(undistorted_frame.copy(), display_sq_points[from_sq],
                                                                  (0, 255, 0), 0.5)
                                        frame_high = overlay_poly(frame_high, display_sq_points[to_sq], (0, 0, 255),
                                                                  0.5)
                                        frame_high = draw_board_labels(frame_high, display_sq_points)
                                        cv2.imshow("Chess Tracker", frame_high)
                                        cv2.waitKey(700)

                                    ref_frame = norm_frame.copy()
                                    comp_turn = True
                                else:
                                    print(f"[!] Invalid move: {detected_move_uci}")
                            except Exception as e:
                                print(f"[!] Move interpretation error: {e}")

                    is_moving = False

            prev_gray = gray
            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                toggle_debug()

            if key == ord('u'):
                if move_history:
                    mv = move_history.pop()
                    board.pop()
                    print(f"[UNDO] Removing last move: {mv}")
                    show_board(board)
                    ref_frame = norm_frame.copy()
                    engine_move = None
                    comp_turn = False
                else:
                    print("[INFO] No moves to undo.")

            if key == ord('U'):
                if len(move_history) >= 2:
                    mv2 = move_history.pop()
                    mv1 = move_history.pop()
                    board.pop()
                    board.pop()
                    print(f"[UNDO] Removing last 2 moves: {mv1}, {mv2}")
                    show_board(board)
                    ref_frame = norm_frame.copy()
                    engine_move = None
                    comp_turn = False
                else:
                    print("[INFO] Not enough moves to undo twice.")

            if comp_turn and not engine_move:
                result = engine.play(board, chess.engine.Limit(time=random.uniform(0.4, 0.9)))
                engine_move = result.move
                last_move = engine_move
                print(f"[AI] Computer played: {engine_move.uci()}")
                show_board(board, last_move)

                try:
                    move_str = engine_move.uci()
                    if display_sq_points:
                        frame_ai = overlay_poly(undistorted_frame.copy(), display_sq_points[move_str[:2]], (0, 255, 255), 0.45)
                        frame_ai = overlay_poly(frame_ai, display_sq_points[move_str[2:]], (0, 165, 255), 0.45)
                        frame_ai = draw_board_labels(frame_ai, display_sq_points)
                        cv2.imshow("Chess Tracker", frame_ai)
                        cv2.waitKey(900)
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Failed to highlight AI: {e}")

                comp_turn = False

            if key == ord('q'):
                print("[INFO] Quitting.")
                break

        print("[INFO] Game over.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.quit()


if __name__ == '__main__':
    main()
