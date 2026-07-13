import cv2
import numpy as np

def generate_aruco(marker_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    marker_size = 200
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imwrite(f'../aruco-markers/marker-{marker_id}.png', marker_image)

def get_chessboard_state(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) == 4:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        sorted_corners = [c for _, c in sorted(zip(ids.flatten(), corners))]

        src = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype="float32")
        dst = np.array([c[0][0] for c in sorted_corners], dtype="float32")
        H = cv2.getPerspectiveTransform(src, dst)

        src_grid = np.array([[[x, y] for x in range(9)] for y in range(9)], dtype=np.float32)
        dst_grid = cv2.perspectiveTransform(src_grid.reshape(-1, 1, 2), H).reshape(9, 9, 2)

        chessboard_state = {}
        for r in range(8):
            for c in range(8):
                tl = dst_grid[r, c].tolist()
                tr = dst_grid[r, c + 1].tolist()
                br = dst_grid[r + 1, c + 1].tolist()
                bl = dst_grid[r + 1, c].tolist()
                chessboard_state[(r, c)] = [tl, tr, br, bl]

        files = 'abcdefgh'
        ranks = '87654321'
        squares_std = {}
        for (r_disp, c_disp), poly in chessboard_state.items():
            r_std, c_std = r_disp, c_disp
            file_letter = files[c_std]
            rank_char = ranks[r_std]
            squares_std[f"{file_letter}{rank_char}"] = poly

        return squares_std
    return None


def draw_grid_in_perspective(frame, sorted_corners):
    size = 800
    ideal_corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
    image_corners = np.array([c[0][0] for c in sorted_corners], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ideal_corners, image_corners)
    step = size // 8
    lines_points = []

    for i in range(9):
        lines_points.extend([[[i * step, 0]], [[i * step, size]]])
        lines_points.extend([[[0, i * step]], [[size, i * step]]])

    lines_points = np.array(lines_points, dtype="float32")
    transformed_points = cv2.perspectiveTransform(lines_points, matrix)

    for i in range(0, len(transformed_points), 2):
        pt1 = tuple(transformed_points[i][0].astype(int))
        pt2 = tuple(transformed_points[i + 1][0].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

def detect_aruco_live():
    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        exit()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 4:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            sorted_corners = [c for _, c in sorted(zip(ids.flatten(), corners))]
            draw_grid_in_perspective(frame, sorted_corners)

        cv2.imshow('Podglad z kamery', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_aruco_live()