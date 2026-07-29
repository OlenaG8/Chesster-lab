import cv2
import numpy as np


class ArucoDetector:
    def __init__(self, dict_type=cv2.aruco.DICT_4X4_250):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) == 4:
            sorted_corners = [c for _, c in sorted(zip(ids.flatten(), corners))]
            return sorted_corners, ids
        return None, None

    def draw_grid(self, frame, sorted_corners, size=800, grid_divisions=8):
        ideal_corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")
        image_corners = np.array([c[0][0] for c in sorted_corners], dtype="float32")

        matrix = cv2.getPerspectiveTransform(ideal_corners, image_corners)
        step = size // grid_divisions
        lines_points = []

        for i in range(grid_divisions + 1):
            lines_points.extend([[[i * step, 0]], [[i * step, size]]])
            lines_points.extend([[[0, i * step]], [[size, i * step]]])

        lines_points = np.array(lines_points, dtype="float32")
        transformed_points = cv2.perspectiveTransform(lines_points, matrix)

        for i in range(0, len(transformed_points), 2):
            pt1 = tuple(transformed_points[i][0].astype(int))
            pt2 = tuple(transformed_points[i + 1][0].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    def draw_markers(self, frame, corners, ids):
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    def generate_marker(self, marker_id, output_path, size=200):
        marker_image = cv2.aruco.generateImageMarker(
            self.dictionary, marker_id, size
        )
        cv2.imwrite(output_path, marker_image)


def run_live(cam_index=3):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    detector = ArucoDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids = detector.detect(frame)
        if corners is not None:
            detector.draw_grid(frame, corners)

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()