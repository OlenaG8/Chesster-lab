import cv2
import numpy as np

DIM = (640, 480)
K1 = np.array([[553.9555809655991, 0.0, 324.45444681034996], [0.0, 554.3429069436585, 228.2748270642295], [0.0, 0.0, 1.0]])
D1 = np.array([[-0.4467195274935519, 0.261307133907364, 0.0016875833637850534, 0.00035716928490734236, -0.07570453491224562]])

K2 = np.array([[551.1560029404905, 0.0, 360.8064089147551], [0.0, 552.0301499038216, 265.9019319371812], [0.0, 0.0, 1.0]])
D2 = np.array([[-0.4483986817696431, 0.2684104960562383, 2.3284080912077103e-05, 5.902916735538078e-05, -0.10662990298229139]])

def undistort_camera(cam_index):
    if cam_index == 3:
        map1, map2 = cv2.initUndistortRectifyMap(K1, D1, np.eye(3), K1, DIM, cv2.CV_16SC2)
    elif cam_index == 4:
        map1, map2 = cv2.initUndistortRectifyMap(K2, D2, np.eye(3), K2, DIM, cv2.CV_16SC2)
    else:
        return None

    return map1, map2

def main():
    cap = cv2.VideoCapture(3)
    map1, map2 = cv2.initUndistortRectifyMap(K1, D1, np.eye(3), K1, DIM, cv2.CV_16SC2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow("Kamera - oryginal", frame)
        cv2.imshow("Kamera - undistorted", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()