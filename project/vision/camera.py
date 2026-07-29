import cv2

class Camera:
    def __init__(self, index, map1=None, map2=None):
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.index = index
        self.map1 = map1
        self.map2 = map2

    def read(self):
        ret, frame = self.cap.read()

        if frame is None:
            return None

        if (self.map1 is not None) and (self.map2 is not None):
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        return frame

    def release(self):
        self.cap.release()