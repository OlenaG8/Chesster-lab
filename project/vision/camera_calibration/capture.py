import cv2

cap = cv2.VideoCapture(3)
i = 0

while(True):
    ret, frame = cap.read()

    cv2.imshow(f'img{i}',frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('y'):
        cv2.imwrite(f'distorted_cam2/{i}.png', frame)
        i += 1
        cv2.destroyAllWindows()

    if key == ord('q'):
        print("Quitting.")
        break

cv2.destroyAllWindows()
cap.release()