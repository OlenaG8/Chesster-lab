import cv2
import numpy as np

def generate_aruco(marker_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    marker_size = 200
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imwrite(f'../aruco-markers/marker-{marker_id}.png', marker_image)

def detect_aruco_live():
    cap = cv2.VideoCapture(1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            warped = get_stabilized_view(frame, corners, ids)
            if warped is not None:
                cv2.imshow('Widok z gory', warped)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow('Podglad z kamery', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_grid_on_chessboard(frame, corners, ids):
    if len(ids) != 4: return frame

    pts_src = np.array([corners[i][0][0] for i in range(4)], dtype="float32")
    width, height = 400, 400
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    warped = cv2.warpPerspective(frame, matrix, (width, height))

    step = width // 8
    for i in range(9):
        cv2.line(warped, (i * step, 0), (i * step, height), (0, 255, 0), 2)
        cv2.line(warped, (0, i * step), (width, i * step), (0, 255, 0), 2)

    return warped


last_matrix = None


def get_stabilized_view(frame, corners, ids):
    global last_matrix

    # --- POPRAWIONY FRAGMENT KODU ---

    # Sprawdź, czy wykryto ID i czy jest ich dokładnie 4
    if ids is not None and len(ids) == 4:

        # 1. Płaska lista ID, żeby łatwiej było szukać
        ids_flat = ids.flatten()

        # 2. Utwórz nową, pustą listę na posortowane narożniki
        # Zamierzamy ułożyć je w kolejności docelowej: [LG, PG, PD, LD]
        # Odpowiada to stałej definicji pts_dst: [[0, 0], [w, 0], [w, h], [0, h]]
        sorted_pts = []

        # 3. Poszukaj każdego ID po kolei i dodaj jego narożnik do listy
        # Zakładając, że Twoje markery mają ID od 0 do 3.
        target_ids = [0, 1, 2, 3]  # Ta kolejność musi odpowiadać pts_dst

        all_found = True
        for target_id in target_ids:
            # Sprawdź, czy target_id jest na liście wykrytych
            try:
                # Znajdź indeks, pod którym znajduje się ten ID
                index = list(ids_flat).index(target_id)

                # Pobierz narożnik dla tego indeksu (konkretnie punkt środkowy lub pierwszy narożnik znacznika)
                # Użyjmy pierwszego narożnika (lewy-góra) samego znacznika
                marker_corner = corners[index][0][0]  # Pierwszy piksel (x, y) znacznika
                sorted_pts.append(marker_corner)

            except ValueError:
                # Jeśli któregoś ID brakuje, nie możemy obliczyć homografii
                all_found = False
                break

        # 4. Jeśli znaleźliśmy wszystkie 4 i posortowaliśmy, oblicz homografię
        if all_found:
            pts_src = np.array(sorted_pts, dtype="float32")

            width, height = 400, 400
            pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(frame, matrix, (width, height))

            # Opcjonalnie narysuj siatkę
            # ... kod rysowania siatki ...

            cv2.imshow('Widok z gory', warped)
        else:
            # Pokaż czarny obraz (lub zachowaj stabilizację z poprzedniej klatki), jeśli brakuje markerów
            cv2.imshow('Widok z gory', np.zeros((400, 400), dtype="uint8"))

    # --- KONIEC POPRAWKI ---

if __name__ == '__main__':
    detect_aruco_live()
