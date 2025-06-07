import cv2
import torch
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
from db import add_video, add_zone, add_pedestrian, add_analysis, add_camera

# Ініціалізація DeepSORT глобально
deepsort = DeepSort(max_age=30, nn_budget=100, override_track_class=None)

# Загрузка модели YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Глобальні змінні для вибору зони
drawing = False
crossing_zone = None
start_point = (-1, -1)

def select_zone(event, x, y, flags, param):
    global start_point, crossing_zone, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        crossing_zone = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        crossing_zone = (start_point[0], start_point[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - start_point[0]) > 20 and abs(y - start_point[1]) > 20:
            crossing_zone = (min(start_point[0], x), min(start_point[1], y),
                             max(start_point[0], x), max(start_point[1], y))
        else:
            crossing_zone = None

def plot_graph(people_per_second, analysis_started, total_video_time, total_time, fps):
    if analysis_started and people_per_second:
        plt.figure(figsize=(10, 6))
        time_axis = [i / fps for i in range(len(people_per_second))]
        plt.plot(time_axis, people_per_second, marker='o')
        plt.title('Кількість унікальних пішоходів на переході за секунду')
        plt.xlabel('Час (секунди)')
        plt.ylabel('Кількість людей')
        plt.grid(True)
        plt.savefig('pedestrian_analysis.png')
        plt.show()
    else:
        print("Графік не побудовано: аналіз не розпочато або немає даних.")

def mock_camera_stream(camera_ip):
    # Заглушка для імітації потоку з камери
    print(f"Спроба підключення до камери за IP: {camera_ip}")
    # Повертаємо тестове відео як приклад (замінити на реальний потік пізніше)
    return cv2.VideoCapture("E:\\хлам\\тест карты меню - Безпечне місто ЧЕРНІГІВ – Mozilla Firefox 2025-04-30 11-52-01.mp4")

def main():
    global crossing_zone, drawing, start_point, deepsort
    # Список імітаційних IP-камер (замінити на реальні пізніше)
    mock_cameras = [
        {"ip": "192.168.1.100", "name": "Camera 1"},
        {"ip": "192.168.1.101", "name": "Camera 2"}
    ]

    # Додаємо камери до бази
    for camera in mock_cameras:
        camera_id = add_camera(camera["ip"], camera["name"])
        print(f"Камера {camera['name']} додана, ID: {camera_id}")

    # Вибір джерела: 0 - відео, 1 - камера (заглушка)
    source_type = 0  # За замовчуванням відео
    cap = None

    if source_type == 0:
        video_path = "E:\\хлам\\тест карты меню - Безпечне місто ЧЕРНІГІВ – Mozilla Firefox 2025-04-30 11-52-01.mp4"
        cap = cv2.VideoCapture(video_path)
    else:
        cap = mock_camera_stream(mock_cameras[0]["ip"])

    if not cap.isOpened():
        print("Не вдалося відкрити джерело!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Не вдалося отримати FPS!")
        return
    frame_duration = 1 / fps
    current_frame = 0

    cv2.namedWindow("Pedestrian Crossing Analysis")
    cv2.setMouseCallback("Pedestrian Crossing Analysis", select_zone)

    track_times = {}
    total_time = 0
    analysis_started = False
    people_per_second = []

    video_id = add_video("mock_video.mp4" if source_type == 0 else mock_cameras[0]["ip"], "Аналіз з камери" if source_type else "Аналіз відео")
    print(f"Джерело додано до бази, ID: {video_id}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Кінець джерела або помилка читання кадру")
                break

            current_frame += 1
            display_frame = frame.copy()

            if crossing_zone is None or drawing:
                if drawing and crossing_zone:
                    cv2.rectangle(display_frame, (crossing_zone[0], crossing_zone[1]),
                                  (crossing_zone[2], crossing_zone[3]), (0, 255, 0), 2)
                cv2.putText(display_frame, "Select crossing zone with mouse", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(display_frame, "Press SPACE to confirm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                if not analysis_started:
                    analysis_started = True
                    polygon_str = f"(({crossing_zone[0]},{crossing_zone[1]}),({crossing_zone[2]},{crossing_zone[1]}),({crossing_zone[2]},{crossing_zone[3]}),({crossing_zone[0]},{crossing_zone[3]}))"
                    zone_id = add_zone(video_id, polygon_str)
                    print(f"Зону додано до бази, zone_id: {zone_id}")

                results = model(frame)
                detections = results.pandas().xyxy[0]
                deepsort_detections = []
                for _, row in detections.iterrows():
                    if row['name'] == 'person':
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        w, h = x2 - x1, y2 - y1
                        deepsort_detections.append(([x1, y1, w, h], row['confidence'], row['name']))

                tracks = deepsort.update_tracks(deepsort_detections, frame=frame)
                cv2.rectangle(display_frame, (crossing_zone[0], crossing_zone[1]),
                              (crossing_zone[2], crossing_zone[3]), (0, 0, 255), 2)

                current_people_count = 0
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    feet_x, feet_y = (x1 + x2) // 2, y2
                    if (crossing_zone[0] < feet_x < crossing_zone[2] and
                        crossing_zone[1] < feet_y < crossing_zone[3]):
                        current_people_count += 1
                        if track_id not in track_times:
                            track_times[track_id] = {'enter_frame': current_frame}
                        elif 'exit_frame' not in track_times[track_id]:
                            track_times[track_id]['exit_frame'] = current_frame
                            duration = (current_frame - track_times[track_id]['enter_frame']) * frame_duration
                            entry_time = timedelta(seconds=track_times[track_id]['enter_frame'] * frame_duration)
                            exit_time = timedelta(seconds=current_frame * frame_duration)
                            pedestrian_id = add_pedestrian(zone_id, entry_time, exit_time, video_id)
                            print(f"Пішохода додано, pedestrian_id: {pedestrian_id}")
                            total_time += duration
                            add_analysis(pedestrian_id, duration, current_frame * frame_duration)

                people_per_second.append(current_people_count)
                print(f"Кадр {current_frame} ({current_frame / fps:.2f} сек): {current_people_count} людей")

                elapsed = current_frame * frame_duration
                cv2.putText(display_frame, f"Time in zone: {total_time:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(display_frame, f"Total time: {elapsed:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(display_frame, f"People in zone: {current_people_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("Pedestrian Crossing Analysis", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if crossing_zone and not drawing:
                    print(f"Zone selected: {crossing_zone}")
            elif key == ord('r'):
                crossing_zone = None
                total_time = 0
                analysis_started = False
                current_frame = 0
                people_per_second = []
                track_times = {}
                deepsort.__init__(max_age=30, nn_budget=100, override_track_class=None)
                print("Скидання налаштувань")
            elif key == 27:
                print("Натиснуто ESC, завершення циклу")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        total_video_time = current_frame * frame_duration
        plot_graph(people_per_second, analysis_started, total_video_time, total_time, fps)

if __name__ == "__main__":
    main()