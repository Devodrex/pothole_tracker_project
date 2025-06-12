from ultralytics import YOLO
import cv2
import os
import gpxpy
import gpxpy.gpx
from datetime import datetime, timedelta

# === Load Model and Video ===
model_path = r"C:\Users\Administrator\pothole_tracker_project\yolov11_pothole.pt"
video_path = r"C:\Users\Administrator\pothole_tracker_project\video.mp4"
gpx_path = r"C:\Users\Administrator\pothole_tracker_project\data\gpx_data.gpx"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")
if not os.path.exists(gpx_path):
    raise FileNotFoundError(f"GPX file not found: {gpx_path}")

model = YOLO(model_path)

# === Load GPX Timestamps and Coordinates ===
def parse_gpx(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    
    coords = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((point.time, point.latitude, point.longitude))
    return coords

gps_data = parse_gpx(gpx_path)

# === Estimate frame-GPS timestamp mapping ===
# Assuming video start matches first GPX timestamp
video_capture = cv2.VideoCapture(video_path)
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_number = 0
start_time = gps_data[0][0]  # First timestamp in GPX

# === Run Detection and Show Frame with GPS Info ===
results = model(video_path, stream=True)
for result in results:
    frame = result.plot()

    # Estimate timestamp for current frame
    current_time = start_time + timedelta(seconds=frame_number / fps)

    # Find nearest GPS point
    nearest = min(gps_data, key=lambda x: abs(x[0] - current_time))
    timestamp_str = nearest[0].strftime("%H:%M:%S")
    lat, lon = nearest[1], nearest[2]

    # Draw GPS on frame
    cv2.putText(frame, f"Time: {timestamp_str} | Lat: {lat:.5f} | Lon: {lon:.5f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Pothole Detection with GPS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

video_capture.release()
cv2.destroyAllWindows()
