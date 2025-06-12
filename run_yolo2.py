from ultralytics import YOLO
import cv2
import os
import csv
import gpxpy
from datetime import datetime, timedelta, timezone

# Paths
model_path = r"C:\Users\Administrator\pothole_tracker_project\yolov11_pothole.pt"
video_path = r"C:\Users\Administrator\pothole_tracker_project\video.mp4"
gpx_path = r"C:\Users\Administrator\pothole_tracker_project\data\gpx_data.gpx"
output_video_path = r"C:\Users\Administrator\pothole_tracker_project\output_annotated_video.mp4"
csv_log_path = r"C:\Users\Administrator\pothole_tracker_project\pothole_detections.csv"

# Validate files
for path in [model_path, video_path, gpx_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Parse GPX data and convert all times to UTC
def parse_gpx_timestamps(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append({
                        'time': point.time.replace(tzinfo=timezone.utc),
                        'lat': point.latitude,
                        'lon': point.longitude
                    })
        return points

gps_points = parse_gpx_timestamps(gpx_path)
start_time = gps_points[0]['time'] if gps_points else datetime.now(timezone.utc)

# Load model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# CSV logging setup
with open(csv_log_path, 'w', newline='') as csvfile:
    fieldnames = ['frame', 'timestamp', 'latitude', 'longitude', 'pothole_count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    results = model(video_path, stream=True)

    for frame_id, result in enumerate(results):
        frame = result.plot()  # Annotated frame
        pothole_count = len(result.boxes)

        # Estimate video timestamp in UTC
        video_time = (start_time + timedelta(seconds=frame_id / fps)).replace(tzinfo=timezone.utc)

        # Find closest matching GPS point
        gps_data = next((p for p in gps_points if p['time'] >= video_time), None)

        # Annotate frame
        if gps_data:
            lat, lon = gps_data['lat'], gps_data['lon']
            label = f"Potholes: {pothole_count} | Lat: {lat:.6f}, Lon: {lon:.6f}"
        else:
            lat, lon = 'N/A', 'N/A'
            label = f"Potholes: {pothole_count} | GPS: Unknown"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Log to CSV
        writer.writerow({
            'frame': frame_id,
            'timestamp': video_time.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'pothole_count': pothole_count
        })

        # Show and write to output
        cv2.imshow("YOLO Pothole Detection", frame)
        out_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
out_writer.release()
cv2.destroyAllWindows()

print("\n✅ Processing complete.")
print(f"📹 Annotated video saved to: {output_video_path}")
print(f"📝 Detection log saved to: {csv_log_path}")
