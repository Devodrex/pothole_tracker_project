from ultralytics import YOLO
import cv2
import os
import csv
import gpxpy
from yolox.tracker.byte_tracker import BYTETracker
from datetime import datetime, timedelta, timezone
import numpy as np
import json

# ========== CONFIG ==========
model_path = "best.pt"
video_path = "video.mp4"
gpx_path = "data/gpx_data.gpx"
output_video_path = "output_tracked_video.mp4"
csv_log_path = "tracked_pothole_log.csv"
summary_path = "summary_report.json"

# ========== LOAD GPS ==========
def parse_gpx(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
        return [
            {'time': point.time.replace(tzinfo=timezone.utc), 'lat': point.latitude, 'lon': point.longitude}
            for track in gpx.tracks for seg in track.segments for point in seg.points
        ]

gps_points = parse_gpx(gpx_path)
start_time = gps_points[0]['time'] if gps_points else datetime.now(timezone.utc)

# ========== INIT YOLO + ByteTrack ==========
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Tracker config
class TrackerArgs:
    track_thresh = 0.4
    match_thresh = 0.8
    buffer_size = 30
    track_buffer = 30
    min_box_area = 10
    mot20 = False

args = TrackerArgs()
tracker = BYTETracker(args, frame_rate=30)

# For unique pothole counting
unique_ids = set()

# ========== LOGGING ==========
with open(csv_log_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['frame', 'object_id', 'timestamp', 'latitude', 'longitude'])
    writer.writeheader()

    results = model(video_path, stream=True)

    for frame_id, result in enumerate(results):
        frame = result.orig_img
        dets = result.boxes

        # Convert detections to format [x1, y1, x2, y2, conf, cls]
        if dets is None or len(dets) == 0:
            out_writer.write(frame)
            continue

        detections = []
        for box in dets:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            detections.append([*xyxy, conf, cls])
        detections = np.array(detections)

        # Run ByteTrack
        online_targets = tracker.update(detections, [height, width], [height, width])

        # Annotate frame time
        video_time = (start_time + timedelta(seconds=frame_id / fps)).replace(tzinfo=timezone.utc)
        gps_data = next((p for p in gps_points if p['time'] >= video_time), None)

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            unique_ids.add(tid)

            x1, y1, w, h = map(int, tlwh)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Log to CSV
            writer.writerow({
                'frame': frame_id,
                'object_id': tid,
                'timestamp': video_time.isoformat(),
                'latitude': gps_data['lat'] if gps_data else 'N/A',
                'longitude': gps_data['lon'] if gps_data else 'N/A'
            })

        # Overlay total pothole count
        cv2.putText(frame, f"Total Potholes Detected: {len(unique_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_writer.write(frame)
        cv2.imshow("Tracked", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out_writer.release()
cv2.destroyAllWindows()

# ========== SUMMARY JSON ==========
summary = {
    "total_unique_potholes": len(unique_ids),
    "start_time": gps_points[0]['time'].isoformat() if gps_points else "N/A",
    "end_time": gps_points[-1]['time'].isoformat() if gps_points else "N/A",
    "start_location": {
        "latitude": gps_points[0]['lat'],
        "longitude": gps_points[0]['lon']
    } if gps_points else "N/A",
    "end_location": {
        "latitude": gps_points[-1]['lat'],
        "longitude": gps_points[-1]['lon']
    } if gps_points else "N/A",
    "video_duration_seconds": round(total_frames / fps, 2) if cap else "N/A"
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

# Final prints
print(f"\n✅ Processing complete.")
print(f"📹 Annotated video saved to: {output_video_path}")
print(f"📝 CSV log saved to: {csv_log_path}")
print(f"📊 Summary report saved to: {summary_path}")
print(f"🚧 Total unique potholes detected: {summary['total_unique_potholes']}")
