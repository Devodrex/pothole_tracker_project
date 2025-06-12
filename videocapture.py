import cv2

# Open the video or camera
cap = cv2.VideoCapture(r"C:\Users\Administrator\pothole_tracker_project\video.mp4")  # or a video file path like "video.mp4"

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Your loop for reading frames or passing it to YOLO model
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame here
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
