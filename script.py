import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import numpy as np

# Initialize model
model = YOLO("train5/weights/best.pt")  # Load your trained model
cap = cv2.VideoCapture("test/Задание.mp4")  # Your video file
assert cap.isOpened(), "Error reading video file"

# Video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region line
region_line = [(0, 250), (640, 250)]  # Horizontal line

# Object tracking and counting variables
object_tracker = {}  # Object trackers
object_counter = {'in': 0, 'out': 0}  # Counters
next_id = 0  # Unique ID for each object
lost_frames = {}  # Counter for lost detections
max_lost_frames = 10  # Maximum allowed lost frames before object is discarded

# Kalman filter for prediction
class Kalman:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # x: (x, y, vx, vy), z: (x, y)
        self.kf.F = np.array([[1., 0., 1., 0.],  # State transition matrix
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],  # Observation matrix
                              [0., 1., 0., 0.]])
        self.kf.R *= 1.0  # Measurement noise
        self.kf.P *= 500.  # Covariance matrix
        self.kf.Q[-1, -1] *= 0.01  # Process noise

    def predict(self, cx, cy):
        self.kf.predict()
        self.kf.update([cx, cy])
        return self.kf.x[:2]

# Function to calculate IoU
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Check line crossing
def is_crossing_line(p1, p2, line_y):
    """Check line crossing."""
    return (p1[1] < line_y and p2[1] >= line_y) or (p1[1] >= line_y and p2[1] < line_y)

# Set up VideoWriter to save the processed video
video_writer = cv2.VideoWriter("/app/output/object_counting_output_final3.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

if not video_writer.isOpened():
    print("Error opening video writer")

# Process video frames
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
kalman_filters = {}  # Kalman filters for each object
crossed_objects = set()  # Track objects that already crossed the line
min_size = 50  # Minimum size for detections

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processed.")
        break

    frame_count += 1

    # Detect objects
    results = model(frame)  # Get results
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections: [x1, y1, x2, y2, confidence, class]

    # Filter detections by size
    filtered_detections = []
    for i, det1 in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det1
        width, height = x2 - x1, y2 - y1
        if width > min_size and height > min_size:  # Filter by size
            overlap = False
            for j, det2 in enumerate(detections):
                if i != j:
                    iou = calculate_iou(det1[:4], det2[:4])
                    if iou > 0.5:  # IoU threshold for overlapping
                        overlap = True
                        break
            if not overlap:
                filtered_detections.append(det1)

    # Update tracker
    new_tracker = {}
    for detection in filtered_detections:
        x1, y1, x2, y2, conf, cls = detection
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Object center
        found = False

        # Match with existing objects
        for obj_id, (prev_cx, prev_cy) in object_tracker.items():
            if abs(cx - prev_cx) < 50 and abs(cy - prev_cy) < 50:  # If object is nearby
                new_tracker[obj_id] = (cx, cy)
                found = True

                # Check line crossing
                if obj_id not in crossed_objects:
                    if is_crossing_line((prev_cx, prev_cy), (cx, cy), region_line[0][1]):
                        if prev_cy < region_line[0][1]:
                            object_counter['in'] += 1
                        else:
                            object_counter['out'] += 1
                        crossed_objects.add(obj_id)  # Mark object as crossed
                break

        # If object is new, add to tracker
        if not found:
            new_tracker[next_id] = (cx, cy)
            kalman_filters[next_id] = Kalman()  # Initialize Kalman filter for new object
            kalman_filters[next_id].predict(cx, cy)  # Initialize Kalman filter state
            lost_frames[next_id] = 0  # Initialize lost frame count
            next_id += 1

    # Handle lost objects
    for obj_id in object_tracker.keys():
        if obj_id not in new_tracker:
            lost_frames[obj_id] += 1
            if lost_frames[obj_id] <= max_lost_frames:
                # Use Kalman filter to predict position
                predicted_pos = kalman_filters[obj_id].predict(*object_tracker[obj_id])
                new_tracker[obj_id] = predicted_pos
            else:
                # Remove object if lost for too long
                lost_frames.pop(obj_id)
                kalman_filters.pop(obj_id)

    object_tracker = new_tracker

    # Draw red line
    cv2.line(frame, region_line[0], region_line[1], (0, 0, 255), 2)  # Red line

    # Display IN/OUT values on a white rectangle
    cv2.rectangle(frame, (w - 225, 10), (w - 5, 90), (255, 255, 255), -1)  # White background
    cv2.putText(frame, f"In: {object_counter['in']}", (w - 215, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 128), 1, lineType=cv2.LINE_AA)  # Center alignment
    cv2.putText(frame, f"Out: {object_counter['out']}", (w - 215, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 128), 1, lineType=cv2.LINE_AA)  # Center alignment

    # Calculate result (OUT - IN) on each frame
    result = object_counter['out'] - object_counter['in']

    # Display result next to IN
    cv2.putText(frame, f"Result: {result}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 104, 24), 1, lineType=cv2.LINE_AA)  # Green color

    # Draw bounding boxes for all objects
    for detection in filtered_detections:
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green bounding box

    # Write frame to video file
    video_writer.write(frame)
    print(f"Кадр {frame_count}/{total_frames} обработан.")

# Release resources
cap.release()
video_writer.release()
