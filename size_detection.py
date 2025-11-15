# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time
# # import pickle
# # from collections import deque
# # from pykalman import KalmanFilter

# # # Load ML Model for Accurate Size Prediction
# # with open("size_model.pkl", "rb") as model_file:
# #     model = pickle.load(model_file)

# # # Initialize MediaPipe Pose
# # mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# # # Open Webcam
# # cap = cv2.VideoCapture(0)
# # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# # # Smoothing parameters
# # height_queue = deque(maxlen=10)
# # shoulder_width_queue = deque(maxlen=10)

# # # Kalman Filter for Height Estimation
# # kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
# # kf = kf.em(np.array([0]), n_iter=5)
# # state_means = [0]

# # frame_count = 0
# # process_interval = 5  # Process every 5th frame

# # def predict_size_ml(height, shoulder_width, chest_width, waist_width):
# #     """Map body measurements to real-time brand size chart and return a single recommended size."""
# #     input_data = np.array([[height, shoulder_width, chest_width, waist_width]])
    
# #     # Predict raw numerical size using ML model
# #     numerical_size = model.predict(input_data)[0]

# #     # Define standard size mapping based on brand charts (example values)
# #     brand_size_chart = {
# #         "S": (150, 160, 35, 40, 28, 32),
# #         "M": (161, 170, 41, 45, 33, 37),
# #         "L": (171, 180, 46, 50, 38, 42),
# #         "XL": (181, 190, 51, 55, 43, 47)
# #     }

# #     # Find the best-matching size
# #     for size, (h_min, h_max, s_min, s_max, w_min, w_max) in brand_size_chart.items():
# #         if h_min <= height <= h_max and s_min <= shoulder_width <= s_max and w_min <= waist_width <= w_max:
# #             return size  # Return only one best-matching size

# #     return "Unknown"  # Fallback if no exact match is found

# # def get_body_size(landmarks, frame_width, frame_height):
# #     """Calculate height, shoulder width, chest width, waist width & map to size."""
# #     try:
# #         height = abs(int((landmarks[mp_pose.PoseLandmark.NOSE].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) * frame_height))
# #         shoulder_width = abs(int((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x) * frame_width))
# #         chest_width = abs(int((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.LEFT_CHEST].x) * frame_width))
# #         waist_width = abs(int((landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x - landmarks[mp_pose.PoseLandmark.LEFT_HIP].x) * frame_width))

# #         # Update Kalman Filter for height smoothing
# #         state_means.append(kf.filter_update(state_means[-1], observation=np.array([height]))[0])

# #         # Smoothed values
# #         smoothed_height = np.mean(state_means[-10:])
# #         height_queue.append(smoothed_height)
# #         shoulder_width_queue.append(shoulder_width)

# #         # Averages for accuracy
# #         avg_height_cm = sum(height_queue) / len(height_queue)
# #         avg_shoulder_width = sum(shoulder_width_queue) / len(shoulder_width_queue)
# #         avg_chest_width = chest_width
# #         avg_waist_width = waist_width

# #         # Predict size using AI model
# #         size = predict_size_ml(avg_height_cm, avg_shoulder_width, avg_chest_width, avg_waist_width)

# #         return avg_height_cm, avg_shoulder_width, avg_chest_width, avg_waist_width, size
# #     except Exception as e:
# #         print("Error in get_body_size:", e)
# #         return 0, 0, 0, 0, "Unknown"

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         print("Failed to grab frame")
# #         break

# #     frame_height, frame_width, _ = frame.shape
# #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     results = pose.process(img_rgb)

# #     if results.pose_landmarks and frame_count % process_interval == 0:
# #         landmarks = results.pose_landmarks.landmark
# #         height_cm, shoulder_width, chest_width, waist_width, size = get_body_size(landmarks, frame_width, frame_height)

# #         print(f"Height: {height_cm:.2f} cm, Shoulder Width: {shoulder_width} px, Chest Width: {chest_width} px, Waist Width: {waist_width} px, Size: {size}")

# #         cv2.putText(frame, f"Height: {int(height_cm)} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# #         cv2.putText(frame, f"Shoulder Width: {shoulder_width} px", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# #         cv2.putText(frame, f"Chest Width: {chest_width} px", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
# #         cv2.putText(frame, f"Waist Width: {waist_width} px", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# #         cv2.putText(frame, f"Recommended Size: {size}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# #         # Draw Pose Landmarks
# #         mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# #     frame_count += 1
# #     cv2.imshow("TRYZA Size Detection", frame)

# #     # Keep the window open and exit on 'q' key
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()
# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe Pose Model
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# # Start Webcam
# cap = cv2.VideoCapture(0)

# # Known Reference Height (Set based on an object in the frame)
# KNOWN_HEIGHT_CM = 170  # Change this based on a real measurement

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert Image to RGB (for MediaPipe)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Detect Body Landmarks
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark

#         # Extract Key Points (Shoulder, Chest, Waist)
#         left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
#         right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

#         # Convert Normalized Points to Pixel Coordinates
#         h, w, _ = frame.shape
#         left_shoulder_x, left_shoulder_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
#         right_shoulder_x, right_shoulder_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
#         left_hip_x, left_hip_y = int(left_hip.x * w), int(left_hip.y * h)
#         right_hip_x, right_hip_y = int(right_hip.x * w), int(right_hip.y * h)

#         # Compute Body Widths in Pixels
#         shoulder_width_px = np.linalg.norm([left_shoulder_x - right_shoulder_x, 
#                                             left_shoulder_y - right_shoulder_y])
#         waist_width_px = np.linalg.norm([left_hip_x - right_hip_x, 
#                                          left_hip_y - right_hip_y])
        
#         # Compute Body Height in Pixels (Using Shoulder to Hip Distance)
#         body_height_px = np.linalg.norm([left_shoulder_y - left_hip_y])

#         # Pixel-to-CM Conversion (Using Known Height)
#         px_to_cm_ratio = KNOWN_HEIGHT_CM / body_height_px
#         shoulder_width_cm = shoulder_width_px * px_to_cm_ratio
#         waist_width_cm = waist_width_px * px_to_cm_ratio

#         # Draw Bounding Box & Measurements
#         cv2.line(frame, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (0, 255, 0), 3)
#         cv2.line(frame, (left_hip_x, left_hip_y), (right_hip_x, right_hip_y), (255, 0, 0), 3)

#         cv2.putText(frame, f"Shoulder: {shoulder_width_cm:.1f} cm", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, f"Waist: {waist_width_cm:.1f} cm", (50, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     # Display the Result
#     cv2.imshow("Body Measurement Detection", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release Resources
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
from pykalman import KalmanFilter

# Load ML Model for Accurate Size Prediction
with open("size_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start Webcam
cap = cv2.VideoCapture(0)

# Buffer for Smoothing
height_queue = deque(maxlen=10)
shoulder_width_queue = deque(maxlen=10)

# Kalman Filter for Height Estimation
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
data = np.array([0, 1])  # Ensure at least 2 timesteps

if len(data) < 2:
    print("Error: Not enough data for Kalman filter EM algorithm.")
else:
    kf = kf.em(data, n_iter=5)
state_means = [0]

frame_count = 0
process_interval = 5  # Process every 5th frame for efficiency

def predict_size_ml(height, shoulder_width, chest_width, waist_width):
    """Predicts best-fitting clothing size using ML model & brand size chart."""
    input_data = np.array([[height, shoulder_width, chest_width, waist_width]])
    
    # Predict raw numerical size using ML model
    numerical_size = model.predict(input_data)[0]

    # Example Brand Size Chart Mapping
    brand_size_chart = {
        "S": (150, 160, 35, 40, 28, 32),
        "M": (161, 170, 41, 45, 33, 37),
        "L": (171, 180, 46, 50, 38, 42),
        "XL": (181, 190, 51, 55, 43, 47)
    }

    # Find the best-matching size
    for size, (h_min, h_max, s_min, s_max, w_min, w_max) in brand_size_chart.items():
        if h_min <= height <= h_max and s_min <= shoulder_width <= s_max and w_min <= waist_width <= w_max:
            return size  # Return best-matching size

    return "Unknown"  # Default fallback size

def get_body_size(landmarks, frame_width, frame_height):
    """Calculates height, shoulder width, chest width, waist width & maps to size."""
    try:
        height = abs(int((landmarks[mp_pose.PoseLandmark.NOSE].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) * frame_height))
        shoulder_width = abs(int((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x) * frame_width))
        chest_width = abs(int((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.LEFT_CHEST].x) * frame_width))
        waist_width = abs(int((landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x - landmarks[mp_pose.PoseLandmark.LEFT_HIP].x) * frame_width))

        # Update Kalman Filter for height smoothing
        state_means.append(kf.filter_update(state_means[-1], observation=np.array([height]))[0])

        # Smoothed values
        smoothed_height = np.mean(state_means[-10:])
        height_queue.append(smoothed_height)
        shoulder_width_queue.append(shoulder_width)

        # Averages for accuracy
        avg_height_cm = sum(height_queue) / len(height_queue)
        avg_shoulder_width = sum(shoulder_width_queue) / len(shoulder_width_queue)
        avg_chest_width = chest_width
        avg_waist_width = waist_width

        # Predict size using AI model
        size = predict_size_ml(avg_height_cm, avg_shoulder_width, avg_chest_width, avg_waist_width)

        return avg_height_cm, avg_shoulder_width, avg_chest_width, avg_waist_width, size
    except Exception as e:
        print("Error in get_body_size:", e)
        return 0, 0, 0, 0, "Unknown"

# Real-time Detection Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_height, frame_width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks and frame_count % process_interval == 0:
        landmarks = results.pose_landmarks.landmark
        height_cm, shoulder_width, chest_width, waist_width, size = get_body_size(landmarks, frame_width, frame_height)

        print(f"Height: {height_cm:.2f} cm, Shoulder Width: {shoulder_width} px, Chest Width: {chest_width} px, Waist Width: {waist_width} px, Size: {size}")

        cv2.putText(frame, f"Height: {int(height_cm)} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Shoulder Width: {shoulder_width} px", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Chest Width: {chest_width} px", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Waist Width: {waist_width} px", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Recommended Size: {size}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw Pose Landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame_count += 1
    cv2.imshow("TRYZA Size Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()