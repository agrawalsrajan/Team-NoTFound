# Initialize MediaPipe pose and hand solutions
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Masking
from tensorflow.keras.models import Model
import numpy as np

# Define the autoencoder architecture
def build_autoencoder(input_dim):
    input_pose = Input(shape=(input_dim,))
    masked_input = Masking(mask_value=0)(input_pose)  # Mask missing values
    encoded = Dense(64, activation='relu')(masked_input)
    encoded = Dense(32, activation='relu')(encoded)
    latent_space = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(latent_space)
    decoded = Dense(64, activation='relu')(decoded)
    output_pose = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_pose, output_pose)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Assume 33 body key points and 21 hand key points for both hands, each with 3 coordinates
pose_input_dim = 33 * 3  # Adjust as per detected key points
hand_input_dim = 21 * 3
autoencoder_pose = build_autoencoder(pose_input_dim)
autoencoder_hand = build_autoencoder(hand_input_dim)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

# Start webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame for pose and hand landmarks
    pose_results = pose.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    # Collect pose landmarks
    pose_landmarks = []
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            if lm.visibility >= 0.5:
                pose_landmarks.extend([lm.x, 1 - lm.y, lm.z])  # Flip y-axis
            else:
                pose_landmarks.extend([0, 0, 0])  # Placeholder for missing values
    else:
        pose_landmarks = [0] * pose_input_dim  # All zeros if no pose detected

    # Collect hand landmarks
    right_hand_landmarks = []
    left_hand_landmarks = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_landmarks_data = []
            for lm in hand_landmarks.landmark:
                hand_landmarks_data.extend([lm.x, 1 - lm.y, lm.z])

            if hand_label.classification[0].label == "Right":
                right_hand_landmarks = hand_landmarks_data
            else:
                left_hand_landmarks = hand_landmarks_data

    # Fill missing landmarks with 0 if hand not detected
    if not right_hand_landmarks:
        right_hand_landmarks = [0] * hand_input_dim
    if not left_hand_landmarks:
        left_hand_landmarks = [0] * hand_input_dim

    # Convert landmarks to numpy arrays for autoencoder input
    pose_landmarks = np.array(pose_landmarks).reshape(1, -1)
    right_hand_landmarks = np.array(right_hand_landmarks).reshape(1, -1)
    left_hand_landmarks = np.array(left_hand_landmarks).reshape(1, -1)

    # Use autoencoder to predict missing values
    reconstructed_pose = autoencoder_pose.predict(pose_landmarks)
    reconstructed_right_hand = autoencoder_hand.predict(right_hand_landmarks)
    reconstructed_left_hand = autoencoder_hand.predict(left_hand_landmarks)

    # Replace original pose and hand data with reconstructed data for visualization
    # Here, draw only the reconstructed data on the frame
    # This visualization is the same as before

    # Display the frame
    cv2.imshow("Pose and Hand Tracking", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()