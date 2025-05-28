import pickle
import cv2
import mediapipe as mp
import numpy as np
import time  # For FPS calculation

# Load the pre-trained model
model_dict = pickle.load(open('C:\\Users\\Lenovo\\Desktop\\+\\medius\\model.p', 'rb'))
model = model_dict['model']

# Initialize video capture and Mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)
detected_word = []

# Labels for predictions
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'U', 18: 'V', 19: 'W', 20: 'X', 21: 'Y',
    22: 'friend', 23: 'meet', 24: 'ready', 25: 'like', 26: 'favourite', 27: 'later'
}

# Initialize FPS variables
prev_time = time.time()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()  # Capture video frame
    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    # Set the window to be maximized (while keeping close button visible)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1280, 720)  # Adjust based on screen size

    H, W, _ = frame.shape  # Get frame dimensions
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

    # Process frame for hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                # Normalize landmarks
                data_aux.append((x - min(x_)) / (max(x_) - min(x_)))
                data_aux.append((y - min(y_)) / (max(y_) - min(y_)))

    # Check if data is available for prediction
    if data_aux:
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict gesture
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        print("Detected letter is:", predicted_character)
        detected_word.append(predicted_character)

        # Draw bounding box and prediction on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1,
                    cv2.LINE_AA)

    # Display detected words vertically on the left side with a translucent grey background
    x_position = 50  # Left side
    y_position = 50  # Starting position
    font_scale = 0.7  # Smaller font size
    rect_width = 200  # Background width
    rect_height = 30 * min(len(detected_word[-10:]), 10) + 10  # Dynamic height based on detected signs

    # Draw translucent grey rectangle as background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_position - 10, y_position - 10), (x_position + rect_width, y_position + rect_height), 
                  (100, 100, 100), -1)
    alpha = 0.5  # Transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw detected words as **white text** over the rectangle
    for i, character in enumerate(detected_word[-10:]):  # Show last 10 detected signs
        cv2.putText(frame, character.strip(), (x_position, y_position + (i * 30)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Resize frame while keeping aspect ratio (prevents horizontal stretching)
    aspect_ratio = W / H
    new_width = 1280  # Desired width
    new_height = int(new_width / aspect_ratio)
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
