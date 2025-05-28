import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

#mediapipe hand detection initialisation
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#process static images with a confidence threshold of 30%
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = [] #store hand keypoints as (x,y)
labels = [] #class names

max_keypoints = 21

for dir_ in os.listdir(DATA_DIR): #reads each class in the dataset
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = np.zeros((max_keypoints, 2))#array for all the keypoints per hand

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image to RGB

        results = hands.process(img_rgb) #hand detection
        if results.multi_hand_landmarks: #if hands are detected
            hand_landmarks = results.multi_hand_landmarks[0] #extract first hand

            for i, landmark in enumerate(hand_landmarks.landmark):
                data_aux[i] = [landmark.x, landmark.y] #insert normalised (x,y)

        data.append(data_aux.flatten()) #converts 2D keypoints array into 1D for easy,accurate storage
        labels.append(dir_)

data = np.asarray(data) #convert to numpy array

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

