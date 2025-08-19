Medius is an ASL interpreter powered by Mediapipe and OpenCV.
It tracks hand gestures in real time and maps them to corresponding American Sign Language letters, words, and numbers.

## Features
- Webcam-based input
- Real-time hand tracking
- ASL alphabet recognition
- List for recognised signs

## Setup
1. Clone the repo: git clone https://github.com/rumaiyza/medius.git
2. Install dependencies: pip install opencv-python mediapipe numpy
3. Run (in order): collect_imgs.py, create_dataset.py, train_classifier.py, and medius.py

## Potential Enhancements
- Support for other sign languages
- Text-to-speech functionality
