import os
import cv2

DATA_DIR = './data'#setting dataset path
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28
dataset_size = 50

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):#creating paths for each class
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')
    collecting = False  #flag to start/stop collection
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        #display Instructions
        if not collecting:
            cv2.putText(frame, 'Press "C" to start collection', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('c'):  # Start collecting when 'C' is pressed
            collecting = True
            print("Started collecting...")

        if key == ord('q'):  # Stop collection when 'Q' is pressed
            print("Stopping collection for this class.")
            break

        if collecting and counter < dataset_size:
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

cap.release()
cv2.destroyAllWindows()
