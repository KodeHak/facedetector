# Made by CodeHack

import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade alogrithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
webcam = cv2.VideoCapture(0)



# iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 10)

    cv2.imshow('CodeHack Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break


print("Code Completed")