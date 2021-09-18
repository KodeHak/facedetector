# Made by CodeHack

import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade alogrithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('side2side.jpg')

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img,(x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)

#print(face_coordinates)

# Display the img with faces
cv2.imshow('CodeHack Face Detector', img)
cv2.waitKey()

print("Code Completed")