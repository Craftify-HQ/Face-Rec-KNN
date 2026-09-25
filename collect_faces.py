import cv2
import os
cap = cv2.VideoCapture(0)
name = 'nod'
os.makedirs(name, exist_ok=True)
j = 1
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (800, 200), (1200, 750), (0, 0, 255), 2)
    face = cv2.cvtColor(frame[100:650, 800:1200], cv2.COLOR_BGR2GRAY)
    cv2.imshow('Face Cropped', face)
    cv2.imshow('Full Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(name + '/' + str(j) + '.jpg', face)
        j += 1