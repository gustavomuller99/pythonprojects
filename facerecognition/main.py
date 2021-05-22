import cv2 as cv
import os
import numpy as np
import easygui

def detectAndDisplay(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    faces = faceCasc.detectMultiScale(gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE)


    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, x+h), (0, 255, 0), 2)

    cv.imshow('video', frame)

filetype = ["*.mp4"]
f = easygui.fileopenbox()

cap = cv.VideoCapture(f)
casc = "haarcascade_frontalface_alt2.xml"
faceCasc = cv.CascadeClassifier(casc)

while cap.isOpened():
    ret, frame = cap.read()

    detectAndDisplay(frame)
    if cv.waitKey(50) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
