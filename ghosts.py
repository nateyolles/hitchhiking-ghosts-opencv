#!/usr/bin/python3

import cv2
import os
import time
from picamera2 import Picamera2
import numpy as np

window_name = 'frame'
show_video = True
full_screen = True

# download the cascade file from the link below
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml

face_detector = cv2.CascadeClassifier("/home/pi/haarcascade_frontalface_default.xml")
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Create a directory to store detected faces
# output_directory = "detected_faces"
# os.makedirs(output_directory, exist_ok=True)

# black blank image
if not show_video:
    blank_image = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    print(blank_image.shape)
    #cv2.imshow("Black Blank", blank_image)

while True:
    im = picam2.capture_array()
    im = cv2.flip(im, 1)

    if full_screen:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        if show_video:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
        else:
            cv2.rectangle(blank_image, (x, y), (x + w, y + h), (0, 255, 0))

        # Generate a unique filename using timestamp for every saved image
        #timestamp = int(time.time())
        #filename = os.path.join(output_directory, f"face_{timestamp}.jpg")
        #cv2.imwrite(filename, im[y:y+h, x:x+w])  # Save only the detected face portion

    if show_video:
        cv2.imshow(window_name, im)
    else:
        cv2.imshow(window_name, blank_image)

    if cv2.waitKey(25) == ord('q'):
        break

#cap.release()
