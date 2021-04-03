import cv2
import numpy as np
from time import sleep

minWidth = 80  #
minHeight = 80  #

offset = 6  # Error allowed between pixel

linePos = 550 #  Position of the counting line

delay = 60  # FPS

detect = []
vehicle = 0


def vehicleCenter(x, y, w, h):  ##Shows Center of object being deteced
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


videoStream = cv2.VideoCapture('video.mp4')
imageDiffrence = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    e, frame1 = videoStream.read()
    time = float(1 / delay)
    sleep(time)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = imageDiffrence.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    outline, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, linePos), (1200, linePos), (255, 127, 0), 3)
    for (i, c) in enumerate(outline):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_outline = (w >= minWidth) and (h >= minHeight)
        if not validate_outline:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = vehicleCenter(x, y, w, h)                                      ## Ask vehicleCenter to send center of object
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (linePos + offset) and y > (linePos - offset):
                vehicle += 1
                cv2.line(frame1, (25, linePos), (1200, linePos), (0, 127, 255), 3)
                detect.remove((x, y))
                print("vehicle is detected : " + str(vehicle))

    cv2.putText(frame1, "VEHICLE COUNT : " + str(vehicle), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detect", dilated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
videoStream.release()