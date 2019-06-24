import os
import cv2 as cv
from xml.dom import minidom

videoName = "P2L_S3_C1.2"
gtBasePath = "/home/benoit/Documents/Stage2A/resources/CP_dataset/groundtruth"
videoBasePath = "/home/benoit/Documents/Stage2A/resources/videos/CP"

size = 10

gtPath = gtBasePath + "/" + videoName + ".xml"
videoPath = videoBasePath + "/" + videoName + ".mp4"

if not os.path.exists(gtPath) or not os.path.exists(videoPath):
    print("GT file or video file is missing !")
    exit()

xml = minidom.parse(gtPath)
frames = xml.getElementsByTagName("frame")
cap = cv.VideoCapture(videoPath)

count = 0
k = None
while True:
    ret, img = cap.read()

    if not ret or k == ord("q"):
        break

    for person in frames[count].getElementsByTagName("person") or []:
        lEye = person.getElementsByTagName("leftEye")[0]
        rEye = person.getElementsByTagName("rightEye")[0]

        if lEye:
            lx = int(lEye.attributes["x"].value)
            ly = int(lEye.attributes["y"].value)
            cv.circle(img, (lx, ly), size, (185, 0, 0), -1)

        if rEye:
            rx = int(rEye.attributes["x"].value)
            ry = int(rEye.attributes["y"].value)
            cv.circle(img, (rx, ry), size, (185, 0, 0), -1)

    count += 1
    cv.imshow("result", img)
    k = cv.waitKey(30)

cap.release()
cv.destroyAllWindows()
