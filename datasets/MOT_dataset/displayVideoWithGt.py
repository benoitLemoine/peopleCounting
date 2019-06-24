import os
import cv2 as cv

gtBasePath = "/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/train"
videoName = "ADL-Rundle-8"

gtPath = gtBasePath + "/" + videoName + "/gt/gt.txt"
videoPath = "/home/benoit/Documents/Stage2A/resources/videos/MOT/" + videoName + ".mp4"

lineWidth = 2

if not os.path.exists(gtPath) or not os.path.exists(videoPath):
    print("GT file or video file is missing !")
    exit()

file = open(gtPath)
cap = cv.VideoCapture(videoPath)

k = None
line = None
count = 0
while True:
    ret, img = cap.read()

    if not ret or k == ord("q"):
        break

    if not line:
        line = file.readline().split(",")

    while int(line[0]) == count:
        x = int(float(line[2]))
        y = int(float(line[3]))
        dx = int(float(line[4]))
        dy = int(float(line[5]))

        cv.rectangle(img, (x, y), (x+dx, y+dy), (185, 0, 0), lineWidth)

        line = file.readline().split(",")

    count += 1
    cv.imshow("result", img)
    k = cv.waitKey(30)

cap.release()
cv.destroyAllWindows()

