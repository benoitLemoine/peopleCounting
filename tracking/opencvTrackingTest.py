import cv2 as cv
import random
import sys
import time

import tracking.tracker as tr

def getRandomColor():
    return random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)


cap = cv.VideoCapture(0)
multiTracker = cv.MultiTracker_create()
# multiTracker = tr.MultiTracker()

res, frame = cap.read()
if not res:
    sys.exit(1)

boxes = cv.selectROIs("Select ROIS", frame)
if len(boxes) < 1:
    sys.exit(1)


colors = []
for i, box in enumerate(boxes):
    colors.append(getRandomColor())
    tracker = cv.TrackerKCF_create()
    # tracker = tr.Tracker("KCF", 8)
    multiTracker.add(tracker, frame, tuple(box))

while cap.isOpened():
    res, frame = cap.read()

    if not res:
        break

    start = time.time()
    res, boxes = multiTracker.update(frame)
    end = time.time()
    enlapsedTime = round((end - start) * 1000, 3)

    print("Update : {} ms".format(enlapsedTime))

    for i, box in enumerate(boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(frame, p1, p2, colors[i], 2)

    cv.imshow("Tracking", frame)

    k = cv.waitKey(20)
    if k == ord("q"):
        break

cv.destroyAllWindows()
cap.release()
