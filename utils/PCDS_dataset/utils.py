import cv2 as cv
import numpy as np


def removeCameraWatermark(frame):
    lowRedBottom = np.array([0, 100, 0])
    highRedBottom = np.array([4, 255, 255])

    lowRedTop = np.array([325, 55, 35])
    highRedTop = np.array([345, 75, 55])

    lowRedMid = np.array([145, 70, 0])
    highRedMid = np.array([180, 255, 255])

    rois = [frame[0:21, 0:189], frame[30:92, 178:292]]
    processedRois = []

    for roi in rois:
        frameHSV = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        maskBottom = cv.inRange(frameHSV, lowRedBottom, highRedBottom)
        maskMid = cv.inRange(frameHSV, lowRedMid, highRedMid)
        maskTop = cv.inRange(frameHSV, lowRedTop, highRedTop)
        mask = maskBottom + maskTop + maskMid

        processedRois.append(cv.inpaint(roi, mask, 3, cv.INPAINT_TELEA))

    frame[0:21, 0:189] = processedRois[0]
    frame[30:92, 178:292] = processedRois[1]

    return frame
