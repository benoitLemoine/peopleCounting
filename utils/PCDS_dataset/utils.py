import cv2 as cv
import numpy as np


def removeCameraWatermark(frame):
    lowRedBottom = np.array([0, 150, 0])
    highRedBottom = np.array([1, 255, 255])

    lowRedMid = np.array([335, 65, 35])
    highRedMid = np.array([345, 75, 55])

    lowRedTop = np.array([175, 150, 0])
    highRedTop = np.array([180, 255, 255])

    frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    print(frameHSV[44][12])

    maskBottom = cv.inRange(frameHSV, lowRedBottom, highRedBottom)
    maskMid = cv.inRange(frameHSV, lowRedMid, highRedMid)
    maskTop = cv.inRange(frameHSV, lowRedTop, highRedTop)
    mask = maskBottom + maskTop + maskMid

    elem = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    mask = cv.dilate(mask, elem)

    cv.rectangle(frame, (40, 8), (48, 16), (0, 185, 0))
    # return mask
    return cv.inpaint(frame, mask, 3, cv.INPAINT_TELEA)
