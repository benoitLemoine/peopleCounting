import random
import cv2 as cv

import math


def getRandomColor():
    return random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)


def rectBoxToTrackBox(rectBox):
    x = rectBox[0][0]
    y = rectBox[0][1]

    dx = rectBox[1][0] - rectBox[0][0]
    dy = rectBox[1][1] - rectBox[0][1]

    return x, y, dx, dy


def trackBoxToRectBox(trackBox):
    p1 = (int(trackBox[0]), int(trackBox[1]))
    p2 = (int(trackBox[0] + trackBox[2]), int(trackBox[1] + trackBox[3]))

    return p1, p2


def toIntegerTrackBox(trackBox):
    x = max(0, int(trackBox[0]))
    y = max(0, int(trackBox[1]))
    dx = max(0, int(trackBox[2]))
    dy = max(0, int(trackBox[3]))
    return x, y, dx, dy


def toIntegerRectBox(rectBox):
    x1 = max(0, int(rectBox[0][0]))
    y1 = max(0, int(rectBox[0][1]))
    x2 = max(0, int(rectBox[1][0]))
    y2 = max(0, int(rectBox[1][1]))
    return (x1, y1), (x2, y2)


def resizeRectBox(rectBox, baseSize, targetSize):
    yDist = targetSize[0] / baseSize[0]
    xDist = targetSize[1] / baseSize[1]

    p1x = rectBox[0][0] * xDist
    p1y = rectBox[0][1] * yDist

    p2x = rectBox[1][0] * xDist
    p2y = rectBox[1][1] * yDist

    return (int(p1x), int(p1y)), (int(p2x), int(p2y))


def resizeTrackBox(trackBox, baseSize, targetSize):
    yDist = targetSize[0] / baseSize[0]
    xDist = targetSize[1] / baseSize[1]

    x = trackBox[0] * xDist
    y = trackBox[1] * yDist

    dx = trackBox[2] * xDist
    dy = trackBox[3] * yDist

    return int(x), int(y), int(dx), int(dy)


def computeRectBoxArea(rectBox):
    dx = rectBox[1][0] - rectBox[0][0]
    dy = rectBox[1][1] - rectBox[0][1]
    return abs(dx * dy)


def getTimeInFrames(timeInSeconds, cap):
    return int(timeInSeconds * cap.get(cv.CAP_PROP_FPS))


def doNothing(tracker):
    pass


def computeRectBoxCenter(rectBox):
    x = (rectBox[1][0] + rectBox[0][0]) / 2
    y = (rectBox[1][1] + rectBox[0][1]) / 2

    return int(x), int(y)


def computeTrackBoxCenter(trackBox):
    x = trackBox[0] + trackBox[2] / 2
    y = trackBox[1] + trackBox[3] / 2

    return int(x), int(y)


def computeDistanceBetweenPoints(p1, p2):
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def computeMaxDimensionRectBox(rectBox):
    return max(rectBox[1][0] - rectBox[0][0], rectBox[1][1] - rectBox[0][1])


def computeMinDimensionRectBox(rectBox):
    return min(rectBox[1][0] - rectBox[0][0], rectBox[1][1] - rectBox[0][1])


def computeNormalizedHistogramTrackBox(trackBox, frame):
    tb = toIntegerTrackBox(trackBox)
    trackerRio = frame[tb[0]:tb[0] + tb[2], tb[1]:tb[1] + tb[3]]
    trackerRio = cv.cvtColor(trackerRio, cv.COLOR_BGR2HSV)
    trackerHist = cv.calcHist(trackerRio, [0, 1], None, [50, 60], [0, 180] + [0, 256], accumulate=False)
    return cv.normalize(trackerHist, trackerHist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)


def computeNormalizedHistogramRectBox(rectBox, frame):
    db = toIntegerRectBox(rectBox)
    detectedRio = frame[db[0][0]:db[1][0], db[0][1]:db[1][1]]
    detectedRio = cv.cvtColor(detectedRio, cv.COLOR_BGR2HSV)
    histDetected = cv.calcHist(detectedRio, [0, 1], None, [50, 60], [0, 180] + [0, 256], accumulate=False)
    return cv.normalize(histDetected, histDetected, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)


# Uses rectBox format !
def computeIoU(rectBox1, rectBox2):
    left = max(rectBox1[0][0], rectBox2[0][0])
    right = min(rectBox1[1][0], rectBox2[1][0])
    top = max(rectBox1[0][1], rectBox2[0][1])
    bottom = min(rectBox1[1][1], rectBox2[1][1])

    if left > right or top > bottom:
        return 0

    intersection = computeRectBoxArea(((left, top), (right, bottom)))
    union = computeRectBoxArea(rectBox1) + computeRectBoxArea(rectBox2) - intersection
    return intersection / union
