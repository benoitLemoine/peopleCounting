import cv2 as cv
import random
import math


def getRandomColor():
    return random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)


def createTracker(trackerType):
    if trackerType == "KCF":
        tracker = cv.TrackerKCF_create()
    else:
        print("This tracker's type doesn't exist")
    return tracker


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


def resizeRectBox(rectBox, baseSize, targetSize):
    yDist = targetSize[0] / baseSize[0]
    xDist = targetSize[1] / baseSize[1]

    print(rectBox)

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


def computeRectBoxCenter(rectBox):
    x = (rectBox[1][0] - rectBox[0][0]) / 2
    y = (rectBox[1][1] - rectBox[0][1]) / 2

    return int(x), int(y)


def computeTrackBoxCenter(trackBox):
    x = trackBox[0] + trackBox[2] / 2
    y = trackBox[1] + trackBox[3] / 2

    return int(x), int(y)


def computeDistanceBetweenPoints(p1, p2):
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def doNothing(tracker):
    pass


# Fit functions
def findMaxIoUTracker(trackers, detectedBox, iouFloor):
    bestTracker = None
    maxIou = 0

    for t in trackers:
        if not t.paired:
            value = computeIoU(trackBoxToRectBox(t.trackBox), detectedBox)
            if value > maxIou:
                bestTracker = t
                maxIou = value

    if maxIou > iouFloor:
        return bestTracker, maxIou
    else:
        return None, None


def findClosestTracker(trackers, detectedBox, maxDistanceRatio):
    bestTracker = None
    minDistance = None

    for t in trackers:
        if not t.paired:
            p1 = computeTrackBoxCenter(t.trackBox)
            p2 = computeRectBoxCenter(detectedBox)
            distance = computeDistanceBetweenPoints(p1, p2)
            if minDistance is None or minDistance > distance:
                bestTracker = t
                minDistance = distance

    if bestTracker is not None and maxDistanceRatio * max(bestTracker.trackBox[2],
                                                          bestTracker.trackBox[3]) > minDistance:
        return bestTracker, minDistance
    else:
        return None, None


class Tracker:
    def __init__(self, life):
        self.life = life
        self.activeTime = 0

        self.trackBox = []
        self.paired = False

        self.color = getRandomColor()
        self.counted = False

    def init(self, trackBox):
        self.trackBox = trackBox


class MultiTracker:
    def __init__(self, trackerLife, trackerActiveTime):
        self.trackers = []
        self.trackerLife = trackerLife
        self.trackerActiveTime = trackerActiveTime

    def add(self, tracker, trackBox):
        tracker.init(trackBox)
        self.trackers.append(tracker)

    def matchDetected(self, detectedBoxes, fitFunction, onJustCounted=doNothing, onCounted=doNothing, onNotCounted=doNothing):
        # Find best tracker for each detection box
        if detectedBoxes is not None:
            for b in detectedBoxes:
                b = (b[0], b[1]), (b[2], b[3])
                bestTracker, fitValue = fitFunction(self.trackers, b)

                if bestTracker is not None:
                    bestTracker.paired = True
                    bestTracker.trackBox = rectBoxToTrackBox(b)
                    bestTracker.activeTime += 1
                else:
                    newTracker = Tracker(self.trackerLife)
                    self.add(newTracker, rectBoxToTrackBox(b))

            # Update trackers' life
            self._updateTrackersLife(onJustCounted, onCounted, onNotCounted)
            self.resetPaired()

    def _updateTrackersLife(self, onJustCounted, onCounted, onNotCounted):
        for tracker in self.trackers:
            if tracker.paired:
                tracker.life = self.trackerLife

                if tracker.activeTime > self.trackerActiveTime:
                    if not tracker.counted:
                        onJustCounted(tracker)
                        tracker.counted = True
                    else:
                        onCounted(tracker)
                else:
                    onNotCounted(tracker)
            else:
                tracker.life -= 1
                if tracker.life == 0:
                    self.trackers.remove(tracker)

    def resetPaired(self):
        for t in self.trackers:
            t.paired = False
