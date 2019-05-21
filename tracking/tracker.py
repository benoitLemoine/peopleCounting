import cv2 as cv
import time
import random


def getRandomColor():
    return random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)


def createTracker(trackerType):
    if trackerType == "KCF":
        tracker = cv.TrackerKCF_create()
    else:
        print("This tracker's type doesn't exist")
    return tracker


def rectBoxToTrackBox(box):
    x = box[0][0]
    y = box[0][1]

    dx = box[1][0] - box[0][0]
    dy = box[1][1] - box[0][1]

    return x, y, dx, dy


def trackBoxToRectBox(box):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

    return p1, p2


def resizeRectBox(box, baseSize, targetSize):
    yDist = targetSize[0] / baseSize[0]
    xDist = targetSize[1] / baseSize[1]

    print(box)

    p1x = box[0][0] * xDist
    p1y = box[0][1] * yDist

    p2x = box[1][0] * xDist
    p2y = box[1][1] * yDist

    return (int(p1x), int(p1y)), (int(p2x), int(p2y))


def resizeTrackBox(box, baseSize, targetSize):
    yDist = targetSize[0] / baseSize[0]
    xDist = targetSize[1] / baseSize[1]

    x = box[0] * xDist
    y = box[1] * yDist

    dx = box[2] * xDist
    dy = box[3] * yDist

    return int(x), int(y), int(dx), int(dy)


def computeRectBoxArea(rect):
    dx = rect[1][0] - rect[0][0]
    dy = rect[1][1] - rect[0][1]
    return abs(dx * dy)


# Uses rectBox format !
def computeIoU(box1, box2):
    left = max(box1[0][0], box2[0][0])
    right = min(box1[1][0], box2[1][0])
    top = max(box1[0][1], box2[0][1])
    bottom = min(box1[1][1], box2[1][1])

    if left > right or top > bottom:
        return 0

    intersection = computeRectBoxArea(((left, top), (right, bottom)))
    union = computeRectBoxArea(box1) + computeRectBoxArea(box2) - intersection
    return intersection / union


def findMaxIoUTracker(trackers, detectedBox, iouFloor):
    maxTracker = None
    maxValue = 0

    for t in trackers:
        if not t.paired:
            value = computeIoU(trackBoxToRectBox(t.trackBox), detectedBox)
            if value > maxValue:
                maxTracker = t
                maxValue = value

    if maxValue > iouFloor:
        return maxTracker, maxValue
    else:
        return None, None


class Tracker:
    def __init__(self, trackerType, life):
        self.life = life
        self.color = getRandomColor()
        self.paired = False
        self.tracker = createTracker(trackerType)
        self.trackBox = []

    def init(self, frame, trackBox):
        self.tracker.init(frame, tuple(trackBox))
        self.trackBox = trackBox

    def update(self, frame):
        res, self.trackBox = self.tracker.update(frame)
        return res, self.trackBox


class MultiTracker:
    def __init__(self, trackerType, trackerLife):
        self.trackers = []
        self.trackerType = trackerType
        self.trackerLife = trackerLife

    def add(self, tracker, frame, trackBox):
        tracker.init(frame, trackBox)
        self.trackers.append(tracker)

    def matchDetected(self, detectedBox, fitFunction, frame):
        # Update all trackers to current frame
        self._update(frame)

        # Find best tracker for each detection box
        iouFloor = 0.5
        if detectedBox is not None:
            for b in detectedBox:
                b = (b[0], b[1]), (b[2], b[3])
                bestTracker, iouValue = fitFunction(self.trackers, b, iouFloor)

                if bestTracker is not None:
                    bestTracker.paired = True
                    bestTracker.trackBox = rectBoxToTrackBox(b)
                else:
                    newTracker = Tracker(self.trackerType, self.trackerLife)
                    self.add(newTracker, frame, rectBoxToTrackBox(b))

            # Update trackers' life
            self._updateTrackersLife()

    def _update(self, frame):
        trackBoxes = []
        retRes = True
        for t in self.trackers:
            res, trackBox = t.update(frame)
            trackBoxes.append(trackBox)
            retRes = retRes & res
        return retRes, trackBoxes

    def _updateTrackersLife(self):
        for tracker in self.trackers:
            if tracker.paired:
                tracker.life = self.trackerLife
            else:
                tracker.life -= 1
                if tracker.life == 0:
                    self.trackers.remove(tracker)

    def resetPaired(self):
        for t in self.trackers:
            t.paired = False
