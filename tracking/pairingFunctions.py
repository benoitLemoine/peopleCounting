import cv2 as cv

from tracking.utils import computeIoU, trackBoxToRectBox, computeTrackBoxCenter, computeRectBoxCenter, \
    computeDistanceBetweenPoints, rectBoxToTrackBox, \
    computeNormalizedHistogramTrackBox, computeNormalizedHistogramRectBox


def pairWithMaxIou(trackers, detectedBoxes, minIou):
    notPairedBoxes = []

    for detectedBox in detectedBoxes or []:
        detectedBox = (detectedBox[0], detectedBox[1]), (detectedBox[2], detectedBox[3])
        bestTracker = None
        maxIou = None

        for t in trackers:
            if not t.paired:
                iou = computeIoU(trackBoxToRectBox(t.trackBox), detectedBox)
                if bestTracker is None or iou > maxIou:
                    bestTracker = t
                    maxIou = iou

        if bestTracker and maxIou > minIou:
            bestTracker.paired = True
            bestTracker.trackBox = rectBoxToTrackBox(detectedBox)
            bestTracker.activeTime += 1
        else:
            notPairedBoxes.append(detectedBox)

    return notPairedBoxes


def pairWithNearestCenter(trackers, detectedBoxes, maxDistanceRatio):
    notPairedBoxes = []

    for detectedBox in detectedBoxes or []:
        detectedBox = (detectedBox[0], detectedBox[1]), (detectedBox[2], detectedBox[3])
        bestTracker = None
        minDistance = None

        for t in trackers:
            if not t.paired:
                p1 = computeTrackBoxCenter(t.trackBox)
                p2 = computeRectBoxCenter(detectedBox)
                distance = computeDistanceBetweenPoints(p1, p2)
                if bestTracker is None or minDistance > distance:
                    bestTracker = t
                    minDistance = distance

        detectedBoxDim = max(detectedBox[1][0] - detectedBox[0][0], detectedBox[1][1] - detectedBox[0][1])
        if bestTracker and maxDistanceRatio * detectedBoxDim > minDistance:
            bestTracker.paired = True
            bestTracker.trackBox = rectBoxToTrackBox(detectedBox)
            bestTracker.activeTime += 1
        else:
            notPairedBoxes.append(detectedBox)

    return notPairedBoxes


def pairWithHistogramCorrelation(trackers, detectedBoxes, frame, minCorrelation):
    notPairedBoxes = []

    # Compute all tracker histograms first
    histograms = []
    for tracker in trackers:
        trackerHist = computeNormalizedHistogramTrackBox(tracker.trackBox, frame)
        histograms.append((tracker, trackerHist))

    for detectedBox in detectedBoxes:
        bestTracker = None
        maxCorrelation = None
        detectedBox = (detectedBox[0], detectedBox[1]), (detectedBox[2], detectedBox[3])
        detectedBoxHist = computeNormalizedHistogramRectBox(detectedBox, frame)

        for tracker, trackerHist in histograms:
            if not tracker.paired:
                correlation = cv.compareHist(detectedBoxHist, trackerHist, cv.HISTCMP_CORREL)

                if bestTracker is None or maxCorrelation < correlation:
                    bestTracker = tracker
                    maxCorrelation = correlation

        if bestTracker and maxCorrelation > minCorrelation:
            bestTracker.paired = True
            bestTracker.trackBox = rectBoxToTrackBox(detectedBox)
            bestTracker.activeTime += 1
        else:
            notPairedBoxes.append(detectedBox)

    return notPairedBoxes
