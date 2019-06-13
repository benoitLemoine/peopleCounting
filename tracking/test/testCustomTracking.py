import statistics
import time

import cv2 as cv
import numpy as np
import tensorflow as tf

import tracking.tracker as tr
from core import utils
from detection.utils import IMAGE_H, IMAGE_W, input_tensor, output_tensors, num_classes, getOnlyDetectedPeople, \
    preprocessFrame
from tracking.pairingFunctions import pairWithMaxIou, pairWithNearestCenter, pairWithHistogramCorrelation
from tracking.resultsExporter import ResultsExporter
from tracking.utils import getTimeInFrames, resizeTrackBox

frameCount = 0
peopleCount = 0

trackerLifeInSecond = 0.5
trackerActiveTimeInSecond = 1

savingResults = False

detectionTime = []
trackingTime = []

videoResBasePath = "/home/benoit/Documents/Stage2A/resources/resultsVideo/tracking"
txtResBasePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/results"
videoPaths = [
    "/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_19_43_00BackColor.avi",
    "/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_18_24_54BackColor.avi",
    "/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/train/ADL-Rundle-6/img1/ADL-Rundle-6.mp4",
    "/home/benoit/Documents/Stage2A/resources/MIVIA_dataset/Dataset People Counting MIVIA/DBc/VIDEOS/RGB/C_I_G_1.mkv",
    "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P2L_S5_C3.1/P2L_S5_C3.1.mp4",
    "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P1E_S1_C1/P1E_S1_C1.mp4"]

with tf.Session() as sess:
    videoPath = videoPaths[5]
    videoName = videoPath.split("/")[-1][:-4]

    videoResPath = videoResBasePath + "/result_hist_" + videoName + ".mp4"
    txtResPath = txtResBasePath + "/CP/" + videoName + ".txt"

    print("Processing {}".format(videoName))

    cap = cv.VideoCapture(videoPath)
    trackerLifeInFrame = getTimeInFrames(trackerLifeInSecond, cap)
    trackerActiveTimeInFrame = getTimeInFrames(trackerActiveTimeInSecond, cap)
    multiTracker = tr.MultiTracker(trackerLifeInFrame, trackerActiveTimeInFrame)

    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    if savingResults:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        writer = cv.VideoWriter(videoResPath, fourcc, 25, (w, h))
        exporter = ResultsExporter(txtResPath, videoPath)

    while True:
        start = time.time()

        frameCount += 1
        res, frame = cap.read()
        if not res:
            break

        # Processing frame
        img_resized = preprocessFrame(frame)

        # Detecting
        s = time.time()
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        e = time.time()
        detectionTime.append(e - s)

        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

        nbrDetected = 0
        nbrTrackers = 0

        if boxes is not None:
            # Keeping only box labelled "person"
            boxes = getOnlyDetectedPeople(boxes, labels)

            # Match detection box with trackers
            # pairingFunction = lambda trackers, boxes: pairWithMaxIou(trackers, boxes, 0.5)
            # pairingFunction = lambda trackers, boxes:  pairWithNearestCenter(trackers, boxes, 1)
            pairingFunction = lambda trackers, boxes: pairWithHistogramCorrelation(trackers, boxes, frame, 0.8, 0.3)

            def onJustCounted(tracker):
                global peopleCount

                if savingResults:
                    exporter.write(frameCount)

                b = resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), tracker.color, -1)
                peopleCount += 1


            def onCounted(tracker):
                b = resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), tracker.color, 2)


            s = time.time()
            multiTracker.matchDetected(boxes, pairingFunction, onJustCounted=onJustCounted, onCounted=onCounted)
            e = time.time()
            trackingTime.append(e - s)

            nbrDetected = len(boxes)
            nbrTrackers = len(multiTracker.trackers)

        cv.imshow("Tracking", frame)
        if savingResults:
            writer.write(frame)

        end = time.time()
        totalTime = round((end - start) * 1000, 3)
        print("[{}] {} trackers / {} detected (update in {} ms)".format(frameCount, nbrTrackers,
                                                                        nbrDetected, totalTime))

        k = cv.waitKey(20)
        if k == ord("q"):
            break

    print("Counted {} people in total on {} frames".format(peopleCount, frameCount))
    print("Detection median time {} ms".format(statistics.median(detectionTime) * 1000))
    print("Tracking median time {} ms".format(statistics.median(trackingTime) * 1000))

    if savingResults:
        writer.release()
        exporter.close()

    cv.destroyAllWindows()
    cap.release()
