import statistics
import time

import cv2 as cv
import numpy as np
import tensorflow as tf

import tracking.tracker as tr
from core import utils
from detection.utils import IMAGE_H, IMAGE_W, input_tensor, output_tensors, num_classes, getOnlyDetectedPeople, \
    preprocessFrame

frameCount = 0
peopleCount = 0

trackerLife = 10
trackerActiveTime = 25

detectionTime = []
trackingTime = []

destPath = "/home/benoit/Documents/Stage2A/resources/results/tracking"
videoPaths = [
    "/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_19_43_00BackColor.avi",
    "/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_18_24_54BackColor.avi",
    "/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/train/ADL-Rundle-6/img1/ADL-Rundle-6.mp4",
    "/home/benoit/Documents/Stage2A/resources/MIVIA_dataset/Dataset People Counting MIVIA/DBc/VIDEOS/RGB/C_I_G_1.mkv",
    "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P2L_S5_C3.1/P2L_S5_C3.1.mp4",
    "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P1E_S1_C1/P1E_S1_C1.mp4"]

with tf.Session() as sess:
    videoPath = videoPaths[4]
    videoName = videoPath.split("/")[-1]
    resultPath = destPath + "/result" + videoName

    cap = cv.VideoCapture(videoPath)
    multiTracker = tr.MultiTracker(trackerLife, trackerActiveTime)

    first = True
    while True:
        start = time.time()

        frameCount += 1
        res, frame = cap.read()
        if not res:
            break

        if first:
            print("Processing {}".format(videoName))
            h, w, l = frame.shape
            size = w, h
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            writer = cv.VideoWriter(resultPath, fourcc, 25, size)
            first = False

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
            # fitFunction = lambda tracker, box:  tr.findClosestTracker(tracker, box, 0.5)
            fitFunction = lambda tracker, box: tr.findMaxIoUTracker(tracker, box, 0.5)

            def onJustCounted(tracker):
                global peopleCount
                b = tr.resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), tracker.color, -1)
                peopleCount += 1

            def onCounted(tracker):
                b = tr.resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), tracker.color, 2)


            s = time.time()
            multiTracker.matchDetected(boxes, fitFunction, onJustCounted=onJustCounted, onCounted=onCounted)
            e = time.time()
            trackingTime.append(e - s)

            nbrDetected = len(boxes)
            nbrTrackers = len(multiTracker.trackers)

        cv.imshow("Tracking", frame)
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

    cv.destroyAllWindows()
    writer.release()
    cap.release()
