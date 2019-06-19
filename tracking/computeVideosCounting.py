import cv2 as cv
import tensorflow as tf
import numpy as np
import os

import tracking.tracker as tr
from core import utils
from detection.utils import preprocessFrame
from tracking.pairingFunctions import pairWithMaxIou
from tracking.resultsExporter import ResultsExporter
from tracking.utils import getTimeInFrames, resizeTrackBox

from detection.utils import IMAGE_H, IMAGE_W, input_tensor, output_tensors, num_classes, getOnlyDetectedPeople, \
    preprocessFrame

gtBasePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt"
videoBasePath = "/home/benoit/Documents/Stage2A/resources/videos"
txtResBasePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/results"

trackerLifeInSeconds = 0.5
trackerActiveTimeInSeconds = 1

with tf.Session() as sess:
    # For each ground truthfile
    for folder in os.listdir(gtBasePath):
        for file in os.listdir(gtBasePath + "/" + folder):
            videoName = file[:-4]

            # Look for video file corresponding to GT file
            for file in os.listdir(videoBasePath + "/" + folder):
                if file.startswith(videoName):
                    break

            videoPath = videoBasePath + "/" + folder + "/" + file
            gtPath = gtBasePath + "/" + folder + "/" + file

            txtResPath = txtResBasePath + "/" + folder + "/" + videoName + ".txt"

            if os.path.isfile(videoPath) and not os.path.isfile(txtResPath):
                cap = cv.VideoCapture(videoPath)

                exporter = ResultsExporter(txtResPath, videoPath)

                trackerLifeInFrames = getTimeInFrames(trackerLifeInSeconds, cap)
                trackerActiveTimeInFrames = getTimeInFrames(trackerActiveTimeInSeconds, cap)
                multiTracker = tr.MultiTracker(trackerLifeInFrames, trackerActiveTimeInFrames)

                totalFrame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                frameCount = 0
                while True:
                    ret, frame = cap.read()
                    frameCount += 1

                    print("{} :  {} / {}".format(file, frameCount, totalFrame))

                    if not ret:
                        break
                    img_resized = preprocessFrame(frame)
                    boxes, scores = sess.run(output_tensors,
                                             feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
                    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

                    if boxes is not None:
                        # Keeping only box labelled "person"
                        boxes = getOnlyDetectedPeople(boxes, labels)

                        # Match detection box with trackers
                        pairingFunction = lambda trackers, boxes: pairWithMaxIou(trackers, boxes, 0.5)


                        def onJustCounted(tracker):
                            exporter.write(frameCount)


                        def onCounted(tracker):
                            tr.doNothing(tracker)


                        multiTracker.matchDetected(boxes, pairingFunction, onJustCounted=onJustCounted,
                                                   onCounted=onCounted)

                cap.release()
                exporter.close()
