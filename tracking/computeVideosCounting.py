import os
import cv2 as cv

import tracking.tracker as tr

from detection.yolov3.yolov3 import YoloV3Net
from tracking.pairingFunctions import pairWithMaxIou
from tracking.resultsExporter import ResultsExporter
from tracking.utils import getTimeInFrames

gtBasePath = "../../resources/resultsTxt/formatedGt"
videoBasePath = "../../resources/videos"
txtResBasePath = "../../resources/resultsTxt/results"

trackerLifeInSeconds = 0.5
trackerActiveTimeInSeconds = 1

net = YoloV3Net()

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
                boxes = net.run(frame)

                if boxes is not None:
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
