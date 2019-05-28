import cv2 as cv
import tensorflow as tf
import numpy as np
import os
from PIL import Image

import tracking.tracker as tr
from core import utils

classesPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/data/coco.names"
modelPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/checkpoint/yolov3_cpu_nms.pb"

IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names(classesPath)
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), modelPath,
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

basePath = "/home/benoit/Documents/Stage2A/resources/hey/25_20160407_back/normal/crowd"
labelPath = basePath + "/label.txt"

file = open(labelPath, "r")
lines = file.readlines()

trackerLife = 5
activeTime = 20

# Parsing groundtruth file
data = []
for line in lines[4:]:
    splitLine = line.split()
    fileName = splitLine[0][:-9] + "Color.avi"
    numberOfPeople = int(splitLine[1]) + int(splitLine[2])

    if os.path.isfile(basePath + "/" + fileName):
        data.append((fileName, numberOfPeople))


# Counting people on each video
resFile = open(basePath + "/results.txt", "w+")
resFile.write("Active time: {}\n".format(activeTime))
resFile.write("Tracker life: {}\n".format(trackerLife))
with tf.Session() as sess:
    for d in data:
        print("Processing video {}".format(d[0]))
        counted = 0
        cap = cv.VideoCapture(basePath+"/"+d[0])
        multiTracker = tr.MultiTracker(trackerLife)

        while cap.isOpened():

            res, frame = cap.read()
            if not res:
                break

            # Resizing frame
            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = Image.fromarray(frameRGB)
            img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
            img_resized = img_resized / 255.

            # Detecting
            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

            if boxes is not None:
                # Keeping only box labelled "person"
                pBoxes = []
                for i in np.arange(len(boxes)):
                    if labels[i] == 0:
                        pBoxes.append(boxes[i])
                boxes = pBoxes

                # Match detection box with trackers
                fitFunction = lambda tracker, box: tr.findMaxIoUTracker(tracker, box, 0.5)
                multiTracker.matchDetected(boxes, fitFunction)

                # Updating tracker life
                for tracker in multiTracker.trackers:
                    if tracker.paired and tracker.activeTime > activeTime:
                        b = tr.resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])

                        if not tracker.counted:
                            counted += 1
                            tracker.counted = True

                multiTracker.resetPaired()

        resFile.write("{} {} {}\n".format(d[0], d[1], counted))
resFile.close()
