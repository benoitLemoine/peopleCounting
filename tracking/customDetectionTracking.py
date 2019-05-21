import cv2 as cv
import tensorflow as tf
import numpy as np
import time
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

frameCount = 0
peopleCount = 0

trackerLife = 10

with tf.Session() as sess:
    # cap = cv.VideoCapture("/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_18_24_54BackColor.avi")
    # cap = cv.VideoCapture("/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/train/ADL-Rundle-6/img1/ADL-Rundle-6.mp4")
    # cap = cv.VideoCapture("/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P2L_S5_C3.1/P2L_S5_C3.1.mp4")
    cap = cv.VideoCapture("/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P1E_S1_C1/P1E_S1_C1.mp4")
    # cap = cv.VideoCapture(0)
    multiTracker = tr.MultiTracker(trackerLife)

    while True:
        start = time.time()

        frameCount += 1
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

        nbrTrackers = 0
        nbrDetected = 0

        if boxes is not None:
            # Keeping only box labelled "person"
            pBoxes = []
            for i in np.arange(len(boxes)):
                if labels[i] == 0:
                    pBoxes.append(boxes[i])
            boxes = pBoxes

            # Match detection box with trackers
            # fitFunction = lambda tracker, box:  tr.findClosestTracker(tracker, box, 0.5)
            fitFunction = lambda tracker, box: tr.findMaxIoUTracker(tracker, box, 0.5)
            multiTracker.matchDetected(boxes, fitFunction)

            # Updating tracker life
            for tracker in multiTracker.trackers:
                if tracker.paired and tracker.activeTime > 25:
                    if not tracker.counted:
                        peopleCount += 1
                        tracker.counted = True

                    b = tr.resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                    cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), tracker.color, 2)

            multiTracker.resetPaired()

            nbrDetected = len(boxes)
            nbrTrackers = len(multiTracker.trackers)

        cv.imshow("Tracking", frame)

        end = time.time()
        totalTime = round((end - start) * 1000, 3)
        print("[{}] {} trackers / {} detected (update in {} ms)".format(frameCount, nbrTrackers,
                                                                        nbrDetected, totalTime))

        k = cv.waitKey(20)
        if k == ord("q"):
            break

    print("Counted {} people in total".format(peopleCount))
    cv.destroyAllWindows()
    cap.release()
