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

ioUFloor = 0.3
trackerType = "KCF"
trackerLife = 15

with tf.Session() as sess:
    cap = cv.VideoCapture("/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P2L_S5_C3.1/P2L_S5_C3.1.mp4")
    # cap = cv.VideoCapture(0)
    multiTracker = tr.MultiTracker()

    while True:
        start = time.time()

        frameCount += 1
        res, frame = cap.read()

        # Resizing frame
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frameRGB)
        img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        img_resized = img_resized / 255.

        # Detecting
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

        # Keeping only box labelled "person"
        pBoxes = []
        for i in np.arange(len(boxes)):
            if labels[i] == 0:
                pBoxes.append(boxes[i])
        boxes = pBoxes

        # Pairing detected with tracked
        multiTracker.resetPaired()
        multiTracker.update(frame)

        if boxes is not None:
            for b in boxes:
                b = (b[0], b[1]), (b[2], b[3])
                bestTracker, iouValue = multiTracker.findMaxIoUTracker(b)

                # Find best tracker
                if iouValue >= ioUFloor:
                    bestTracker.paired = True
                    bestTracker.trackBox = tr.rectBoxToTrackBox(b)
                else:
                    newTracker = tr.Tracker(trackerType, trackerLife)
                    multiTracker.add(newTracker, frame, tr.rectBoxToTrackBox(b))

        # Updating tracker life
        for tracker in multiTracker.trackers:
            if tracker.paired:
                tracker.life = trackerLife
                b = tr.resizeTrackBox(tracker.trackBox, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), tracker.color, 2)
            else:
                tracker.life -= 1
                if tracker.life == 0:
                    multiTracker.trackers.remove(tracker)

        cv.imshow("Tracking", frame)

        end = time.time()
        totalTime = round((end - start) * 1000, 3)
        # print("[{}] {} trackers alive  (update in {} ms)".format(frameCount, len(multiTracker.trackers), totalTime))

        k = cv.waitKey(20)
        if k == ord("q"):
            break

    cv.destroyAllWindows()
    cap.release()
