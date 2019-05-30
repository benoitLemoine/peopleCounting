import os
import cv2 as cv
import tensorflow as tf
import numpy as np

from xml.dom import minidom
from core import utils
from PIL import Image

IMAGE_H, IMAGE_W = 416, 416


def imagePreprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = np.array(img.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
    img = img / 255.
    return img


def areEyesInsideRectangle(leftEye, rightEye, r1, r2):
    xmin = min(r1[0], r2[0])
    xmax = max(r1[0], r2[0])
    ymin = min(r1[1], r2[1])
    ymax = max(r1[1], r2[1])

    if xmin > int(leftEye[0]) or xmax < int(leftEye[0]):
        return False
    if ymin > int(leftEye[1]) or ymax < int(leftEye[1]):
        return False

    if xmin > int(rightEye[0]) or xmax < int(rightEye[0]):
        return False
    if ymin > int(rightEye[1]) or ymax < int(rightEye[1]):
        return False

    return True


xmlPath = "/home/benoit/Documents/Stage2A/resources/CP_dataset/groundtruth"
imgPath = "/home/benoit/Documents/Stage2A/resources/CP_dataset/data"

modelPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/checkpoint/yolov3_cpu_nms.pb"
classesPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/data/coco.names"

# Initiate stats
nbrDetected = 0

truePositive = 0
falsePositive = 0


# Load model
classes = utils.read_coco_names(classesPath)
num_classes = len(classes)

input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), modelPath, ["Placeholder:0", "concat_9:0", "mul_6:0"])

with tf.Session() as sess:
    # For each xml file
    xmlFiles = os.listdir(xmlPath)
    for xmlFile in xmlFiles:
        imgFolder = xmlFile[:-4]
        xml = minidom.parse(xmlPath + "/" + xmlFile)

        # For each node
        framesNodes = xml.getElementsByTagName("frame")
        for node in framesNodes:
            frameNbr = node.attributes["number"].value

            # Compute res box
            img = cv.imread(imgPath + "/" + imgFolder + "/" + frameNbr + ".jpg")
            img = imagePreprocess(img)

            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

            # For each person, get eyes pair
            eyes = []
            people = node.getElementsByTagName("person")
            for p in people:

                lx = None
                ly = None
                rx = None
                ry = None

                leftEye = p.getElementsByTagName("leftEye")
                if leftEye:
                    lx = leftEye[0].attributes["x"].value
                    ly = leftEye[0].attributes["y"].value

                rightEye = p.getElementsByTagName("leftEye")
                if rightEye:
                    rx = rightEye[0].attributes["x"].value
                    ry = rightEye[0].attributes["y"].value

                if rightEye and leftEye:
                    eyes.append(((lx, ly), (rx, ry)))

            # For each box we need to find a pair of eyes that fit in the box
            for i, box in enumerate(boxes):
                if classes[labels[i]] == "person":

                    nbrDetected += 1

                    r1 = (box[0], box[1])
                    r2 = (box[2], box[3])

                    if next((eye for eye in eyes if areEyesInsideRectangle(eye[0], eye[1], r1, r2)), None):
                        truePositive += 1
                    else:
                        falsePositive += 1

            if nbrDetected != 0:
                truePositive_rate = truePositive / nbrDetected
                falsePositive_rate = falsePositive / nbrDetected

                # print("TP% = {0:.2f} / FP% = {0:.2f}".format(truePositive_rate, falsePositive_rate))
                print("D = {} / TP = {} / FP = {}".format(nbrDetected, truePositive, falsePositive))