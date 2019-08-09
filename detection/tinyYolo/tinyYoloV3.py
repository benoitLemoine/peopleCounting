##
##  The TinyYoloV3Net is currently not working due to incompatibilities
##  between classes from tensorflow.keras and keras
##
##  Please consider only using YoloV3Net for now
##

import tensorflow as tf
import numpy as np

from detection.tinyYolo.src.yolov3_tf2.dataset import transform_images
from detection.tinyYolo.src.yolov3_tf2.models import YoloV3Tiny

IMAGE_H, IMAGE_W = 416, 416

weightsPath = "./src/checkpoints/yolov3-tiny.tf"
classesPath = "./src/data/coco.names"

classNames = [c.strip() for c in open(classesPath).readlines()]


class TinyYoloV3Net:
    def __init__(self):
        self.net = YoloV3Tiny()
        self.net.load_weights(weightsPath)

    def run(self, img):
        # Processing frame
        img_resized = self._preprocessFrame(img)

        boxes, scores, labels, nums = self.net(img_resized)

        print(boxes)
        print(scores)
        print(labels)
        print(nums)

        if boxes is not None:
            boxes = self._getOnlyDetectedPeople(boxes, labels, nums)

        return boxes

    def __del__(self):
        pass

    @staticmethod
    def _getOnlyDetectedPeople(boxes, labels, nums):
        pBoxes = []
        for i in range(tf.size(nums)):
            if labels[i] == 0:
                pBoxes.append(boxes[i])
        return pBoxes

    @staticmethod
    def _preprocessFrame(frame):
        img = tf.expand_dims(frame, 0)
        img = transform_images(img, 416)
        print(img.shape)
        return img
