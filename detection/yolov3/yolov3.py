import tensorflow as tf
import cv2 as cv
import numpy as np

from PIL import Image
from core import utils

classesPath = "../../data/coco.names"
modelPath = "../../checkpoint/yolov3_cpu_nms.pb"

IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names(classesPath)
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), modelPath,
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])


class YoloV3Net:
    def __init__(self):
        self.sess = tf.Session()

    def run(self, img):
        # Processing frame
        img_resized = self._preprocessFrame(img)

        boxes, scores = self.sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

        # Keeping only box labelled "person"
        if boxes is not None:
            boxes = self._getOnlyDetectedPeople(boxes, labels)

        return boxes

    def __del__(self):
        self.sess.close()

    @staticmethod
    def _getOnlyDetectedPeople(boxes, labels):
        pBoxes = []
        for i in np.arange(len(boxes)):
            if labels[i] == 0:
                pBoxes.append(boxes[i])
        return pBoxes

    @staticmethod
    def _preprocessFrame(frame):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frameRGB)
        img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        return img_resized / 255.
