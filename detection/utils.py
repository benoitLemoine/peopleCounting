import cv2 as cv
import tensorflow as tf
import numpy as np
from PIL import Image

from datasets.PCDS_dataset.utils import removeCameraWatermark
from core import utils

classesPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/data/coco.names"
modelPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/checkpoint/yolov3_cpu_nms.pb"

IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names(classesPath)
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), modelPath,
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])


def preprocessFrame(frame):
    # frame = removeCameraWatermark(frame)
    # frame = cv.convertScaleAbs(frame, alpha=5, beta=50)
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frameRGB)
    img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
    return img_resized / 255.


def getOnlyDetectedPeople(boxes, labels):
    pBoxes = []
    for i in np.arange(len(boxes)):
        if labels[i] == 0:
            pBoxes.append(boxes[i])
    return pBoxes
