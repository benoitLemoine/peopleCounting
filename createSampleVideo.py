import cv2 as cv
import numpy as np
import tensorflow as tf

from PIL import Image
from core import utils


def resizeBox(box, baseSize, targetSize):
    yDist = targetSize[0] / baseSize[0]
    xDist = targetSize[1] / baseSize[1]

    p1x = box[0] * xDist
    p1y = box[1] * yDist

    p2x = box[2] * xDist
    p2y = box[3] * yDist

    return int(p1x), int(p1y), int(p2x), int(p2y)


classesPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/data/coco.names"
modelPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/checkpoint/yolov3_cpu_nms.pb"
destPath = "/home/benoit/Documents/Stage2A/resources/results/detection"

IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names(classesPath)
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), modelPath,
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

videosPaths = {
    "CP": "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P2L_S5_C3.1/P2L_S5_C3.1.mp4",
    # "PCDS": "/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_18_24_54BackColor.avi",
    # "MIVIA": "/home/benoit/Documents/Stage2A/resources/MIVIA_dataset/Dataset People Counting MIVIA/DBc/VIDEOS/RGB/C_I_S_1.mkv",
    # "MOT": "/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/train/PETS09-S2L1/img1/PETS09-S2L1.mp4"
}

with tf.Session() as sess:
    for name, videoPath in videosPaths.items():
        video = cv.VideoCapture(videoPath)
        name = destPath + "/" + name + ".avi"

        first = True
        while True:
            ret, frame = video.read()

            if first:
                print("Processing {}".format(name))
                h, w, l = frame.shape
                size = w, h
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                writer = cv.VideoWriter(name, fourcc, 25, size)
                first = False

            if not ret:
                video.release()
                writer.release()
                break

            # Resize input frame
            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = Image.fromarray(frameRGB)
            img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
            img_resized = img_resized / 255.

            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)

            pBoxes = []
            for i in np.arange(len(boxes)):
                if labels[i] == 0:
                    pBoxes.append(boxes[i])
            boxes = pBoxes

            if boxes is not None:
                for b in boxes:
                    b = resizeBox(b, (IMAGE_H, IMAGE_W), frame.shape[0:2])
                    cv.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (175, 0, 0), 2)

            writer.write(frame)
