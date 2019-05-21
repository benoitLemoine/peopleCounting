import cv2 as cv
import tensorflow as tf
import numpy as np

from xml.dom import minidom
from core import utils
from PIL import Image

IMAGE_H, IMAGE_W = 416, 416


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


file = "P1E_S1_C3.xml"

xmlFile = "/home/benoit/Documents/Stage2A/resources/CP_dataset/groundtruth/" + file
imgFolder = "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/" + file[:-4]

modelPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/checkpoint/yolov3_cpu_nms.pb"
classesPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/data/coco.names"

# Load model
classes = utils.read_coco_names(classesPath)
num_classes = len(classes)

input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), modelPath,
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

with tf.Session() as sess:
    xml = minidom.parse(xmlFile)

    # For each node
    framesNodes = xml.getElementsByTagName("frame")
    for node in framesNodes:
        frameNbr = node.attributes["number"].value

        # Compute res box
        image_path = imgFolder + "/" + frameNbr + ".jpg"

        frame = cv.imread(image_path)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        img_resized = img_resized / 255.
        # prev_time = time.time()

        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
        image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)

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
                cv.circle(image, (lx, ly), 3, (175, 0, 0), -1)
                cv.circle(image, (rx, ry), 3, (175, 0, 0), -1)

        cv.imshow("Preview", image)
        cv.waitKey(0)

cv.destroyAllWindows()