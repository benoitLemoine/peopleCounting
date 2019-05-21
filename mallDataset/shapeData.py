import cv2 as cv
import scipy.io
import numpy as np
import os
import math as m
import random
from sklearn.cluster import KMeans


def getPoint(imageIndex, gtIndex=-1):
    if gtIndex == -1:
        return mat[imageIndex][0][0][0][:]
    else:
        return mat[imageIndex][0][0][0][gtIndex]


def getImageName(index):
    nbrOfZero = 6 - len(str(index + 1))
    name = imgPath + "/" + baseImgName
    for i in np.arange(nbrOfZero):
        name += "0"
    name += str(index + 1) + ".jpg"
    return name


def getImage(i):
    name = getImageName(i)
    return cv.imread(name)


def getImageDimensions(i=0):
    img = getImage(i)
    return img.shape[:2]


def getRectangleGroundTruth(i, size=21):
    imgHeight, imgWidth = getImageDimensions(i)

    gt = []
    for x, y in getPoint(i):
        p1 = (int(max(0, m.floor(x) - size / 2)), int(max(0, m.floor(y) - size / 2)))
        p2 = (int(min(imgHeight, m.floor(x) + size / 2)), int(min(imgWidth, m.floor(y) + size / 2)))
        gt.append((p1, p2))
    return gt


def getImageWithRectangleGroundTruth(i, line_width=2):
    img = getImage(i)
    gt = getRectangleGroundTruth(i)

    for p1, p2 in gt:
        cv.rectangle(img, p1, p2, (175, 0, 0), line_width)
    return img


def writeImageGroundTruthInFile(file, name, gt):
    file.write(name)
    for p1, p2 in gt:
        file.write(" {},{},{},{},0".format(p1[0], p1[1], p2[0], p2[1]))
    file.write("\n")


matPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/resources/mall_dataset/mall_gt.mat"
imgPath = "/home/benoit/Documents/Stage2A/tensorflow-yolov3/resources/mall_dataset/frames"
baseImgName = "seq_"

trainFileName = "mall_train.txt"
testFileName = "mall_test.txt"
anchorFileName = "mall_anchors.txt"

relativeSizeTrainSet = 0.9
nbrClusters = 9

# Load ground truth
mat = scipy.io.loadmat(matPath)
mat = mat["frame"]
mat = np.squeeze(mat, axis=0)

# Load images
images = os.listdir(imgPath)
np.random.shuffle(images)

# Computing ground truth
trainFile = open(trainFileName, "w+")
testFile = open(testFileName, "w+")

testSetGoalSize = int((1 - relativeSizeTrainSet) * len(images))
testSetCurrentSize = 0
for i in np.arange(len(images)):
    name = getImageName(i)
    gt = getRectangleGroundTruth(i)

    if random.randrange(0, 1) < relativeSizeTrainSet and testSetCurrentSize <= testSetGoalSize:
        writeImageGroundTruthInFile(testFile, name, gt)
        testSetCurrentSize += 1
    else:
        writeImageGroundTruthInFile(trainFile, name, gt)

trainFile.close()
testFile.close()

# Computing anchors
points = []
for i in np.arange(len(images)):
    for p in getPoint(i):
        points.append([p[0], p[1]])

kmeans = KMeans(n_clusters=nbrClusters, init="k-means++", random_state=42)
kmeans.fit(points)

h, w = getImageDimensions()
anchors = []
for x, y in kmeans.cluster_centers_:
    anchors.append((x / w, y / h))

anchorFile = open(anchorFileName, "w+")
for i in np.arange(len(anchors)):
    if i != 0:
        anchorFile.write(" ")
    anchorFile.write("{},{}".format(anchors[i][0], anchors[i][1]))
    if i != nbrClusters - 1:
        anchorFile.write(",")

anchorFile.close()

# Print anchors
img = getImage(0)
for x, y in kmeans.cluster_centers_:
    cv.circle(img, (int(x), int(y)), 5, (175, 0, 0), -1)

cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()
