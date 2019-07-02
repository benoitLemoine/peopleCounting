import os
import cv2 as cv
from tracking.resultsExporter import ResultsExporter

videoBasePath = "/home/benoit/Documents/Stage2A/resources/videos"
resBasePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt"

files = [
    ("CP", "P2L_S3_C2.1"),
    ("CP", "P2L_S2_C1.2"),
    ("CP", "P2E_S2_C2.2"),
    ("CP", "P2L_S2_C3.1"),
    ("CP", "P2E_S3_C1.1"),
    ("CP", "P2L_S3_C3.1"),
    ("CP", "P2L_S2_C1.1"),
    ("CP", "P2E_S3_C2.2"),
    ("CP", "P2L_S1_C3.1"),
    ("CP", "P2E_S1_C2.2"),
    ("CP", "P2L_S4_C3.1"),
    ("CP", "P2L_S1_C3.2"),
    ("CP", "P1E_S4_C2"),
    ("CP", "P2L_S4_C2.2"),
    ("CP", "P2L_S2_C2.1"),
    ("CP", "P2E_S4_C2.1"),
    ("CP", "P2L_S2_C3.2"),
    ("CP", "P2E_S4_C3.1"),
    ("CP", "P2L_S3_C1.1"),
    ("CP", "P1L_S4_C1"),
    ("CP", "P2L_S4_C2.1"),
    ("CP", "P2E_S2_C1.2"),
    ("CP", "P2L_S1_C1.2"),
    ("CP", "P2L_S4_C3.2")
]

for file in files:
    datasetName = file[0]
    fileName = file[1]

    # Check if video folder exists
    videoFolder = videoBasePath + "/" + datasetName
    if not os.path.isdir(videoFolder):
        print("{} is not a directory".format(videoFolder))
        exit()

    # Look for corresponding video
    videoPath = None
    for video in os.listdir(videoFolder):
        if video.startswith(fileName):
            videoPath = videoFolder + "/" + video
            resFilePath = resBasePath + "/" + datasetName + "/" + fileName + ".txt"
            break

    if not videoPath:
        print("{} video doesn't exist".format(fileName))
        exit()

    cap = cv.VideoCapture(videoPath)
    exporter = ResultsExporter(resFilePath, videoPath)

    print("-- {} --".format(file))

    count = 0
    nbrDetected = 0
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    k = None
    while True:

        ret, img = cap.read()
        if not ret:
            break
        cv.imshow("img", img)

        k = cv.waitKeyEx(0)
        if k == 65363:
            count = min(totalFrames - 1, count + 1)
        elif k == 65362:
            count = min(totalFrames - 1, count + 10)
        elif k == 65361:
            count = max(0, count - 1)
        elif k == 65364:
            count = max(0, count - 10)
        elif k == 32:
            exporter.write(count)
            nbrDetected += 1
            print("\tLabeled ! ({})".format(nbrDetected))
        elif k == ord("n"):
            break
        elif k == ord("q"):
            exit()

        if count == totalFrames - 1:
            print("=> End of the video")

        cap.set(1, count)

    cap.release()
    exporter.close()
