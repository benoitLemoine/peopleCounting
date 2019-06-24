import os
import cv2 as cv

from tracking.resultsExporter import ResultsExporter

videoBasePath = "/home/benoit/Documents/Stage2A/resources/videos/PCDS"
resFileBasePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt/PCDS"

for file in os.listdir(videoBasePath):

    fileName = file[:-4]

    videoPath = videoBasePath + "/" + file
    resFilePath = resFileBasePath + "/" + fileName + ".txt"

    if not os.path.exists(resFilePath):
        cap = cv.VideoCapture(videoPath)
        exporter = ResultsExporter(resFilePath, videoPath)

        print("-- {} --".format(file))

        count = 0
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
                print("\tLabeled !")
            elif k == ord("n"):
                break
            elif k == ord("q"):
                exit()

            cap.set(1, count)

        cap.release()
        exporter.close()
