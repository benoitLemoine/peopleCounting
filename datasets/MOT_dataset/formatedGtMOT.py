import os

from tracking.resultsExporter import ResultsExporter

basePaths = [
    "/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/train",
    "/home/benoit/Documents/Stage2A/resources/MOT_dataset/2DMOT2015/test"
]
formatedGtFolderPath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt/MOT"

for bp in basePaths:
    for file in os.listdir(bp):
        gtPath = bp + "/" + file + "/gt/gt.txt"
        videoPath = bp + "/" + file + "/img1/" + file + ".mp4"

        if os.path.isfile(gtPath) and os.path.isfile(videoPath):
            resultsPath = formatedGtFolderPath + "/" + file + ".txt"
            exporter = ResultsExporter(resultsPath, videoPath)

            gtFile = open(gtPath, "r")
            countedIDs = []
            for _, line in enumerate(gtFile):
                line = line.split(",")
                frameIndex = line[0]
                id = line[1]

                if id not in countedIDs:
                    countedIDs.append(id)
                    exporter.write(frameIndex)

            exporter.close()
