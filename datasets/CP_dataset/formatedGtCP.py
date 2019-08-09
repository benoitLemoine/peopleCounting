import os
from xml.dom import minidom

from tracking.resultsExporter import ResultsExporter

gtPaths = "../../../resources/CP_dataset/groundtruth"
videoPaths = "../../../resources/CP_dataset/data"

formatedGtFolderPath = "../../../resources/resultsTxt/formatedGt/CP"

for file in os.listdir(gtPaths):
    if file.endswith(".xml"):
        videoName = file[:-4]
        videoPath = videoPaths + "/" + videoName + "/" + videoName + ".mp4"

        if os.path.isfile(videoPath):
            resultsPath = formatedGtFolderPath + "/" + videoName + ".txt"
            exporter = ResultsExporter(resultsPath, videoPath)

            xml = minidom.parse(gtPaths + "/" + file)
            countedIDs = []
            frameCount = 0
            for frame in xml.getElementsByTagName("frame"):
                frameCount += 1
                for person in frame.getElementsByTagName("person"):
                    id = person.attributes["id"].value

                    if id not in countedIDs:
                        countedIDs.append(id)
                        exporter.write(frameCount)

            exporter.close()
