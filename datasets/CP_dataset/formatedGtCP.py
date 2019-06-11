import os
from xml.dom import minidom

from tracking.resultsExporter import ResultsExporter

gtPaths = "/home/benoit/Documents/Stage2A/resources/CP_dataset/groundtruth"
videoPaths = "/home/benoit/Documents/Stage2A/resources/CP_dataset/data"

formatedGtFolderPath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt/CP"

for file in os.listdir(gtPaths):
    if file.endswith(".xml"):
        videoName = file[:-4]
        videoPath = videoPaths + "/" + videoName + "/" + videoName + ".mp4"

        if os.path.isfile(videoPath):
            resultsPath = formatedGtFolderPath + "/" + videoName + ".txt"
            exporter = ResultsExporter(resultsPath, videoPath)

            xml = minidom.parse(gtPaths + "/" + file)
            countedIDs = []
            for frame in xml.getElementsByTagName("frame"):

                for person in frame.getElementsByTagName("person"):
                    id = person.attributes["id"].value

                    if id not in countedIDs:
                        countedIDs.append(id)
                        exporter.write(int(frame.attributes["number"].value))

            exporter.close()
