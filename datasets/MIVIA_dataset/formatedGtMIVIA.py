import os
from xml.dom import minidom

from tracking.resultsExporter import ResultsExporter

gtPaths = "/home/benoit/Documents/Stage2A/resources/MIVIA_dataset/Dataset People Counting MIVIA/DBc/GROUND TRUTH"
videoPaths = "/home/benoit/Documents/Stage2A/resources/MIVIA_dataset/Dataset People Counting MIVIA/DBc/VIDEOS/RGB"

formatedGtFolderPath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt/MIVIA"

for file in os.listdir(gtPaths):
    if file.endswith(".xgtf"):
        videoName = file[:-5]
        videoPath = videoPaths + "/" + videoName + ".mkv"

        if os.path.isfile(videoPath):
            resultsPath = formatedGtFolderPath + "/" + videoName + ".txt"
            exporter = ResultsExporter(resultsPath, videoPath)

            xml = minidom.parse(gtPaths + "/" + file)
            for obj in xml.getElementsByTagName("object"):
                frameIndex = obj.attributes["framespan"].value
                frameIndex = frameIndex.split(":")[0]
                exporter.write(int(frameIndex))

            exporter.close()
