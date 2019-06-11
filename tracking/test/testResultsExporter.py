from tracking.resultsExporter import ResultsExporter

resFilePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/test.txt"
videoPath = "/home/benoit/Documents/Stage2A/resources/CP_dataset/data/P1E_S1_C1/P1E_S1_C1.mp4"

exporter = ResultsExporter(resFilePath, videoPath)

exporter.write(256)
exporter.write(332)
exporter.close()
