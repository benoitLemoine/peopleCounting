from tracking.resultsExporter import ResultsExporter

resFilePath = "../../../resources/resultsTxt/test.txt"
videoPath = "../../../resources/CP_dataset/data/P1E_S1_C1/P1E_S1_C1.mp4"

exporter = ResultsExporter(resFilePath, videoPath)

exporter.write(256)
exporter.write(332)
exporter.close()
