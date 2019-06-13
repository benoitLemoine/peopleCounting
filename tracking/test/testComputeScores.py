from tracking.resultsExporter import computeResultsScores, computeDeltaTimeWithOffset

gtFilePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt/CP/P1E_S1_C1.txt"
resFilePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/results/CP/P1E_S1_C1.txt"


deltaTimeFunction = lambda resTime, gtTime: computeDeltaTimeWithOffset(resTime, gtTime, 1000)
computeResultsScores(gtFilePath, resFilePath, 1500, deltaTimeFunction)
