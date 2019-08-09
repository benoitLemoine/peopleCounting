from tracking.resultsExporter import computeResultsScores, computeDeltaTimeWithOffset

gtFilePath = "../../../resources/resultsTxt/formatedGt/CP/P1E_S1_C1.txt"
resFilePath = "../../../resources/resultsTxt/results/CP/P1E_S1_C1.txt"


deltaTimeFunction = lambda resTime, gtTime: computeDeltaTimeWithOffset(resTime, gtTime, 1000)
p, r, f1 = computeResultsScores(gtFilePath, resFilePath, 1500, deltaTimeFunction)

print(p)
print(r)
print(f1)