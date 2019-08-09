import os
from tracking.resultsExporter import computeResultsScores, computeDeltaTimeWithOffset

gtFolderPath = "../../resources/resultsTxt/formatedGt"
resFolderPath = "../../resources/resultsTxt/results"

deltaTimeFunction = lambda resTime, gtTime: computeDeltaTimeWithOffset(resTime, gtTime, 1000)
maxSpan = 1500

sumPrecision = 0
sumRecall = 0
sumF1 = 0
count = 0
averagePrecision = 0
averageRecall = 0
averageF1 = 0
for folder in os.listdir(gtFolderPath):
    for file in os.listdir(gtFolderPath + "/" + folder):

        gtFilePath = gtFolderPath + "/" + folder + "/" + file
        resFilePath = resFolderPath + "/" + folder + "/" + file

        if os.path.exists(resFilePath):
            count += 1

            precision, recall, f1 = computeResultsScores(gtFilePath, resFilePath, maxSpan, deltaTimeFunction)

            print("{} ({} / {} / {})".format(file.split(".")[0], round(precision, 3), round(recall, 3), round(f1, 3)))

            sumPrecision += precision
            sumRecall += recall
            sumF1 += f1

if count != 0:
    averagePrecision = sumPrecision / count
    averageRecall = sumRecall / count
    averageF1 = sumF1 / count

print("\nAverage precision : {}".format(averagePrecision))
print("Average recall : {}".format(averageRecall))
print("Average F1 : {}".format(averageF1))
