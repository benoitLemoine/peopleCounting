import os
from tracking.resultsExporter import computeResultsScores, computeDeltaTimeWithOffset

gtFolderPath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt"
resFolderPath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/results"

deltaTimeFunction = lambda resTime, gtTime: computeDeltaTimeWithOffset(resTime, gtTime, 1000)
maxSpan = 1500

sumTpr = 0
sumFpr = 0
count = 0
averageTpr = 0
averageFpr = 0
for folder in os.listdir(gtFolderPath):
    for file in os.listdir(gtFolderPath + "/" + folder):

        gtFilePath = gtFolderPath + "/" + folder + "/" + file
        resFilePath = resFolderPath + "/" + folder + "/" + file

        if os.path.exists(resFilePath):
            count += 1

            tpr, fpr = computeResultsScores(gtFilePath, resFilePath, maxSpan, deltaTimeFunction)

            if tpr < 0.66:
                print("{} ({}  /  {})".format(file, round(tpr, 2), round(fpr, 2)))

            sumTpr += tpr
            sumFpr += fpr

if count != 0:
    averageTpr = sumTpr / count
    averageFpr = sumFpr / count

print("\nAverage tpr : {}".format(averageTpr))
print("Average fpr : {}".format(averageFpr))
