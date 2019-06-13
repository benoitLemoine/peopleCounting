import cv2 as cv


class ResultsExporter:
    def __init__(self, resultsFilePath, videoPath):
        self.file = open(resultsFilePath, "w+")

        cap = cv.VideoCapture(videoPath)
        videoName = videoPath.split("/")[-1]
        framesCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        cap.release()

        self.file.write("# {}, {} frames, {} fps\n".format(videoName, framesCount, self.fps))

    def write(self, frameIndex):
        self.file.write("{}\n".format(int(1000 * int(frameIndex) / self.fps)))

    def close(self):
        self.file.close()


# Delta time in ms
def computeResultsScores(gtFilePath, resultsFilePath, maxSpan, deltaTimeFunction):
    # Storing all gt timestamps
    gtFile = open(gtFilePath)
    gtTimestamps = []
    for count, line in enumerate(gtFile):
        if count != 0:
            gtTimestamps.append(int(line))
    gtFile.close()
    gtCount = len(gtTimestamps)

    # Storing all results timestamps
    resFile = open(resultsFilePath)
    resTimestamps = []
    for count, line in enumerate(resFile):
        if count != 0:
            resTimestamps.append(int(line))
    gtFile.close()
    resCount = len(resTimestamps)

    map = []

    # Find best match
    truePositive = []
    falsePositive = []
    for resTime in resTimestamps:
        minDelta = deltaTimeFunction(resTime, gtTimestamps[0])
        bestMatch = gtTimestamps[0]

        for gtTime in gtTimestamps[1:] or []:
            delta = deltaTimeFunction(resTime, gtTime)

            if delta < minDelta:
                minDelta = delta
                bestMatch = gtTime

        if minDelta < maxSpan:
            gtTimestamps.remove(bestMatch)
            truePositive.append(resTime)
            map.append((bestMatch, resTime))
        else:
            falsePositive.append(resTime)

    for unfoundGt in gtTimestamps:
        map.append((unfoundGt, None))

    map.sort()

    print("|GT|  |RESULT|")
    for gt, res in map:
        print("{} -> {}".format(gt, res))

    print("\n{} people in the groundtruth".format(gtCount))
    print("{} people counted\n".format(resCount))
    print("{} true positives".format(len(truePositive)))
    print("{} false positives: {}".format(len(falsePositive), falsePositive))


def computeDeltaTimeWithOffset(resTime, gtTime, offset):
    return abs(offset + gtTime - resTime)
