import cv2 as cv


class ResultsExporter:
    def __init__(self, resultsFilePath, videoPath):
        self.file = open(resultsFilePath, "w+")

        cap = cv.VideoCapture(videoPath)
        videoName = videoPath.split("/")[-1]
        framesCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        cap.release()

        self.file.write("# {}, {} frames, {} fps\n".format(videoName, framesCount, fps))

    def write(self, frameIndex):
        self.file.write("{}\n".format(int(frameIndex)))

    def close(self):
        self.file.close()
