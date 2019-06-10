import os
import cv2 as cv


folderPath = "/home/benoit/Documents/Stage2A/resources/MIVIA_dataset/Dataset People Counting MIVIA/DBc/VIDEOS/RGB"
resFile = open(folderPath + "/info.txt", "w+")

for f in os.listdir(folderPath):
    if f.endswith(".mkv"):
        videoPath = folderPath + "/" + f

        cap = cv.VideoCapture(videoPath)

        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv.CAP_PROP_FPS)

        resFile.write("{}   {}  {}  {}  {}\n".format(f, width, height, frames, fps))
        cap.release()

resFile.close()
