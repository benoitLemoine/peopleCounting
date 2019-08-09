import os
import cv2 as cv

root = "../../../resources/MOT_dataset/2DMOT2015"

resFile = open(root + "/info.txt", "w+")
for folderPath in os.listdir(root):

    if os.path.isdir(root+"/"+folderPath):
        folders = os.listdir(root + "/" + folderPath)
        for f in folders:
            videoName = f.split("/")[-1] + ".mp4"
            videoPath = root + "/" + folderPath + "/" + f + "/img1/" + videoName

            cap = cv.VideoCapture(videoPath)
            width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv.CAP_PROP_FPS)

            resFile.write("{}   {}  {}  {}  {}\n".format(videoName, width, height, frames, fps))
            cap.release()

        resFile.write("\n")

resFile.close()
