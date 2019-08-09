import os
import cv2 as cv

folderPaths = [
    "../../../resources/PCDS_dataset/25_20160407_back/noisy/uncrowd",
    "../../../resources/PCDS_dataset/25_20160407_back/noisy/crowd",
    "../../../resources/PCDS_dataset/25_20160407_back/normal/crowd",
    "../../../resources/PCDS_dataset/25_20160407_back/normal/uncrowd"
]


for folderPath in folderPaths:
    folders = os.listdir(folderPath)
    resFile = open(folderPath + "/info.txt", "w+")

    for f in folders:
        videoName = f.split("/")[-1]
        videoPath = folderPath + "/" + f

        if videoName.endswith("Color.avi"):
            print(videoName)
            print(videoPath)
            cap = cv.VideoCapture(videoPath)
            width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv.CAP_PROP_FPS)

            resFile.write("{}   {}  {}  {}  {}\n".format(videoName, width, height, frames, fps))
            cap.release()

    resFile.close()
