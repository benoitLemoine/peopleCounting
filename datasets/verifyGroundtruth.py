import os
import cv2 as cv

videoPath = "/home/benoit/Documents/Stage2A/resources/videos/MOT/Venice-2.mp4"
gtFilePath = "/home/benoit/Documents/Stage2A/resources/resultsTxt/formatedGt/MOT/Venice-2.txt"

if not os.path.exists(videoPath) or not os.path.exists(gtFilePath):
    print("These files don't exist :(")
    exit()

cap = cv.VideoCapture(videoPath)
fps = cap.get(cv.CAP_PROP_FPS)

# Storing all gt timestamps
gtFile = open(gtFilePath)
gtTimestamps = []
for count, line in enumerate(gtFile):
    if count != 0:
        gtTimestamps.append(int(int(line) * fps / 1000))
gtFile.close()

h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

count = 0
k = None
while True:
    ret, img = cap.read()

    if not ret or k == ord("q"):
        break

    if count in gtTimestamps:
        cv.rectangle(img, (0, 0), (w, h), (0, 0, 185), 30)

    cv.imshow("Verify", img)
    k = cv.waitKey(60)
    count += 1

cap.release()
cv.destroyAllWindows()