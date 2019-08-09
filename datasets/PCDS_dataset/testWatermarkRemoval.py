import cv2 as cv

from datasets.PCDS_dataset.utils import removeCameraWatermark

# cap = cv.VideoCapture("/home/benoit/Documents/Stage2A/resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_19_43_00BackColor.avi")
cap = cv.VideoCapture("../../../resources/PCDS_dataset/25_20160407_back/normal/crowd/2016_04_07_18_24_54BackColor.avi")

i = 0
while cap.isOpened():

    ret, frame = cap.read()

    k = cv.waitKey(30)
    if k == ord("q") or not ret:
        break

    frame = removeCameraWatermark(frame)
    color = frame[87][190]
    print("[{}] : {}".format(i, color))

    i += 1

    frame = cv.resize(frame, (640, 480))
    cv.imshow("Res", frame)

cv.destroyAllWindows()
cap.release()
