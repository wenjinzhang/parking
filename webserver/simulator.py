import cv2
import time
import os
from checkSpaces import checkSpaces
from application import prediction
import numpy as np
import cv2

def show_detect(array_path_prefix=".", prediction_result=[], orginal_img_path= "./static/play.jpg"):
    vertical = np.load("{}/vertical.npy".format(array_path_prefix), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    horizontal = np.load("{}/horizontal.npy".format(array_path_prefix), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    alpha = 0.7
    img = cv2.imread(orginal_img_path)
    height, width, channels = img.shape
    overlay = np.copy(img)
    counter = 0
    print(vertical)
    print(horizontal)
    count = 0
    for x in range(0, len(horizontal) - 1, 2):
        for y in range(len(vertical) - 1):
            if prediction_result[count]:
                cv2.rectangle(img, (vertical[y], horizontal[x]), (vertical[y + 1], horizontal[x + 1]), (0, 0, 255), -1)
            else:
                cv2.rectangle(img, (vertical[y], horizontal[x]), (vertical[y + 1], horizontal[x + 1]), (0, 255, 0), -1)
            count += 1

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imwrite("./static/detect.jpg", img)

def play_video(video_name):
    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", fps, "frames=", frames)

    for i in range(int(frames)):
        ret, frame = video_capture.read()
        cv2.imwrite("static/play.jpg", frame)
        # process the prediction
        # split the images
        checkSpaces("./static/play.jpg")
        result = prediction("./segment")
        show_detect("../setupData", result)
        time.sleep(2)

def simulator():
    video_fold = "../testvideo/"
    videos = os.listdir(video_fold)
    for video_name in videos:
        print(os.path.join(video_fold, video_name))
        play_video(os.path.join(video_fold, video_name))



if __name__ == '__main__':
    simulator()
