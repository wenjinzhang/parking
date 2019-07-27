import cv2
import time
import os
from checkSpaces import checkSpaces


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
        break
        time.sleep(2)


def simulator():
    video_fold = "../testvideo/"
    videos = os.listdir(video_fold)
    for video_name in videos:
        print(os.path.join(video_fold, video_name))
        play_video(os.path.join(video_fold, video_name))



if __name__ == '__main__':
    simulator()
