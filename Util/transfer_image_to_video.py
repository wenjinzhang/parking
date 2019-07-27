import cv2
import os


# im_dir = '/home/suanfa/data/out/201708231503440'
# video out put path
# video_dir = '/home/suanfa/data/out/201708231503440-1018.avi'
def transform(image_dir="../testImages/Cloudy/",
              video_dir="../testvideo/Cloudy.avi",
              fps=5, img_size=(823, 334)):

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    file_names = os.listdir(image_dir)
    file_names.sort()
    for file in file_names:
        image_name = os.path.join(image_dir, file)
        frame = cv2.imread(image_name)
        videoWriter.write(frame)
        #
    videoWriter.release()


def main():
    transform("../testImages/Rainy/", "../testvideo/Rainy.avi")


if __name__ == '__main__':
    main()



