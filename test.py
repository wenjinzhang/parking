import os
from PIL import Image
import numpy as np
import dataset
import cv2
data_generate = dataset.get_data_gen()

runtime_data_dir = "runtime"
file_names = os.listdir(runtime_data_dir)
file_names.sort()

print(len(file_names))


def get_image_array(filename):
    img = Image.open(filename)
    img = img.resize((40, 40))
    return np.array(img)


images = [get_image_array("{}/{}".format(runtime_data_dir, name)) for name in file_names]

print(np.shape(images))

print(np.rank(images))



