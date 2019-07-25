import model
import dataset
import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def get_image_array(filename):
    img = image.load_img(
        filename,
        grayscale=False,
        color_mode='rgb',
        interpolation='nearest',
        target_size=(48, 48, 3))

    img = image.img_to_array(img)

    return img / 255.0

batch_size = 32
epochs = 15
num_classes = 2

# build model
model_final = model.model((48, 48, 3), num_classes)

model_final.load_weights("car1.h5", by_name=True)

data_generate = dataset.get_data_gen()

image_shape = (48, 48)

runtime_data_dir = "runtime"

runtime_data_dir = "train_data/test/empty"

file_names = os.listdir(runtime_data_dir)
file_names.sort()

images = [get_image_array("{}/{}".format(runtime_data_dir, name)) for name in file_names]
print(len(images))
images = np.array(images)
images.reshape(-1, 48, 48, 1)
result = model_final.predict(images)

print(np.argmax(result, axis=1))


img = image.load_img(
    "train_data/test/empty/spot89.jpg",
    grayscale=False,
    color_mode='rgb',
    interpolation='nearest',
    target_size=(48, 48, 3))

x = image.img_to_array(img)

x = x / 255.0
x = np.expand_dims(x, axis=0)
result = model_final.predict(x)
print(result)
print(np.argmax(result, axis=1))
