import os
from keras.preprocessing.image import ImageDataGenerator


def get_data_gen():
    return ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=5)


def get_data(batch_size=32, image_shape=(48, 48), train_data_dir="./PKLotSegmented/PUC/Sunny/train", validation_data_dir="./PKLotSegmented/PUC/Sunny/test"):
    files_train = 136959
    files_validation = 71428

    # for sub_folder in os.listdir(train_data_dir):
    #     path, dirs, files = next(os.walk(os.path.join(train_data_dir, sub_folder)))
    #     files_train += len(files)

    # for sub_folder in os.listdir(validation_data_dir):
    #     path, dirs, files = next(os.walk(os.path.join(validation_data_dir, sub_folder)))
    #     files_validation += len(files)

    print(files_train, files_validation)

    # Initiate the train and test.py generators with data Augumentation
    datagen = get_data_gen()

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode="categorical")

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_shape,
        class_mode="categorical")

    return files_train, files_validation, train_generator, validation_generator
