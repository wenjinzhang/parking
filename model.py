from keras import applications, optimizers
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D


def model_cnn(input_shape=(48, 48, 3), num_classes=2):
    x = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    Flatten()(x)
    Dense(128, activation='relu')
    Dense(num_classes, activation='softmax')
    return model


def model(input_shape=(48, 48, 3), num_classes=2):
    image_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in image_model.layers[:10]:
        layer.trainable = False

    x = image_model.output
    x = Flatten()(x)
    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation="relu")(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model
    model_final = Model(inputs=image_model.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy",
                        optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])

    return model_final
