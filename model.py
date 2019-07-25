from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense


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
                        metrics=["accuracy"])  # See learning rate is very low

    return model_final



