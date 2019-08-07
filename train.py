import model
import dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import time

files_train = 0
files_validation = 0

batch_size = 32
epochs = 15
num_classes = 2

# globe config
nb_train_samples, nb_validation_samples, train_generator, validation_generator = dataset.get_data()


# build model
model_final = model.model_cnn((48, 48, 3), num_classes)

# model_final.load_weights("car1.h5", by_name=True)
# Save the model according to the conditions
checkpoint = ModelCheckpoint("car1_2.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

tensor_board = TensorBoard(log_dir="logs_cnn/PUC/Cloudy", histogram_freq=0, batch_size=32, update_freq='batch')

# Start training!
history_object = model_final.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    callbacks=[checkpoint, early, tensor_board])


import matplotlib.pyplot as plt

print(history_object.history.keys())
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test.py'], loc='upper left')
plt.show()

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test.py'], loc='upper left')
plt.show()
