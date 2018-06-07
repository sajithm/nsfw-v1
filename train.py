from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

IMAGE_LENGTH = 128
EPOCH_COUNT = 12
MODEL_PATH = 'model.hdf5'
WEIGHTS_PATH = 'weights.hdf5'

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (IMAGE_LENGTH, IMAGE_LENGTH, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/train_set',
        target_size = (IMAGE_LENGTH, IMAGE_LENGTH),
        batch_size = 32,
        class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (IMAGE_LENGTH, IMAGE_LENGTH),
        batch_size = 32,
        class_mode = 'binary')

checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit_generator(
        training_set,
        epochs = EPOCH_COUNT,
        validation_data = test_set,
        callbacks = [checkpoint])
model.save(MODEL_PATH, True, True)
