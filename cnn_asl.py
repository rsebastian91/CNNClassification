# -*- coding: utf-8 -*-
"""

@author: robin
"""

import ML_module as ML

data_augmentation=True

# Load data
import pandas as pd

train_df = pd.read_csv('asl_data/sign_mnist_train.csv')
valid_df = pd.read_csv('asl_data/sign_mnist_test.csv')

# Split between train and validation sets
y_train = train_df['label'].values
y_valid = valid_df['label'].values
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values


MLobj=ML.Classification(x_train,y_train,x_valid,y_valid)

# Explore data
MLobj.check_data()


#Data preparation for training
MLobj.data_preparation(flaten=False,normalise=True)


#Target encoding
num_categories =25
MLobj.target_encoding(num_categories,encoding='binarymartix')


#################################################
#Creating model
from tensorflow.keras.models import Sequential
x_train=MLobj.x_train
y_train=MLobj.y_train

x_valid=MLobj.x_valid
y_valid=MLobj.y_valid

x_train = x_train.reshape(-1,28,28,1)
x_valid = x_valid.reshape(-1,28,28,1)


print("Creating model")
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_categories, activation="softmax"))

if data_augmentation==True:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=False, # Don't randomly flip images vertically
    )
    
    
    batch_size = 64
    img_iter = datagen.flow(x_train, y_train, batch_size=batch_size)
    
    datagen.fit(x_train)


#Model summary
model.summary()

#Model compiling
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

#Training model
nb_epochs=20


if data_augmentation==True:
    history=model.fit(img_iter,
              epochs=nb_epochs,
              steps_per_epoch=len(x_train)/batch_size, # Run same number of steps we would if we were not using a generator.
              validation_data=(x_valid, y_valid))

else:
    history = model.fit(
    x_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(x_valid, y_valid)
    )

acc = [element * 100 for element in history.history['accuracy']]
val_acc = [element * 100 for element in history.history['val_accuracy']]
loss = history.history['loss']
val_loss = history.history['val_loss']

#################################################


#plot accuracy and loss
MLobj.plot_acc_and_loss(acc,val_acc,loss,val_loss)

