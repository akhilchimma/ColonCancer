#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os,re,h5py
import time
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten,Conv2D,MaxPooling2D, Dropout,BatchNormalization,ZeroPadding2D,Concatenate,Input
from keras.models import Model
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return true_negatives / (true_negatives+false_positives + K.epsilon())
    
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

PATH = "/content/drive/MyDrive/Minor_Project/crchistophenotypes32_32"
print("PWD",PATH)
# Define data path
data_path = PATH
data_dir_list = os.listdir(data_path)
print(data_dir_list)

x_train=[]
x_test = []

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

for dataset in sorted_alphanumeric(data_dir_list):
	img_list=sorted_alphanumeric(os.listdir(data_path+'/'+ dataset))
	#print(img_list)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		#print(img)
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(32,32))
		#print(type(img))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		#print('Input image shape:', x.shape)
		x_train.append(x)

x_train = np.array(x_train)
print (x_train.shape)
x_train=np.rollaxis(x_train,1,0)
print (x_train.shape)
x_train=x_train[0]
print ('train shape',x_train.shape)

num_classes = 4
num_of_samples = x_train.shape[0]
print("sample",num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7722] = 0
labels[7722:13434] = 1
labels[13434:20225] = 2
labels[20225:] = 3

from sklearn.utils import shuffle
Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(x_train, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = Sequential()
model.add(ZeroPadding2D(padding=(2,2), input_shape=(32,32,3)))
model.add(Conv2D(36,5, activation='relu'))

model.add(Conv2D(36, 4,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48, 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.summary()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=6, min_lr=0.5e-10)
csv_logger = CSVLogger('/content/drive/MyDrive/Minor_Project/Records/MobileNet/MobileNet_without_emjay.csv', append=True, separator=';')

adam = Adam(lr=0.00006, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1.0e-6,amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',f1,specificity,sensitivity])

batch_size =64
data_augmentation = True
epochs = 500

import time
start_time = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=[lr_reducer,csv_logger])
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  validation_data=(x_test, y_test), callbacks=[lr_reducer, csv_logger])
print("---  Training time in seconds ---%s " % (time.time() - start_time))


scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test f1:',scores[2])
print('Test specificity:',scores[3])
print('Test sensitivity:',scores[4])


import matplotlib.pyplot as plt
print('Max Test accuracy:', max(history.history['val_f1']))
# # visualizing losses and accuracy
print(history.history.keys())
print(history.history.values())
# # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/content/drive/MyDrive/Minor_Project/Records/MobileNet/MobileNet_without_acc_emjay.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/content/drive/MyDrive/Minor_Project/Records/MobileNet/MobileNet_without_loss_emjay.png')
plt.show()

