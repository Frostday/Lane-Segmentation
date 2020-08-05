import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import sys
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *


# Part 1 - Data Preprocessing

def get_mask(img_path, label_path):
    label_file = open(label_path, "r")
    if label_file.mode == 'r':
        contents = label_file.read()
        lines_text = contents.split('\n')

        x_coordinate, y_coordinate, lanes = [], [], []

        for line_text in lines_text:
            number_lines = line_text.split(" ")
            number_lines.pop()

            x = list([float(number_lines[i]) for i in range(len(number_lines)) if i % 2 == 0])
            y = list([float(number_lines[i]) for i in range(len(number_lines)) if i % 2 != 0])

            x_coordinate.append(x)
            y_coordinate.append(y)

            lanes.append(set(zip(x, y)))

        lanes.pop()
        img = cv2.imread(img_path)
        mask = np.zeros_like(img)
        # colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0]]
        colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
        for i in range(len(lanes)):
            cv2.polylines(img, np.int32([list(lanes[i])]), isClosed=False, color=colors[i], thickness=10)
        label = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return label


img = get_mask("data/CU_Lane/content/data/images/driver_161_90frame_06030819_0755.MP400000.jpg",
               "data/CU_Lane/content/data/labels/driver_161_90frame_06030819_0755.MP400000.lines.txt")
plt.imshow(img)
print(img.shape)

import os
from tensorflow.keras.utils import Sequence

import os
from tensorflow.keras.utils import Sequence


class DataGenerator2D(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, base_path, img_size=256, batch_size=1, shuffle=True):

        self.base_path = base_path
        self.img_size = img_size
        self.id = os.listdir(os.path.join(base_path, "images"))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.id) / float(self.batch_size)))

    def __load__(self, id_name):
        id_name_actual, text, _ = id_name.split('.')
        image_path = os.path.join(self.base_path, "images", (id_name_actual + '.' + text + '.jpg'))
        label_path = os.path.join(self.base_path, "labels", (id_name_actual + '.' + text + '.lines.txt'))

        image = cv2.imread(image_path, 1)  # Reading Image in RGB format
        image = cv2.resize(image, (self.img_size, self.img_size))
        # image = cv2.resize(image, (int(img.shape[1]/2), int(img.shape[0]/2)))

        mask = get_mask(image_path, label_path)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        # mask = cv2.resize(mask, (int(img.shape[1]/2), int(img.shape[0]/2)))

        # Normalizing the image
        image = image / 255.0
        mask = mask / 255.0

        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.id):
            file_batch = self.id[index * self.batch_size:]
        else:
            file_batch = self.id[index * self.batch_size:(index + 1) * self.batch_size]

        images, masks = [], []

        for id_name in file_batch:
            _img, _mask = self.__load__(id_name)
            images.append(_img)
            masks.append(_mask)

        images = np.array(images)
        masks = np.array(masks)

        return images, masks

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


train_generator = DataGenerator2D(base_path='data/CU_Lane/content/data', img_size=256, batch_size=64, shuffle=False)
X, y = train_generator.__getitem__(0)
print(X.shape, y.shape)

fig = plt.figure(figsize=(17, 8))
columns = 4
rows = 3
for i in range(1, columns*rows + 1):
    img = X[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

fig = plt.figure(figsize=(17, 8))
columns = 4
rows = 3
for i in range(1, columns*rows + 1):
    img = y[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()


# Part 2 - Model

def unet(sz = (256, 256, 3)):
  x = Input(sz)
  inputs = x
  
  #down sampling 
  f = 8
  layers = []
  
  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    layers.append(x)
    x = MaxPooling2D() (x)
    x = BatchNormalization()(x)
    f = f*2
  ff2 = 64
  
  #bottleneck 
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = BatchNormalization()(x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = BatchNormalization()(x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1 
  
  #upsampling 
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2 
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = BatchNormalization()(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 
    
  
  #classification 
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = BatchNormalization()(x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = BatchNormalization()(x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)
  
  model = Model(inputs=[inputs], outputs=[outputs])
  
  return model


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

from tensorflow.keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = 100*precision(y_true, y_pred)
    recall = 100*recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# COMPILING MODEL

model = unet()

opt = tf.keras.optimizers.Adam(0.001)
metrics = ["acc", iou]

model.compile(loss=dice_coef_loss,
              optimizer=opt,
              metrics=metrics)

model.summary()

train_generator = DataGenerator2D('data/CU_Lane/content/data', img_size=256, batch_size=128, shuffle=True)
val_generator = DataGenerator2D('data/CU_Lane/content/data', img_size=256, batch_size=128, shuffle=False)

# Part 3 - Training

history = model.fit_generator(generator=train_generator,
                              validation_data=val_generator,
                              steps_per_epoch=200,
                              validation_steps=5,
                              epochs=10)

print(history.history.keys())


# Part 4 - Visualization

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

val_generator = DataGenerator2D('content/data/', img_size=256,batch_size=128, shuffle=True)
X, y = val_generator.__getitem__(10)
print(X.shape, y.shape)

plt.imshow(X[2])

predict = model.predict(X)
print(predict.shape)
img = cv2.cvtColor(predict[2], cv2.COLOR_GRAY2BGR)
plt.imshow(img)