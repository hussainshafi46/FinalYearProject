# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:03:51 2021

@author: anuja
"""

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model

import matplotlib.pyplot as plt

#trainPath = r'C:\Users\anuja\Downloads\emotion-images\images\images\train'
#validationPath = r'C:\Users\anuja\Downloads\emotion-images\images\images\validation'
trainPath= r'F:\Final Year Project\New DataSet\Train'
validationPath=r'F:\Final Year Project\New DataSet\Val'
testPath=r'F:\Final Year Project\New DataSet\Test'

#train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_generator = ImageDataGenerator(rescale=1./255)
valid_generator = ImageDataGenerator(rescale=1./255)

train_ds = train_generator.flow_from_directory(
    trainPath,
    target_size=(48,48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    #subset='training'
    )

test_ds = train_generator.flow_from_directory(
    testPath,
    #trainPath,
    target_size=(48,48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    #subset='validation'
    )

valid_ds = valid_generator.flow_from_directory(
    validationPath,
    target_size=(48,48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=True
    )

def conv(filters, input_size=None, pool=False):
    layers = []
    if input_size:
        layers.append(L.Input(shape=input_size))
    layers.extend([
        L.Conv2D(filters, (3, 3), padding='same'),
        L.ReLU(),
        L.BatchNormalization(axis=-1, momentum=1e-5, epsilon=1e-3)
    ])
    if pool:
        layers.append(L.MaxPool2D(pool_size=(2, 2)))
    return Sequential(layers)

def res(filters, input_size=None):
    return Sequential([conv(filters, input_size=input_size), conv(filters, input_size=input_size)])

def classify(classes):
    return Sequential([
        L.Flatten(),
        L.Dense(1024),
        L.ReLU(),
        L.Dense(classes, activation='softmax')
    ])

class FER(Model):
    def __init__(self, classes=3, input_size=(48, 48, 1)):
        super(FER, self).__init__()
        
#         size = input_size # 48x48
        size = None
    
        self.conv1_1 = conv(32, input_size=input_size)
        self.conv1_2 = conv(64, input_size=size, pool=True)
        
#         size = (size[0]//2, size[1]//2, size[2]) # 24x24
        self.res1 = res(64, input_size=size)
        
        self.conv2_1 = conv(64, input_size=size)
        self.conv2_2 = conv(128, input_size=size, pool=True)
        
#         size = (size[0]//2, size[1]//2, size[2]) # 12x12
        self.res2 = res(128, input_size=size)
        
        self.conv3_1 = conv(128, input_size=size)
        self.conv3_2 = conv(256, input_size=size, pool=True)
        
#         size = (size[0]//2, size[1]//2, size[2]) # 6x6
        self.res3 = res(256, input_size=size)
        
        self.conv4_1 = conv(256, input_size=size)
        self.conv4_2 = conv(512, input_size=size, pool=True)
        
#         size = (size[0]//2, size[1]//2, size[2]) # 3x3
        self.res4 = res(512, input_size=size)
        
        self.classifier = classify(classes)
        
    def call(self, batch):
        out = self.conv1_1(batch)
        out = self.conv1_2(out)
        out = self.res1(out) + out
        
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.res2(out) + out
        
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.res3(out) + out
        
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.res4(out) + out
        
        out = self.classifier(out)
        
        return out
    
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5
)
checkpoint = ModelCheckpoint(
    filepath='checkpoint',
    save_freq='epoch',
    save_best_only=True
)

callbacks=[early_stop, checkpoint]

fer = FER(classes=3)
fer.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(.0001),
    metrics=['accuracy']
)

history = fer.fit(
    train_ds,
    epochs=10,
    validation_data=valid_ds,
    #callbacks=callbacks,
    steps_per_epoch=train_ds.n//32,
    validation_steps=valid_ds.n//32,
    verbose=1,
    workers=4, 
    use_multiprocessing=False
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
