import os

import keras.src.losses
import tensorflow as tf
from keras import layers, models, Sequential
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Parametros

epocas = 25
longitud,altura = 100,100
batch_size=64
pasos=812
pasos_validacion=200
filtroConv1=32
filtroConv2=64
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=20
lr=0.001



data_entrenamiento='./Dataset/entrenamiento'
#data_validacion='./Dataset/validacion'

#Con ImageDataGenerator

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
)
#validacion_datagen = ImageDataGenerator(
#    rescale=1./255
#
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse'
)

#imagen_validacion=validacion_datagen.flow_from_directory(
#    data_validacion,
#    target_size=(altura, longitud),
#    batch_size=batch_size,
#    color_mode='grayscale',
#    class_mode='sparse'
#)

# Construcción del modelo
model = models.Sequential()

model.add(layers.Conv2D(filtroConv1, tamano_filtro1, padding='same', activation='relu', input_shape=(altura, longitud, 1)))
model.add(layers.MaxPooling2D(pool_size=tamano_pool))
model.add(layers.Conv2D(filtroConv2, tamano_filtro2, padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=tamano_pool))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(clases, activation='softmax'))

model.compile(loss=keras.src.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])

#model.fit(imagen_entrenamiento,steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)

model.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas)

dir='./modelo'

if not os.path.exists(dir):
    os.mkdir(dir)

model.save('./modelo/modelo.h5')
model.save_weights('./modelo/pesos.weights.h5')
