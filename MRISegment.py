# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:49:33 2026

@author: sam-p
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import random
import keras
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.utils import load_img, img_to_array
from keras.layers import Rescaling, Conv2D, Conv2DTranspose, Dense
    
def InputPath(path):
    return img_to_array(load_img(path, target_size=ImgSize))

def TargetPath(path):
    img = img_to_array(
        load_img(path, target_size=ImgSize, color_mode="grayscale")
    )
    # Make sure lesions masks are [0,1]
    img = np.divide(np.abs(np.subtract(img.astype("uint8"),256)),255)
    return img

def get_model(ImgSize, num_classes):
    inputs = keras.Input(shape=ImgSize + (3,))
    # Rescale MRI images as they often start at 40-60
    x = Rescaling(1.0 / 255)(inputs)

    x = Conv2D(8, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(8, 3, activation="relu", padding="same")(x)
    x = Conv2D(16, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(16, 3, activation="relu", padding="same")(x)

    x = Conv2DTranspose(16, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(8, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same")(x)

    # Classify using sigmoid 
    outputs = Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

InputDir = pathlib.Path("C:/Users/sam-p/Documents/MRIData/MRImgs")
TargetDir = pathlib.Path("C:/Users/sam-p/Documents/MRIData/LesionMasks")
ImgSize = (32, 32)
InputImgPath = sorted(InputDir.glob("*.png"))
TargetImgPath = sorted(TargetDir.glob("*.png"))

# Total number of samples in the data
ImgCount = len(InputImgPath)
random.Random(1066).shuffle(InputImgPath)
random.Random(1066).shuffle(TargetImgPath)

InputImgs = np.zeros((ImgCount,) + ImgSize + (3,), dtype="float32")
TargetImgs = np.zeros((ImgCount,) + ImgSize + (1,), dtype="uint8")
for i in range(ImgCount):
    InputImgs[i] = InputPath(InputImgPath[i])
    TargetImgs[i] = TargetPath(TargetImgPath[i])
    
# Create Training and validation
NumVals = 1000
TrainInputImgs = InputImgs[:-NumVals]
TrainTargetImgs = TargetImgs[:-NumVals]
ValidationInputImgs = InputImgs[-NumVals:]
ValidationTargetImgs = TargetImgs[-NumVals:]

model = get_model(ImgSize=ImgSize, num_classes=2)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy()]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "oxford_segmentation.keras",
        save_best_only=True,
    ),
]

history = model.fit(
    TrainInputImgs,
    TrainTargetImgs,
    epochs=50,
    callbacks=callbacks,
    batch_size=64,
    validation_data=(ValidationInputImgs, ValidationTargetImgs),
)

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
ValLoss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "r--", label="Training MSE")
plt.plot(epochs, ValLoss, "k", label="Validation MSE")
plt.title("Training and validation MSE")
plt.ylabel("Mean square error")
plt.xlabel("# of Epochs")
plt.legend()
    