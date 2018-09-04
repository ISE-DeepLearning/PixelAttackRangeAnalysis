# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from keras.models import load_model
import json

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
model_path = './model_cnn/model3.hdf5'
model = load_model(model_path)

IMAGE_INDEX = 40
POSITION = [29, 7, 1]
VAL_START = 2
VAL_END = 18

for i in range(VAL_START, VAL_END + 1):
    images = X_test[IMAGE_INDEX: IMAGE_INDEX + 1]
    images[0][POSITION[0]][POSITION[1]][POSITION[2]] = i

    preds = model.predict(images)
    preds = list(map(lambda x: np.argmax(x), preds))

    test_labels = Y_test[IMAGE_INDEX: IMAGE_INDEX + 1]
    print(accuracy_score(test_labels, preds))
