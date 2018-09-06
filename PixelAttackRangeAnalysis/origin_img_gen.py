# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

INDEX = 8447
image = Image.fromarray(np.uint8(X_test[INDEX]))
image.save(str(INDEX) + '.jpg')
