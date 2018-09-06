# -*- coding: utf-8 -*-

import json

import numpy as np
from PIL import Image
from keras.datasets import cifar10

PATH = 'images/1x1'

f = open('data/RtW1.json', 'r')
RtW_data = json.loads(f.read())


def modify_image(image, position=None, val=0):
    if position is None:
        position = [0, 0, 0]
    image = np.asarray(image, dtype=np.float32)
    image[position[0]][position[1]][position[2]] = val
    return image


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

for item in RtW_data:
    image = modify_image(X_test[item['image_index']], position=item['pixel_position'], val=item['channel_val'])
    img = Image.fromarray(np.uint8(image))
    img.save('{}/image{}_pixel{}_val{}_{}to{}.jpg'.format(PATH, item['image_index'],
                                                    item['pixel_position'],
                                                    item['channel_val'],
                                                    item['origin_predict'],
                                                    item['changed_predict']))
