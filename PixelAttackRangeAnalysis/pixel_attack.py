# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
from sklearn.metrics import accuracy_score
import json

model_path = './model_cnn/model3.hdf5'
model = load_model(model_path)

IMG_WIDTH = 32
IMG_HEIGHT = 32


def accuracy_cifar(model):
    """
    Get cifar cnn model accuracy.

    :param model: CNN_model
    :return: Accuracy of cifar
    """
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    print(type(X_test))
    X_test = X_test.astype(np.float32)
    X_test = X_test[0: 100]
    pred = model.predict(X_test)
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = Y_test.reshape(-1, 1)
    test_label = test_label[0: 100]
    # 好像没有必要写成下面的形状，pd.get_dummies可以把argmax后的值展开为[0 0 ... 0]的one-hot数组，
    # 然后再argmax回去有点多此一举
    # test_label = list(map(lambda x: np.argmax(x), pd.get_dummies(Y_test.reshape(-1)).values))
    return accuracy_score(test_label, pred)


def accuracy_cifar_change(model, x_random, y_random, channel_val_random, channel=0, index_random=0):
    """
    Change image one pixel channel value.

    :param model:
    :param x_random: pixel's position on x axis
    :param y_random: pixel's position on y axis
    :param channel_val_random: Random value to modify channel
    :param channel: RGB channel
    :param index_random: Random image batch start point
    :return:
    """
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_test = X_test.astype(np.float32)
    X_test = X_test[100 * index_random: 100 + 100 * index_random]
    for i in range(100):
        X_test[i][x_random][y_random][channel] = channel_val_random
    pred = model.predict(X_test)
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = Y_test.reshape(-1, 1)
    return accuracy_score(test_label, pred)


class PixelAttack(object):
    """
    Pixel Attack Class, for random sampling.
    """

    def __init__(self, model):
        self.list = []
        self.right_to_right_list = []
        self.right_to_wrong_list = []
        self.wrong_to_right_list = []
        self.wrong_to_wrong_list = []

    def one_channel_change(self, x_random, y_random, channel_val_random, channel=0, index_random=0):
        """
        Change image one pixel channel value.

        :param x_random: pixel's position on x axis
        :param y_random: pixel's position on y axis
        :param channel_val_random: Random value to modify channel
        :param channel: RGB channel
        :param index_random: Random image batch start point
        :return:
        """
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_test = X_test.astype(np.float32)
        X_test = X_test[100 * index_random: 100 + 100 * index_random]
        origin_pred = model.predict(X_test)
        origin_pred = list(map(lambda x: np.argmax(x), origin_pred))

        for i in range(100):
            X_test[i][x_random][y_random][channel] = channel_val_random

        changed_pred = model.predict(X_test)
        changed_pred = list(map(lambda x: np.argmax(x), changed_pred))

        assert len(origin_pred) == len(changed_pred)

        test_label = Y_test.reshape(-1, 1)
        test_label = test_label[100 * index_random: 100 + 100 * index_random]

        for i in range(len(origin_pred)):
            result = {
                'image_index': int(index_random * 100 + i),
                'pixel_position': [int(x_random), int(y_random), int(channel)],
                'channel_val': int(channel_val_random),
                'origin_predict': int(origin_pred[i]),
                'changed_predict': int(changed_pred[i])
            }
            if origin_pred[i] == changed_pred[i]:
                if origin_pred[i] == test_label[i]:
                    self.right_to_right_list.append(result)
                else:
                    self.wrong_to_wrong_list.append(result)
            else:
                if origin_pred[i] == test_label[i]:
                    self.right_to_wrong_list.append(result)
                elif changed_pred[i] == test_label[i]:
                    self.wrong_to_right_list.append(result)
        print('Right to Right Set Size: {}'.format(len(self.right_to_right_list)))
        print('Right to Wrong Set Size: {}'.format(len(self.right_to_wrong_list)))
        print('Wrong to Right Set Size: {}'.format(len(self.wrong_to_right_list)))
        print('Wrong to Wrong Set Size: {}'.format(len(self.wrong_to_wrong_list)))
        pass


if __name__ == '__main__':
    # print(model.summary())
    origin_accuracy = accuracy_cifar(model)
    print('accuracy of origin: {}'.format(origin_accuracy))

    x_random = np.random.randint(0, 8)
    y_random = np.random.randint(0, 8)
    channel_val = np.random.randint(0, 16)

    ran = np.random.randint(0, 100)

    print('ran: {}'.format(ran))

    attack = PixelAttack(model)

    for i in range(0, int(IMG_WIDTH / 8)):
        for j in range(0, int(IMG_HEIGHT / 8)):
            for k in range(0, 3):
                for l in range(16):
                    x_temp = x_random + i * 8
                    y_temp = y_random + j * 8
                    channel_val_temp = channel_val + l * 16

                    print('position: ({}, {}, {}), channel_val: {}'.format(x_temp, y_temp, k, channel_val_temp))

                    attack.one_channel_change(x_temp, y_temp, channel_val_temp, k, ran)

    with open('data/RtR1.json', 'w') as f:
        f.write(json.dumps(attack.right_to_right_list))

    with open('data/RtW1.json', 'w') as f:
        f.write(json.dumps(attack.right_to_wrong_list))

    with open('data/WtR1.json', 'w') as f:
        f.write(json.dumps(attack.wrong_to_right_list))

    with open('data/WtW1.json', 'w') as f:
        f.write(json.dumps(attack.wrong_to_wrong_list))
