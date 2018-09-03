#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.datasets import cifar10

from keras.layers import Conv2D
from keras.layers import Input
from keras.models import Model
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.layers import Convolution2D
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32
#Y是标签

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print X_train.shape
print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

nb_classes=10

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print('data success')
print(X_train.shape[1:])
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(64, 6, 6))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()
'''
model = Sequential()
model.add(Convolution2D(3072, 3, 3, border_mode='valid'))
model.add(Activation('sigmoid'))

model.add(Convolution2D(3072, 3, 3))
model.add(Activation('tanh'))
#model.add(Dropout(0.25))

model.add(Convolution2D(3072, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
'''

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print X_train.shape
model.fit(X_train, Y_train, batch_size=100, nb_epoch=10,validation_data=(X_test, Y_test))
#Y_pred = model.predict(X_test, verbose=0)
score = model.evaluate(X_test, Y_test, verbose=0)
print('测试集 score(val_loss): %.4f' % score[0])
print('测试集 accuracy: %.4f' % score[1])
model.save('./model_cnn/model3.hdf5')

