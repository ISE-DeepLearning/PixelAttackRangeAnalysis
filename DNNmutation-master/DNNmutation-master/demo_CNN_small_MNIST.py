#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import MaxPooling2D
from keras.layers import Dense,Activation, Flatten
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.optimizers import SGD
from keras.utils import np_utils
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#(X_train, Y_train), (X_test, Y_test) = input_data.load_data()  # 28*28
Y_train = mnist.train.labels
X_train = mnist.train.images
X_test = mnist.test.images
Y_test = mnist.test.labels

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print X_train.shape
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

nb_classes=10

#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)
print (Y_train[0])
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
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

#model.summary()


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print X_train.shape
model.fit(X_train, Y_train, batch_size=100, nb_epoch=2,validation_data=(X_test, Y_test))
#Y_pred = model.predict(X_test, verbose=0)
score = model.evaluate(X_test, Y_test, verbose=0)
print('测试集 score(val_loss): %.4f' % score[0])
print('测试集 accuracy: %.4f' % score[1])
model.save('./model_cnn/model4.hdf5')
