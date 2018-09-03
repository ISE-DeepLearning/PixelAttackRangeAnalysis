#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Activation
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import SGD
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#(X_train, Y_train), (X_test, Y_test) = input_data.load_data()  # 28*28
Y_train = mnist.train.labels
X_train = mnist.train.images
X_test = mnist.test.images
Y_test = mnist.test.labels
print X_train.shape
input_data=Input((28*28,))
temp_data=Dense(128)(input_data)
temp_data=Activation('relu')(temp_data)
temp_data=Dense(64)(temp_data)
temp_data=Activation('relu')(temp_data)
temp_data=Dense(10)(temp_data)
output_data=Activation('softmax')(temp_data)
model=Model(inputs=[input_data],outputs=[output_data])
modelcheck=ModelCheckpoint('model.hdf5',monitor='loss',verbose=1,save_best_only=True)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit([mnist.train.images],[mnist.train.labels],batch_size=256,epochs=5,callbacks=[modelcheck],validation_data=(mnist.test.images,mnist.test.labels))





#Y_pred = model.predict_proba(X_test, verbose=0)
score = model.evaluate(X_test, Y_test, verbose=0)
print('测试集 score(val_loss): %.4f' % score[0])
print('测试集 accuracy: %.4f' % score[1])
model.save('./model_bp/model1.hdf5')