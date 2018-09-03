# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
import csv
import json
from keras import Model

def HDF5_structure(data):
    root=data.keys()
    print root

#    print data[u'input_1'].items()
    final_path=[]
    data_path=[]
    while True:
        if len(root)==0:
            break
        else:
            for item in root:
                if isinstance(data[item],h5py._hl.dataset.Dataset) or len(data[item].items())==0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item],h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item,sub_item[0]))
                    root.remove(item)

    #data_path = os.path.join(u'input_1')
    return data_path


def model_mutation_single_neuron(model,cls='kernel',random_ratio=0.001,extent=1):
    '''
    model:keras DNN_model
    cls: 'kernel' or 'bias'
    extent:ratio
    '''
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    data=h5py.File('my_model_weight.h5','r+')
    data_path=HDF5_structure(data)
    lst=[]

    for path in data_path:
        if os.path.basename(path.split(':')[0])!=cls:
            continue
        if len(data[path].shape)==2:
            row,col=data[path].shape
            lst.extend([(path,i,j) for i in range(row) for j in range(col)])
        else:
            row=data[path].shape[0]
            lst.extend([(path,i) for i in range(row)])
    random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
    lst_random=np.array(lst)[[random_choice]]

    for path in lst_random:
        try:
            arr=data[path[0]][int(path[1])].copy()
            arr[int(path[2])]*=extent
            data[path[0]][int(path[1])]=arr
        except:
            arr=data[path[0]][int(path[1])]
            arr*=extent
            data[path[0]][int(path[1])]=arr
    data.close()
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    #print('parameter:{}'.format(model.count_params()))
    #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
    #print('extend :{}'.format(extent))
    return len(lst),data_path,model.count_params(),int(random_ratio*model.count_params()),model_change

def model_mutation_single_neuron_cnn(model,cls='kernel',layers='dense',random_ratio=0.001,extent=1):
    '''
    model:keras DNN_model or CNN_model
    cls: 'kernel' or 'bias'
    layers: 'dense' or 'conv'
    extent:ratio
    '''
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    data=h5py.File('my_model_weight.h5','r+')
    data_path=HDF5_structure(data)
    lst=[]

    for path in data_path:
        if os.path.basename(path.split(':')[0])!=cls:
            continue
        if layers not in path.split('/')[0]:
            continue

        if len(data[path].shape)==4:
            a,b,c,d=data[path].shape
            lst.extend([(path,a_index,b_index,c_index,d_index) for a_index in range(a) for b_index in range(b) for c_index in range(c) for d_index in range(d)])
        if len(data[path].shape)==2:
            row,col=data[path].shape
            lst.extend([(path,i,j) for i in range(row) for j in range(col)])
        else:
            row=data[path].shape[0]
            lst.extend([(path,i) for i in range(row)])
    random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
    lst_random=np.array(lst)[[random_choice]]

    for path in lst_random:
        if len(path)==3:
            arr=data[path[0]][int(path[1])].copy()
            arr[int(path[2])]*=extent
            data[path[0]][int(path[1])]=arr
        elif len(path)==2:
            arr=data[path[0]][int(path[1])]
            arr*=extent
            data[path[0]][int(path[1])]=arr
        elif len(path)==5:
            arr=data[path[0]][int(path[1])][int(path[2])][int(path[3])].copy()
            arr[int(path[4])]*=extent
            data[path[0]][int(path[1])][int(path[2])][int(path[3])]=arr
    data.close()
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    #print('parameter:{}'.format(model.count_params()))
    #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
    #print('extend :{}'.format(extent))
    return len(lst),data_path,model.count_params(),int(random_ratio*model.count_params()),model_change



class model_mutation_del_neuron(object):
    '''
    1、初始化是模型
    2、首先看有几个全链接层，以及全连接层每层有多少个神经元
    3、del_neuron的输入是第几层神经元和第几个神经元
    4、可反复变异
    '''

    def __init__(self,model):
        self.model=model

    def get_neuron(self):
        neuron_num=0
        layer_num=[]
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        data_path=HDF5_structure(data)
        #data_path = [u'conv2d_1/conv2d_1/kernel:0']
        self.data_path=[]
        #print data[u'input_1']
        for path in data_path:
            if os.path.basename(path.split(':')[0])!='kernel':
                continue

            self.data_path.append(path)
            neuron_num+=data[path].shape[0]
            layer_num.append(data[path].shape[0])
        print self.data_path
        #print self.data_path[2]
        #print data[self.data_path[2]].shape
        #print type(data[self.data_path[2]])
        #print data[self.data_path[2]][783].shape
        #print data[self.data_path[2]][456][67]
        print data_path
        data.close()
        return neuron_num,layer_num

    def del_neuron(self,data,neuron_index):
        '''
        neuron_index:(layer_num,index)
        '''
        layer_num,index=neuron_index
        print layer_num,index
        json_string=self.model.to_json()
        path=self.data_path[layer_num]
        #print data[path].shape
        data_change=data
        arr=data[path][index].copy()
        print arr
        arr*=0
        data_change[path][index]=arr
        
        #print('parameter:{}'.format(model.count_params()))
        #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
        #print('extend :{}'.format(extent))
        return

    def mask_input(self,ndim,index):
        '''
        ndim:总维数
        index:需要删除的维
        '''
        json_string=self.model.to_json()
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        for i in range(len(self.data_path)):
            if data[self.data_path[i]].shape[0]==ndim:
                for j in range(data[self.data_path[i]].shape[1]):
                    arr = data[self.data_path[i]][index].copy()
                    arr[j]*=0
                    data[self.data_path[i]][index]=arr
        data.close()
        model_change = model_from_json(json_string)
        model_change.load_weights('my_model_weight.h5')
        return model_change
    
    def del_neuron_random(self,ndim,num,loopnum):
    #num:每次删除的神经元个数
    #loopnum:循环的次数
        #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        lst=[]
        statis=[]
        for i in range(len(self.data_path)):
            #if data[self.data_path[i]].shape[0]==ndim:#记录一下是不是输入层神经元
                #break
            print self.data_path[i]
            print data[self.data_path[i]].shape[0]
            #if data[self.data_path[i]].shape[0]!=ndim:#记录一下是不是输入层神经元
            #    continue

            for index in range(data[self.data_path[i]].shape[0]):
                temp =(i,index)
                lst.append(temp)
        print lst
        #print lst
        json_string=self.model.to_json()
        model_temp=model_from_json(json_string)
        for loop in range(loopnum):
            random_choice=np.random.choice(len(lst),num)
            print random_choice
            for j in range(num):
                #print 'num',num
                #print lst[random_choice[j]] 
                self.del_neuron(data,lst[random_choice[j]])
            model_temp.load_weights('my_model_weight.h5')
            #statis.append(accuracy_mnist(model_temp,mnist))
            statis.append(accuracy_cifar(model_temp))

            #print 'accuracy of origin:',accuracy_mnist(self.model,mnist)
            #print 'accuracy of change:',accuracy_mnist(model_temp,mnist)
            print 'accuracy of origin:', accuracy_cifar(self.model)
            print 'accuracy of change:', accuracy_cifar(model_temp)
            self.killnumber(model_temp)
            data.close()
            self.model.save_weights('my_model_weight.h5')
            data=h5py.File('my_model_weight.h5','r+')
        data.close()
        return statis

    def killnumber(self, model):
        (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
        X_test = X_test.astype('float32')
        X_test /= 255
        pred1 = model.predict(X_test)  # The result of the trained model
        pred1 = list(map(lambda x: np.argmax(x), pred1))
        pred2 = self.model.predict(X_test)
        pred2 = list(map(lambda x: np.argmax(x), pred2))
        test_label = list(map(lambda x: np.argmax(x), pd.get_dummies(Y_test.reshape(-1)).values))
        rightToWrongResult = []
        wrongToRightResult = []
        for i in range (len(pred1)):

            if(pred1[i]==pred2[i]):
                continue
            if(pred2[i]==test_label[i]):
                rightToWrongResult.append(i)
            else:
                wrongToRightResult.append(i)

        print ('Right to Wrong is :' , rightToWrongResult)
        print ('Wrong to Right is :' , wrongToRightResult)

    def killnumber2(self, model,random1,random2,random3):
        (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
        X_test = X_test.astype('float32')
        pred1 = model.predict(X_test)  # The result of the trained model
        pred1 = list(map(lambda x: np.argmax(x), pred1))
        for i in range(10000):
            X_test[i][random1][random2][random3] = 0
        pred2 = model.predict(X_test)
        pred2 = list(map(lambda x: np.argmax(x), pred2))
        test_label = list(map(lambda x: np.argmax(x), pd.get_dummies(Y_test.reshape(-1)).values))
        rightToWrongResult = []
        wrongToRightResult = []
        for i in range (len(pred1)):
            if(pred1[i]==pred2[i]):
                continue
            if(pred1[i]==test_label[i]):
                rightToWrongResult.append(i)
            else:
                wrongToRightResult.append(i)

        print ('Right to Wrong is :' , rightToWrongResult)
        print len(rightToWrongResult)
        print ('Wrong to Right is :' , wrongToRightResult)
        print len(wrongToRightResult)

    '''
    def killnumber3(self, model,random1,random2,random3,l,ran):
        (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
        X_test = X_test.astype('float32')
        X_test = X_test[100*ran:100+100*ran]
        pred1 = model.predict(X_test)  # The result of the trained model
        pred1 = list(map(lambda x: np.argmax(x), pred1))
        for i in range(100):
            X_test[i][random1][random2][l] = random3

        pred2 = model.predict(X_test)
        pred2 = list(map(lambda x: np.argmax(x), pred2))
        test_label = list(map(lambda x: np.argmax(x), pd.get_dummies(Y_test.reshape(-1)).values))
        test_label = test_label[100*ran:100+100*ran]
        rightToWrongResult = []
        wrongToRightResult = []
        for i in range (len(pred1)):
            if(pred1[i]==pred2[i]):
                continue
            if(pred1[i]==test_label[i]):
                rightToWrongResult.append(i)
            if (pred2[i] == test_label[i]):
                wrongToRightResult.append(i)

        print ('Right to Wrong is :' , rightToWrongResult)
        print len(rightToWrongResult)
        print ('Wrong to Right is :' , wrongToRightResult)
        print len(wrongToRightResult)
        del_.rightToWrongList.append(rightToWrongResult)
        del_.wrongToRightList.append(wrongToRightResult)
    '''

    def killnumber3(self, model,random1,random2,random3,random4, ran, number1, number2):
        (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
        X_test = X_test.astype('float32')
        X_test = X_test[100*ran:100+100*ran]
        pred1 = model.predict(X_test)  # The result of the trained model
        pred1 = list(map(lambda x: np.argmax(x), pred1))
        for i in range(100):
            X_test[i][random1][random2][number1] = random3
            X_test[i][random1][random2][number2] = random4
        pred2 = model.predict(X_test)
        pred2 = list(map(lambda x: np.argmax(x), pred2))
        test_label = list(map(lambda x: np.argmax(x), pd.get_dummies(Y_test.reshape(-1)).values))
        test_label = test_label[100*ran:100+100*ran]
        rightToWrongResult = []
        wrongToRightResult = []
        for i in range (len(pred1)):
            if(pred1[i]==pred2[i]):
                continue
            if(pred1[i]==test_label[i]):
                rightToWrongResult.append(i)
            if (pred2[i] == test_label[i]):
                wrongToRightResult.append(i)

        print ('Right to Wrong is :' , rightToWrongResult)
        print len(rightToWrongResult)
        print ('Wrong to Right is :' , wrongToRightResult)
        print len(wrongToRightResult)
        del_.rightToWrongList.append(rightToWrongResult)
        del_.wrongToRightList.append(wrongToRightResult)
    '''
    def killnumber3(self, model, random1, random2, random3, random4, random5, random6, l, ran):
        (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
        X_test = X_test.astype('float32')
        X_test = X_test[100 * ran:100 + 100 * ran]
        pred1 = model.predict(X_test)  # The result of the trained model
        pred1 = list(map(lambda x: np.argmax(x), pred1))
        for i in range(100):
            X_test[i][random1][random2][l] = random3
        for i in range(100):
            X_test[i][random4][random5][l] = random6
        pred2 = model.predict(X_test)
        pred2 = list(map(lambda x: np.argmax(x), pred2))
        test_label = list(map(lambda x: np.argmax(x), pd.get_dummies(Y_test.reshape(-1)).values))
        test_label = test_label[100 * ran:100 + 100 * ran]
        rightToWrongResult = []
        wrongToRightResult = []
        for i in range(len(pred1)):
            if (pred1[i] == pred2[i]):
                continue
            if (pred1[i] == test_label[i]):
                rightToWrongResult.append(i)
            if (pred2[i] == test_label[i]):
                wrongToRightResult.append(i)

        print ('Right to Wrong is :', rightToWrongResult)
        print len(rightToWrongResult)
        print ('Wrong to Right is :', wrongToRightResult)
        print len(wrongToRightResult)
        del_.rightToWrongList.append(rightToWrongResult)
        del_.wrongToRightList.append(wrongToRightResult)
    '''
    def killnumberMnist(self, model, random1, random2, random3, ran):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        Y_train = mnist.train.labels
        X_train = mnist.train.images
        X_test = mnist.test.images
        Y_test = mnist.test.labels
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_test = X_test[100 * ran:100 + 100 * ran]

        pred1 = model.predict(X_test)  # The result of the trained model
        pred1 = list(map(lambda x: np.argmax(x), pred1))
        print (pred1)
        for i in range(100):
            X_test[i][temp1][temp2] = temp3
        pred2 = model.predict(X_test)
        pred2 = list(map(lambda x: np.argmax(x), pred2))
        print (pred2)
        test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
        test_label = test_label[100 * ran:100 + 100 * ran]
        rightToWrongResult = []
        wrongToRightResult = []
        print (test_label)
        for i in range(len(pred1)):
            if (pred1[i] == pred2[i]):
                continue
            if (pred1[i] == test_label[i]):
                rightToWrongResult.append(i)
            if (pred2[i] == test_label[i]):
                wrongToRightResult.append(i)

        print ('Right to Wrong is :', rightToWrongResult)
        print len(rightToWrongResult)
        print ('Wrong to Right is :', wrongToRightResult)
        print len(wrongToRightResult)
        del_.rightToWrongList.append(rightToWrongResult)
        del_.wrongToRightList.append(wrongToRightResult)


'''
def accuracy_mnist(model,mnist):
    ''''''
    model: DNN_model
    return : acc of mnist
    ''''''
    pred=model.predict(mnist.test.images)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),mnist.test.labels))
    return accuracy_score(test_label,pred)
'''
def accuracy_mnist(model):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    Y_train = mnist.train.labels
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    pred = model.predict(X_test)
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    return accuracy_score(test_label, pred)

def accuracy_mnist_part1(model,ran):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    Y_train = mnist.train.labels
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_test = X_test[ran*100:ran*100+100]

    pred = model.predict(X_test)
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    test_label = test_label[ran*100:ran*100+100]
    return accuracy_score(test_label, pred)

def accuracy_mnist_part2(model,temp1, temp2, temp3, ran):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    Y_train = mnist.train.labels
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_test = X_test[ran * 100:ran * 100 + 100]
    for i in range(100):
        X_test[i][temp1][temp2] = temp3
    pred = model.predict(X_test)
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    test_label = test_label[ran * 100:ran * 100 + 100]
    return accuracy_score(test_label, pred)


def accuracy_cifar(model):
    #model: CNN_model
    #return : acc of cifar
    #(_, _), (X_test, Y_test) = cifar10.load_data()
    (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
    print (type(X_test))
    X_test=X_test.astype('float32')
    X_test = X_test[0:100]
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    test_label = test_label[0:100]
    return accuracy_score(test_label,pred)

def accuracy_cifar_change(model,random1,random2,random3):
    #model: CNN_model
    #return : acc of cifar
    #(_, _), (X_test, Y_test) = cifar10.load_data()
    (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    for i in range (10000):
        X_test[i][random1][random2][random3] = 0
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    return accuracy_score(test_label,pred)

#SinglePixel
def accuracy_cifar_change(model,random1,random2,random3,l,ran):
    #model: CNN_model
    #return : acc of cifar
    #(_, _), (X_test, Y_test) = cifar10.load_data()
    (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    X_test = X_test[100*ran:100+100*ran]
    for i in range (100):
        X_test[i][random1][random2][l] = random3
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    test_label = test_label[100*ran:100+100*ran]
    return accuracy_score(test_label,pred)

#TwoPixels
def accuracy_cifar_change(model,random1,random2,random3,random4,random5,random6,l,ran):
    #model: CNN_model
    #return : acc of cifar
    #(_, _), (X_test, Y_test) = cifar10.load_data()
    (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    X_test = X_test[100*ran:100+100*ran]
    for i in range (100):
        X_test[i][random1][random2][l] = random3
    for i in range(100):
        X_test[i][random4][random5][l] = random6
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    test_label = test_label[100*ran:100+100*ran]
    return accuracy_score(test_label,pred)

#TwoColors
def accuracy_cifar_change(model,random1,random2,random3,random4, ran, number1, number2):
    #model: CNN_model
    #return : acc of cifar
    #(_, _), (X_test, Y_test) = cifar10.load_data()
    (x_train, y_train), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    X_test = X_test[100*ran:100+100*ran]
    for i in range (100):
        X_test[i][random1][random2][number1] = random3
        X_test[i][random1][random2][number2] = random4
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    test_label = test_label[100*ran:100+100*ran]
    return accuracy_score(test_label,pred)

def get_activation_layers(model):
    layers = []
    model_detail = json.loads(model.to_json())
    layer_detials = model_detail['config']['layers']
    for layer in layer_detials:
        print(layer)
        if layer['class_name'] == 'Activation':
            layer_model = Model(inputs=model.input, outputs=model.get_layer(layer['name']).output)
            layers.append(layer_model)
    # 删除最后一层的softmax/sigmod之类的分类激活函数
    layers = layers[:-1]
    return layers

if __name__=='__main__':
    '''
    #cnn_mnist
    model_path = './model_cnn/model4.hdf5'
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model = load_model(model_path)
    #print(model.summary())
    #print 'accuracy of origin:', accuracy_mnist(model,mnist)
    ran = np.random.randint(0, 100)
    print 'accuracy of origin:', accuracy_mnist_part1(model,ran)
    del_ = model_mutation_del_neuron(model)
    del_.rightToWrongList = []
    del_.wrongToRightList = []
    random1 = np.random.randint(0,4)
    random2 = np.random.randint(0,4)
    random3 = np.random.randint(0,10)
    del_.list = []
    Y_train = mnist.train.labels
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    for i in range(7):
        for j in range(7):
            for k in range(10):
                temp1 = random1 + i*4
                temp2 = random2 + j*4
                temp3 = (random3+k*10)*0.01
                print temp1,temp2,temp3
                tempS = (str)(temp1) + ' ' + (str)(temp2) + ' ' + ' ' + (str)(temp3)
                del_.list.append(tempS)

                print 'accuracy of change:', accuracy_mnist_part2(model,temp1, temp2, temp3, ran)
                del_.killnumberMnist(model, temp1, temp2, temp3, ran)

    result1List = []
    result2List = []
    result1List.append(['number', 'point'])
    result2List.append(['number', 'point'])
    for i in range(100):
        result1List.append([])
        result2List.append([])
    for i in range(490):
        len1 = len(del_.rightToWrongList[i])
        for j in range(len1):
            tmp = del_.rightToWrongList[i][j]
            result1List[tmp].append(del_.list[i])
        len2 = len(del_.wrongToRightList[i])
        for j in range(len2):
            tmp = del_.wrongToRightList[i][j]
            result2List[tmp].append(del_.list[i])
    with open('RTW.csv', 'wb') as f:
        writer = csv.writer(f)
        csv.writer
        #     for row in datas:
        #         writer.writerow(row)
        #
        #     # 还可以写入多行
        writer.writerows(result1List)
    with open('WTR.csv', 'wb') as f:
        writer = csv.writer(f)
        csv.writer
        #     for row in datas:
        #         writer.writerow(row)
        #
        #     # 还可以写入多行
        writer.writerows(result2List)
    '''
    #cnn_cifar
    model_path = './model_cnn/model3.hdf5'
    model = load_model(model_path)
    print(model.summary())
    print 'accuracy of origin:', accuracy_cifar(model)
    del_ = model_mutation_del_neuron(model)
    del_.rightToWrongList = []
    del_.wrongToRightList = []
    random1 = np.random.randint(0,8)
    random2 = np.random.randint(0,8)
    random3 = np.random.randint(0, 16)
    del_.list = []
    ran = np.random.randint(1, 100)
    print ran
    for i in range (4):
        for j in range (4):
            for k in range (16):
                temp1 = random1 + i * 8
                temp2 = random2 + j * 8
                temp3 = random3 + k * 16
                random4 = np.random.randint(0, 16)
                for l in range (16):
                    temp4 = random4 + l*16

                    print (str)(temp1)+' '+(str)(temp2)+' R:'+(str)(temp3) + ' G:' +(str)(temp4)
                    tempS = (str)(temp1)+' '+(str)(temp2)+' R:'+(str)(temp3)+ ' G:' +(str)(temp4)
                    del_.list.append(tempS)
                    # print 'accuracy of change:', accuracy_cifar_change(model, temp1,temp2,temp3,l,ran)
                    print 'accuracy of change:', accuracy_cifar_change(model, temp1, temp2, temp3, temp4, ran,0,1)
                    del_.killnumber3(model, temp1, temp2, temp3, temp4, ran, 0, 1)

                    print (str)(temp1)+ ' ' + (str)(temp2) + ' R:' + (str)(temp3) + ' B:' + (str)(temp4)
                    tempS = (str)(temp1) + ' ' + (str)(temp2) + ' R:' + (str)(temp3) + ' B:' + (str)(temp4)
                    del_.list.append(tempS)
                    # print 'accuracy of change:', accuracy_cifar_change(model, temp1,temp2,temp3,l,ran)
                    print 'accuracy of change:', accuracy_cifar_change(model, temp1, temp2, temp3, temp4, ran, 0, 2)
                    del_.killnumber3(model, temp1, temp2, temp3, temp4, ran, 0, 2)

                    print (str)(temp1)+ ' ' + (str)(temp2) + ' G:' + (str)(temp3) + ' B:' + (str)(temp4)
                    tempS = (str)(temp1) + ' ' + (str)(temp2) + ' G:' + (str)(temp3) + ' B:' + (str)(temp4)
                    del_.list.append(tempS)
                    # print 'accuracy of change:', accuracy_cifar_change(model, temp1,temp2,temp3,l,ran)
                    print 'accuracy of change:', accuracy_cifar_change(model, temp1, temp2, temp3, temp4, ran, 1, 2)
                    del_.killnumber3(model, temp1, temp2, temp3, temp4, ran, 1, 2)
    '''
    random1 = np.random.randint(0, 8)
    random2 = np.random.randint(0, 8)
    random3 = np.random.randint(0, 16)
    random4 = np.random.randint(0, 8)
    random5 = np.random.randint(0, 8)
    random6 = np.random.randint(0, 16)
    del_.list = []
    ran = np.random.randint(1, 100)

    print ran
    for i in range (4):
        for j in range (4):
            for k in range (16):
                for l in range (3):
                    #print 'accuracy of origin:', accuracy_cifar(model)
                    temp1 = random1 + i*8
                    temp2 = random2 + j*8
                    temp3 = random3 + k*16
                    print temp1,temp2,temp3
                    temp4 = random4 + i * 8
                    temp5 = random5 + j * 8
                    temp6 = random6 + k * 16
                    print temp4, temp5, temp6
                    tempS  = 'point1:'+(str)(temp1)+' '+(str)(temp2)+' '+(str)(l)+' '+(str)(temp3)+ '  ' + 'point2:'+(str)(temp4)+' '+(str)(temp5)+' '+(str)(l)+' '+(str)(temp6)
                    del_.list.append(tempS)
                    #print 'accuracy of change:', accuracy_cifar_change(model, temp1,temp2,temp3,l,ran)
                    print 'accuracy of change:', accuracy_cifar_change(model, temp1,temp2,temp3,temp4, temp5, temp6,l,ran)
                    #del_.killnumber3(model, temp1,temp2,temp3,l,ran)
                    del_.killnumber3(model, temp1, temp2, temp3, temp4, temp5, temp6, l, ran)
    '''
    result1List = []
    result2List = []
    result1List.append(['number','point'])
    result2List.append(['number', 'point'])
    for i in range (100):
        result1List.append([])
        result2List.append([])
    for i in range (12288):
        len1 = len(del_.rightToWrongList[i])
        for j in range (len1):
            tmp = del_.rightToWrongList[i][j]
            result1List[tmp].append(del_.list[i])
        len2 = len(del_.wrongToRightList[i])
        for j in range(len2):
            tmp = del_.wrongToRightList[i][j]
            result2List[tmp].append(del_.list[i])
    with open('RTW.csv', 'wb') as f:
        writer = csv.writer(f)
        csv.writer
            #     for row in datas:
            #         writer.writerow(row)
            #
            #     # 还可以写入多行
        writer.writerows(result1List)
    with open('WTR.csv', 'wb') as f:
        writer = csv.writer(f)
        csv.writer
            #     for row in datas:
            #         writer.writerow(row)
            #
            #     # 还可以写入多行
        writer.writerows(result2List)

    '''
    for i in range (random):
        print 'accuracy of origin:', accuracy_cifar(model)
        random1 = np.random.randint(0, 32)
        random2 = np.random.randint(0, 32)
        random3 = np.random.randint(0, 3)
        print random1, random2, random3
        print 'accuracy of change:', accuracy_cifar_change(model,random1,random2,random3)

        del_.killnumber2(model,random1,random2,random3)
    

#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#    model_path='./model_bp/model_raw.hdf5'
    model_path='./model_cnn/model2.hdf5'

#    ndim =784
    ndim = 1024
    model=load_model(model_path)
    print (model.summary())
    statislst=[]
#    print 'accuracy of origin:',accuracy_mnist(model,mnist)
    print 'accuracy of origin:',accuracy_cifar(model)
    del_=model_mutation_del_neuron(model)
    del_.get_neuron()
    #num_lst=[1,3,5,10,15,20]
    #num_lst=[1,2,5]
    num_lst=[1]
    xlabels=[]
    for num in num_lst:
        xlabels.append(str(num))
        statis = del_.del_neuron_random(ndim,num,20)
        statislst.append(statis)
    plt.boxplot(statislst,labels=xlabels) 
    plt.xlabel('Randomly delete n neurons from the hidden layer')
    plt.ylabel('Accuracy of the mutated model')
    plt.savefig("del_statis.png")
    
    #print statislst
    
    #model_change.save_weights('my_model_weight.h5')
    ##print model_change.load_weights
    #print HDF5_structure(data)
    #print accuracy_mnist(model_change,mnist)
''''''

    acc=[[]for i in range(28)]
    for i in range(ndim):
        model_change = del_.mask_input(ndim,i)
        acc[i/28].append(accuracy_mnist(model_change,mnist))
        #print accuracy_mnist(model_change,mnist)
    with open("mask_input.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(acc)
'''
    #neuron_index=(0,2)
    #model_del=del_neuron.del_neuron(neuron_index)

    #print('accuracy before mutaion:{}'.format(accuracy_cifar(model)))
    #_,_,_,_,model_mut=model_mutation_single_neuron(model,cls='kernel',extent=10,random_ratio=0.001)
    #_,_,_,_,model_mut=model_mutation_single_neuron_cnn(model,cls='kernel',layers='conv',random_ratio=0.001,extent=1)
    #print('accuracy after mutation:{}'.format(accuracy_cifar(model_mut)))
