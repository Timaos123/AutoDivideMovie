#coding:utf8
'''
Created on 2018年11月7日

@author: Administrator
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Lambda,Input,Dense,Reshape, Activation, Convolution2D, MaxPooling2D, Flatten, Bidirectional
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras import backend as K
from keras.callbacks import TensorBoard

from Bio.Cluster import kcluster 
from sklearn import metrics
import tqdm

class MyModel:
    
    def covLoss(self,y_true, y_pred):
        ytVar=K.var(y_true)
        ypVar=K.var(y_pred)
        cov=(ytVar+ypVar)/2
        return cov
        
    
    def __init__(self,**kwargs):
        '''
        you can design any kinds of model in this class
        ***note:add <subtitles>***
        ==================================================
        for <CNNClassifierModel>:
        
        batch_size(default: 50):batch size
        nb_classes(default: 10):the number of classes
        nb_epoch(default: 12):the number of epoch
        nb_filters(default: 32):the number of filters
        img_rows(default: 28 ):the height of matrixes
        img_cols(default: 28):the width of matrixes
        pool_size(default: (2,2)):the size of pools
        kernel_size(default: (3,3)): the size of kernels
        ==================================================
        for <**model name**>
        parameter
        '''
        self.paraDict=kwargs
    
    def buildCNNSeriesModel(self,\
                                nb_filters = 32,\
                                img_num=16,\
                                img_rows=256,\
                                img_cols =256,\
                                img_passes=3,\
                                pool_size = 5,\
                                kernel_size = (8,8),
                                vecLen=100):
        '''
        batch_size(default: 50):batch size
        nb_filters(default: 32):the number of filters
        img_rows(default: 28 ):the height of matrixes
        img_cols(default: 28):the width of matrixes
        img_passes(defaulst: 3):the passages of tensors
        pool_size(default: (2,2)):the size of pools
        kernel_size(default: (3,3)): the size of kernels
        '''
            
        try:
            self.nb_filters = self.paraDict['nb_filters']
        except:
            self.nb_filters =nb_filters
            
        try:
            self.img_rows, self.img_cols = self.paraDict['img_rows'],self.paraDict['img_cols']
        except:
            self.img_rows, self.img_cols =img_rows, img_cols
            
        try:
            self.pool_size = self.paraDict['pool_size']
        except:
            self.pool_size = pool_size
            
        try:
            self.kernel_size = self.paraDict['kernel_size']
        except:
            self.kernel_size=kernel_size
        
        try:
            self.img_passes=self.paraDict['img_passes']
        except:
            self.img_passes=img_passes
            
        try:
            self.img_num=self.paraDict['img_num']
        except:
            self.img_num=img_num
        
        try:
            self.vecLen=self.paraDict['vecLen']
        except:
            self.vecLen=vecLen
        self.input_shape = (self.img_num,self.img_rows, self.img_cols,self.img_passes)
        
        denseList=[]
        inputList=[Input(shape=self.input_shape[1:]) for i in range(self.img_num)]
        for i in range(self.img_num):
            conv2DL1=Convolution2D(self.nb_filters,\
                                   input_shape=self.input_shape,\
                                   kernel_size=(self.kernel_size[0],self.kernel_size[1]),\
                                   border_mode='valid',\
                                   activation="relu",\
                                   data_format="channels_last")(inputList[i])
            print(conv2DL1)
            maxPoolL1=MaxPooling2D(pool_size=pool_size,\
                                  padding="same",\
                                  data_format="channels_last")(conv2DL1)
            print(maxPoolL1)
            conv2DL2=Convolution2D(self.nb_filters,\
                                   kernel_size=(self.kernel_size[0],self.kernel_size[1]),\
                                   border_mode='valid',\
                                   activation="relu",\
                                   data_format="channels_last")(maxPoolL1)
            print(conv2DL2)
            maxPoolL2=MaxPooling2D(pool_size=pool_size,\
                                  padding="same",\
                                  data_format="channels_last")(conv2DL2)
            print(maxPoolL2)
            denseL1=Dense(units=16,activation="relu")(maxPoolL2)
            print(denseL1)
            flattenL=Flatten()(denseL1)
            print(flattenL)
            denseL2=Dense(units=64,activation="relu")(flattenL)
            print(denseL2)
            denseList.append(Reshape((64,1))(denseL2))
        concateL=Lambda(K.concatenate)(denseList)
        print(concateL)
        BLSTML=Bidirectional(LSTM(units=64,activation="relu"))(concateL)
        print(BLSTML)
        denseL3=Dense(units=self.vecLen,activation="relu",name="vector_dense")(BLSTML)
        denseL4=Dense(units=self.img_rows*self.img_cols*self.img_passes,activation="tanh")(denseL3)
        reshapeL=Reshape((self.img_rows,self.img_cols,self.img_passes),name="vector_name")(denseL4)
        print(reshapeL)
        
        model=Model(inputs=inputList,outputs=reshapeL)
        model.compile(optimizer="rmsprop",\
                      loss="mse")
        
        return model
        
    def buildCNNClassifierModel(self,\
                                batch_size = 50,\
                                nb_classes = 10,\
                                nb_epoch = 12,\
                                nb_filters = 32,\
                                img_rows=28,\
                                img_cols =28,\
                                pool_size = (2,2),\
                                kernel_size = (3,3)):
        '''
        batch_size(default: 50):batch size
        nb_classes(default: 10):the number of classes
        nb_epoch(default: 12):the number of epoch
        nb_filters(default: 32):the number of filters
        img_rows(default: 28 ):the height of matrixes
        img_cols(default: 28):the width of matrixes
        pool_size(default: (2,2)):the size of pools
        kernel_size(default: (3,3)): the size of kernels
        '''
        try:
            self.batch_size = self.paraDict['batch_size']
        except:
            self.batch_size=batch_size
            
        try:
            self.nb_classes = self.paraDict['nb_classes']
        except:
            self.nb_classes=nb_classes
            
        try:
            self.nb_epoch = self.paraDict['nb_epoch']
        except:
            self.nb_epoch=nb_epoch
            
        try:
            self.nb_filters = self.paraDict['nb_filters']
        except:
            self.nb_filters =nb_filters
            
        try:
            self.img_rows, self.img_cols = self.paraDict['img_rows'],self.paraDict['img_cols']
        except:
            self.img_rows, self.img_cols =img_rows, img_cols
            
        try:
            self.pool_size = self.paraDict['pool_size']
        except:
            self.pool_size = pool_size
            
        try:
            self.kernel_size = self.paraDict['kernel_size']
        except:
            self.kernel_size=kernel_size
            
        self.input_shape = (self.img_rows, self.img_cols,1)
            
        model = Sequential()
        
        model.add(Convolution2D(self.nb_filters, self.kernel_size[0] ,self.kernel_size[1],
                                border_mode='valid',
                                input_shape=self.input_shape))
        model.add(Activation('relu'))
        
        # 卷积层，激活函数是ReLu
        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1]))
        model.add(Activation('relu'))
        
        # 池化层，选用Maxpooling，给定pool_size，dropout比例为0.25
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))
        
        # Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
        model.add(Flatten())
        
        # 包含128个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        # 包含10个神经元的输出层，激活函数为Softmax
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
        
        model.summary()
        
        self.DPmodel=model
        
        return model
    
if __name__ == '__main__':
    print("loading data ...")
    with open("structuredData/seriesPkl.pkl","rb") as structuredDataFile:
        myData=pkl.load(structuredDataFile)
    x_train=[row[0] for row in myData]
    x_train=[[row[i] for row in x_train] for i in range(len(x_train[0]))]
#     print(len(x_train))
    y_train=[row[1] for row in myData]
#     print(y_train)
    print("building model ...")
    cnnGenModel=MyModel(img_rows=36,\
                        img_cols=36,\
                        kernel_size = (3,3)).buildCNNSeriesModel()
    cnnGenModel.fit(x_train,np.array(y_train),epochs=100,\
                    callbacks=[TensorBoard(log_dir="./log")])
    intermediate_layer_model = Model(inputs=cnnGenModel.input, 
                                 outputs=cnnGenModel.get_layer("dense_27").output)
    reStruMovieList=[intermediate_layer_model.predict([[col[i]] for col in x_train])[0] for i in range(len(x_train[0]))]
    
    ssList=[]
    clusterListList=[]
    clusterRange=range(2,19,2)
    for i in tqdm.tqdm(clusterRange):
        reStruMovieArr=np.array(reStruMovieList)
#         clusterModel=KMeans(n=i)
#         clusterList=clusterModel.fit_predict(reStruMovieArr).tolist()
        clusterList=kcluster(reStruMovieArr,nclusters=i,dist="u")[0].tolist()
        clusterListList.append(clusterList)
        clusterMat=-np.dot(reStruMovieArr,reStruMovieArr.T)/\
                    np.dot(np.sqrt(np.sum(reStruMovieArr*reStruMovieArr,axis=1)),\
                           np.sqrt(np.sum(reStruMovieArr*reStruMovieArr,axis=1)))
        ss=metrics.silhouette_score(clusterMat, clusterList, metric="precomputed")
        ssList.append(ss)
    minIndex=ssList.index(min(ssList))
    minClu=list(clusterRange)[minIndex]
    
    plt.plot(np.array(list(clusterRange)),np.array(ssList))
    plt.show()
    
    print(clusterListList[1])
#     clusterModel=KMeans(n_clusters=minClu)
#     clusterList=clusterModel.fit_predict(reStruMovieArr).tolist()
#     print(np.array(clusterList))
    