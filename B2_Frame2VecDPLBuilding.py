#coding:utf8
'''
Created on 2018年11月22日

@author: Administrator
'''
from tryDPModel import MyModel

import pickle as pkl
import numpy as np

import keras
from keras.models import Model
from keras.callbacks import TensorBoard

def main(imgWidth,\
         imgHeight,\
         kernel_size = (3,3),\
         validation_split=0.5,\
         epochs=250,\
         batch_size = 50,\
         nb_filters = 32,\
         img_num=9,\
         img_passes=3,\
         pool_size = 5,\
         vecLen=100):
    
    print("loading data ...")
    with open("structuredData/seriesPkl.pkl","rb") as structuredDataFile:
        myData=pkl.load(structuredDataFile)
    x_train=[row[0] for row in myData]
    x_train=[[row[i] for row in x_train] for i in range(len(x_train[0]))]
#     print(len(x_train))
    y_train=[row[1] for row in myData]
#     print(y_train)
    print("building model ...")
    cnnGenModel=MyModel(img_rows=imgHeight,\
                        img_cols=imgWidth,\
                        kernel_size = kernel_size,\
                        validation_split=validation_split,\
                        epochs=epochs,\
                        batch_size = batch_size,\
                        nb_filters = nb_filters,\
                        img_num=img_num,\
                        img_passes=img_passes,\
                        pool_size = pool_size,\
                        vecLen=vecLen).buildCNNSeriesModel()
                          
    print("training model ...")
    cnnGenModel.fit(x_train,np.array(y_train),\
                    validation_split=validation_split,\
                    epochs=epochs,\
                    batch_size=batch_size,\
                    callbacks=[TensorBoard(log_dir="./log")])
     
    print("saving training model ...")
#     cnnGenModel.save("models/DPModel.h5")
    cnnGenModel=keras.models.load_model("models/DPModel.h5")
    
    print("predicting model construction ...")
    intermediate_layer_model = Model(inputs=cnnGenModel.input, 
                                 outputs=cnnGenModel.get_layer("vector_dense").output)
    
    print("saving generation model ...")
    intermediate_layer_model.save("models/DPGenModel.model")
    
    print("predicting ...")
    reStruMovieList=[intermediate_layer_model.predict([[col[i]] for col in x_train])[0] for i in range(len(x_train[0]))]
    
    print("saving restructured data prepared for cluster ...")
    with open("structuredData/reStruMovieList.pkl","wb+") as reStruMovieListFile:
        pkl.dump(reStruMovieList,reStruMovieListFile)
        
    print("finished!")

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
                        
    print("training model ...")
    cnnGenModel.fit(x_train,np.array(y_train),\
                    validation_split=0.5,\
                    epochs=1,\
                    callbacks=[TensorBoard(log_dir="./log")])
    
    print("saving training model ...")
    cnnGenModel.save("models/DPModel.model")
    
    print("predicting model construction ...")
    intermediate_layer_model = Model(inputs=cnnGenModel.input, 
                                 outputs=cnnGenModel.get_layer("vector_dense").output)
    
    print("saving generation model ...")
    intermediate_layer_model.save("models/DPGenModel.model")
    
    print("predicting ...")
    reStruMovieList=[intermediate_layer_model.predict([[col[i]] for col in x_train])[0] for i in range(len(x_train[0]))]
    
    print("saving restructured data prepared for cluster ...")
    with open("structuredData/reStruMovieList.pkl","wb+") as reStruMovieListFile:
        pkl.dump(reStruMovieList,reStruMovieListFile)
        
    print("finished!")