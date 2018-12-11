#coding:utf8
'''
Created on 2018年12月1日

@author: Administrator
'''

import keras
import pickle as pkl
from keras.models import Model
from PIL import Image

if __name__ == '__main__':
    print("loading model ...")
    cnnGenModel=keras.models.load_model("models/DPModel.h5")
    
    print("predicting model construction ...")
    generalizeModel = Model(inputs=cnnGenModel.input, 
                                 outputs=cnnGenModel.get_layer("vector_name").output)
    
    print("loading data ...")
    with open("structuredData/seriesPkl.pkl","rb") as structuredDataFile:
        myData=pkl.load(structuredDataFile)
        
    print("generalizing validation data ...")
    X=myData[0][0]
    y=myData[0][1]
    
    print("generalizing frame ...")
    yPre=generalizeModel.predict(X)*255
    img=Image.fromarray(yPre.astype('uint8')).convert('RGB')
    img.imsave("structuredData/try_gen_fame.jpg")
    
    print("finished!")