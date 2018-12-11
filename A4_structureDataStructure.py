#coding:utf8
'''
Created on 2018年11月8日

@author: Administrator
'''

import os
import cv2
import numpy as np
import pickle as pkl

    
windowSize=14
step=1

def paddingSeq(row,maxLen):
    neededShape=row[0].shape
    while len(row)<maxLen:
        row.append(np.zeros(neededShape))
    return row
    
def main(windowSize=windowSize,step=step):
    '''
    structure the image data into structure of:
     [[img1,img2,...,imgi-1,imgi+1,...imgn],[imgi]]
    ----------------------------------------
    imgPath: the path of the images to be saved
    windowSize: the flow window's size
    step: the flow window's step
    '''
    imgPath="imageDatas"
    fileNameList= [[int(fileName.split(".")[0]) for fileName in fileNameList]\
                    for _,_,fileNameList in os.walk(imgPath)]
    fileNameList=fileNameList[0]
    fileNameList=list(set(fileNameList))
    
    print("structuring ...")
    structuredDataList=[[[cv2.imread(os.path.join(imgPath,str(fileNameItem)+".jpg"))/255\
                     for fileNameItem in fileNameList[i:i+int(windowSize/2)]+fileNameList[i+int(windowSize/2)+1:windowSize]],\
                     cv2.imread(os.path.join(imgPath,str(fileNameList[i+int(windowSize/2)]))+".jpg")/255]\
      for i in range(0,len(fileNameList)-windowSize+1,step)]
    lenList=[len(row[0]) for row in structuredDataList]
    maxLen=max(lenList)
    structuredDataList=[[paddingSeq(row[0],maxLen),row[1]] for row in structuredDataList]
    
    print("saving data ...")
    if os.path.exists("structuredData")==False:
        os.mkdir("structuredData")
    with open("structuredData/seriesPkl.pkl","wb+") as seriesFile:
        pkl.dump(structuredDataList,seriesFile)
        
    print("---finished!---")
    print("-total structure:\n[[img[0],img[1],img[2],...,img[windowSize-1]],img[int(windowSize/2)]]")
    print("-total shape:",np.array(structuredDataList).shape)
    print("-single shape:",structuredDataList[0][0][0].shape)
    print("---------------")
    
    return structuredDataList

if __name__ == '__main__':
    main(windowSize=windowSize,step=step,jumpStep=True)
    
    