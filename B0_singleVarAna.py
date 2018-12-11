#coding:utf8
'''
Created on 2018年11月19日

@author: Administrator
'''
import pickle as pkl
import numpy as np

def getSingleWindowVar(imgList):
    '''
    get a window's variance
    imgList:a window of images
    '''
    print("-getting the window's variance...")
    windowVar=np.mean([np.var([imgItem[:,:,i] for imgItem in imgList])\
                        for i in range(3)])
    return windowVar
    
def main():
    print("loading data ...")
    with open("structuredData/seriesPkl.pkl","rb") as imgSeqDataFile:
        imgSeqList=pkl.load(imgSeqDataFile)
        
    print("calculating the average variance...")
    meanVar=np.mean([getSingleWindowVar(windowItem[0]) for windowItem in imgSeqList])
    
    print("---finished!---")
    print("the average variance is:",meanVar)
    print("---------------")
    return meanVar
    
if __name__ == '__main__':
    
    print("loading data ...")
    with open("structuredData/seriesPkl.pkl","rb") as imgSeqDataFile:
        imgSeqList=pkl.load(imgSeqDataFile)
        
    print("calculating the average variance...")
    meanVar=np.mean([getSingleWindowVar(windowItem[0]) for windowItem in imgSeqList])
    
    print("---finished!---")
    print("the average variance is:",meanVar)
    