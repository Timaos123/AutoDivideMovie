#coding:utf8
'''
Created on 2018年11月23日

@author: Administrator
'''
import pickle as pkl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def isCutPoint(cmpedArr):
    if cmpedArr[0]-cmpedArr[1]==0:
        return False
    else:
        return True
    
def main(varSize=50,divideMethod="mean",thresholdAdj=0.1,showFig=False):
    
    windowStep=1
    
    print("loading data ...")
    with open("structuredData/bestClusterList.pkl","rb") as bestClusterListFile:
        bestClusterArr=np.array(pkl.load(bestClusterListFile))
    i=0
    varList=[]
    
    print("moving var ...")
    while i+varSize<bestClusterArr.shape[0]:
        varList.append(np.var(bestClusterArr[i:i+varSize]))
        i+=windowStep
    varArr=np.array(varList)
    
    print("moving filter ...")
    print("divide method:",divideMethod)
    if divideMethod=="mean":
        threshold=np.mean(varArr)
    elif divideMethod=="median":
        threshold=np.median(varArr)
    elif divideMethod=="hptp":
        threshold=np.ptp(varArr)/2+np.min(varArr)
    else:
        raise("err in divide method")
    threshold=threshold+thresholdAdj
    zeroOneVarArr=np.array([int(varItem>threshold) for varItem in varArr.tolist()])\
                    *np.max(varArr)
    
    print("get cut point in the frame of fraction of the movie ...")
    cutPointList=[i/zeroOneVarArr.shape[0] for i in range(zeroOneVarArr.shape[0]-1) if isCutPoint(zeroOneVarArr[i:i+2])==True]
    cutPointList=[0]+cutPointList+[1]
     
     
    clipLenMean=np.mean([cutPointList[lenI+1]-cutPointList[lenI]\
                  for lenI in range(len(cutPointList)-1)])
    clipLenStd=np.sqrt(np.var([cutPointList[lenI+1]-cutPointList[lenI]\
                  for lenI in range(len(cutPointList)-1)]))
     
    print("the mean length is",clipLenMean)
    print("the var of length is",clipLenStd)
    
    tempCutPointList=cutPointList.copy()
    for lenI in range(len(cutPointList)-1):
        if cutPointList[lenI+1]-cutPointList[lenI]<clipLenMean-clipLenStd:
            tempCutPointList.remove(cutPointList[lenI])
  
    cutPointList=tempCutPointList
    
    zeroOneList=zeroOneVarArr.tolist()
    
    flag=np.max(varArr)
    for cutI in range(len(cutPointList)-1):
        for i in range(int(cutPointList[cutI]*zeroOneVarArr.shape[0]),\
                           int(cutPointList[cutI+1]*zeroOneVarArr.shape[0])):
            zeroOneList[i]=flag
        if flag==np.max(varArr):flag=0
        else:flag=np.max(varArr)
    zeroOneVarArr=np.array(zeroOneList)
    
    print("developing figures ...")
    zeroOneVarPlot,=plt.plot(zeroOneVarArr)
    varPlot,=plt.plot(varArr)
    thresholdPlot,=plt.plot(np.array([i for i in range(varArr.shape[0])]),[threshold for i in range(varArr.shape[0])])
    plt.legend([zeroOneVarPlot,varPlot,thresholdPlot],\
               ["clips","variances",divideMethod+" threshold"],\
               loc="upper right")
    
    print("saving data and figures ...")
    plt.savefig("figures/var_figure.jpg")
    plt.close()
    with open("structuredData/zeroOneArr.pkl","wb+") as zeroOneArrFile:
        pkl.dump(zeroOneVarArr,zeroOneArrFile)
    if showFig==True:
        plt.show()
    print("finished!")
            
if __name__ == '__main__':
    main(showFig=True)