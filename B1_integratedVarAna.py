#coding:utf8
'''
Created on 2018年11月19日

@author: Administrator
'''
import A4_structureDataStructure as A4
import B0_singleVarAna as B0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tqdm

def main(startPoint,endPoint,varSearchStep):
    '''
    search the best window size to define the variance
    startPoint: the start point to expand the size of the window
    endPoint: the end point of the expansion of the size of the window
    varSearchStep: the change step of expansion of the size of the window
    '''
    try:
        with open("structuredData/varListFile.pkl","rb") as varListFile:
            windowVarList=pkl.load(varListFile)
    except:
        windowVarList=[]
        
    print("calculating variances...")
    for windowSizeI in tqdm.tqdm(range(startPoint,endPoint,varSearchStep)):
        A4.main(windowSize=windowSizeI,step=windowSizeI)
        #A4 changed the file which is needed in B0
        windowVarList.append((windowSizeI,B0.main()))
    windowVarList=np.array(windowVarList)
    print("the window variances are:",windowVarList)
    
    print("saving variance list ...")
    with open("structuredData/varListFile.pkl","wb") as varListFile:
        pkl.dump(windowVarList.tolist(),varListFile)
    
    print("saving var figure ...")
    plt.plot(windowVarList[:,0],windowVarList[:,1])
    plt.savefig("figures/varSeries_Figure.jpg")
    plt.close()
    
    varList=[varI[1] for varI in windowVarList]
    minVar=min(varList)
    for row in windowVarList:
        if row[1]==minVar:
            bestSize=row[0]
    
    print("the best window's size is",int(bestSize))
    
    return int(bestSize)
    
    
if __name__ == '__main__':
    try:
        with open("structuredData/varListFile.pkl","rb") as varListFile:
            windowVarList=pkl.load(varListFile)
    except:
        windowVarList=[]
        
    print("calculating variances...")
    for windowSizeI in tqdm.tqdm(range(20,40,1)):
        A4.main(windowSize=windowSizeI,step=windowSizeI)
        windowVarList.append((windowSizeI,B0.main()))
    windowVarList=np.array(windowVarList)
    print("the window variances are:",windowVarList)
    
    print("saving variance list ...")
    with open("structuredData/varListFile.pkl","wb") as varListFile:
        pkl.dump(windowVarList.tolist(),varListFile)
    
    plt.plot(windowVarList[:,0],windowVarList[:,1])
    plt.show()