#coding:utf8
'''
Created on 2018年11月22日

@author: Administrator
'''
import pickle as pkl
import tqdm
import numpy as np
from Bio.Cluster import kcluster 
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image as Img

def main(clusterStarter=2,\
         clusterEnder=9,\
         clusterStep=2):
    
    ssList=[]
    clusterListList=[]
    
    print("loading data ...")
    with open("structuredData/reStruMovieList.pkl","rb") as reStruMovieListFile:
        reStruMovieList=pkl.load(reStruMovieListFile)
    clusterRange=range(clusterStarter,clusterEnder,clusterStep)
    for i in tqdm.tqdm(clusterRange):
        reStruMovieArr=np.array(reStruMovieList)
#         clusterModel=KMeans(n=i)
#         clusterList=clusterModel.fit_predict(reStruMovieArr).tolist()
        clusterList=kcluster(reStruMovieArr,nclusters=i,dist="u",npass=150)[0].tolist()
        clusterListList.append(clusterList)
        clusterMat=-np.dot(reStruMovieArr,reStruMovieArr.T)/\
                    np.dot(np.sqrt(np.sum(reStruMovieArr*reStruMovieArr,axis=1)),\
                           np.sqrt(np.sum(reStruMovieArr*reStruMovieArr,axis=1)))
        ss=metrics.silhouette_score(clusterMat, clusterList, metric="precomputed")
        ssList.append(ss)
    maxIndex=ssList.index(max(ssList))
    maxClu=list(clusterRange)[maxIndex]
    
    print("the best cluster number is",maxClu)
    
    print("developing figures...")
    plt.plot(np.array(list(clusterRange)),np.array(ssList))
    clusterListArr=np.array(clusterListList)
    clusterListArrMaxNum=np.max(clusterListArr)
    clusterListArr=clusterListArr/clusterListArrMaxNum*255
    clusterListImg=Img.fromarray(clusterListArr)
    clusterListImg=clusterListImg.resize((500,1000))
    
    print("saving figure ...")
    plt.savefig("figures/clustering_figure.jpg")
    plt.close()
    clusterListImg=clusterListImg.convert('RGB')
    clusterListImg.save("figures/clusterList_figure.png")
    
    print("clustering ...")
    bestClusterList=kcluster(reStruMovieArr,nclusters=maxClu)[0].tolist()
    
    print("saving data ...")
    with open("structuredData/bestClusterList.pkl","wb+") as bestClusterListFile:
        pkl.dump(bestClusterList,bestClusterListFile)
    
    print("finished!")
    
if __name__ == '__main__':
    
    ssList=[]
    clusterListList=[]
    clusterStarter=2
    clusterEnder=9
    clusterStep=2
    
    print("loading data ...")
    with open("structuredData/reStruMovieList.pkl","rb") as reStruMovieListFile:
        reStruMovieList=pkl.load(reStruMovieListFile)
    
    clusterRange=range(clusterStarter,clusterEnder,clusterStep)
    for i in tqdm.tqdm(clusterRange):
        reStruMovieArr=np.array(reStruMovieList)
#         clusterModel=KMeans(n=i)
#         clusterList=clusterModel.fit_predict(reStruMovieArr).tolist()
        clusterList=kcluster(reStruMovieArr,nclusters=i,dist="u",npass=150)[0].tolist()
        clusterListList.append(clusterList)
        clusterMat=-np.dot(reStruMovieArr,reStruMovieArr.T)/\
                    np.dot(np.sqrt(np.sum(reStruMovieArr*reStruMovieArr,axis=1)),\
                           np.sqrt(np.sum(reStruMovieArr*reStruMovieArr,axis=1)))
        ss=metrics.silhouette_score(clusterMat, clusterList, metric="precomputed")
        ssList.append(ss)
    maxIndex=ssList.index(max(ssList))
    maxClu=list(clusterRange)[maxIndex]
    
    print("developing figures...")
    plt.plot(np.array(list(clusterRange)),np.array(ssList))
    clusterListArr=np.array(clusterListList)
    clusterListArrMaxNum=np.max(clusterListArr)
    clusterListArr=clusterListArr/clusterListArrMaxNum*255
    clusterListImg=Img.fromarray(clusterListArr)
    clusterListImg=clusterListImg.resize((500,1000))
    
    print("saving figure ...")
    plt.savefig("figures/clustering_figure.jpg")
    clusterListImg=clusterListImg.convert('RGB')
    clusterListImg.save("figures/clusterList_figure.png")
    plt.close()
    
    print("clustering ...")
    bestClusterList=kcluster(reStruMovieArr,nclusters=maxClu)[0].tolist()
    
    print("saving data ...")
    with open("structuredData/bestClusterList.pkl","wb+") as bestClusterListFile:
        pkl.dump(bestClusterList,bestClusterListFile)
    
    print("finished!")
