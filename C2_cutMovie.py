#coding:utf8
'''
Created on 2018年11月23日

@author: Administrator
'''

import pickle as pkl
import numpy as np
import cv2

def isCutPoint(cmpedArr):
    if cmpedArr[0]-cmpedArr[1]==0:
        return False
    else:
        return True

def main():
    
    print("loading data ...")
    with open("structuredData/zeroOneArr.pkl","rb") as zeroOneArrFile:
        zeroOneArr=pkl.load(zeroOneArrFile)
        
    print("get cut point in the frame of fraction of the movie ...")
    cutPointList=[i/zeroOneArr.shape[0] for i in range(zeroOneArr.shape[0]-1) if isCutPoint(zeroOneArr[i:i+2])==True]
    cutPointList=[0]+cutPointList+[1]
    
    print("cut point list:",cutPointList)    
    
    vc = cv2.VideoCapture('movie/antMan2.mp4')
    
    clipI=0
    i=0
    while 1:
        if cutPointList[clipI]==0:
            print("generaling the",clipI+1,"th clip ...")
            videoWriter = cv2.VideoWriter('movie/clip'+str(clipI)+'.mp4',\
                                          cv2.VideoWriter_fourcc('X','V','I','D'),\
                                          vc.get(cv2.CAP_PROP_FPS),\
                                          frameSize=(int(vc.get(3)),int(vc.get(4))))
            clipI+=1
        success,frame=vc.read()
        if not success:
            print("finished")
            break
        if i<int(cutPointList[clipI]*vc.get(7)):
            videoWriter.write(frame)
        else:
            videoWriter.write(frame)
            print("generaling the",clipI+1,"th clip ...")
            videoWriter = cv2.VideoWriter('movie/clip'+str(clipI)+'.mp4',\
                                          cv2.VideoWriter_fourcc('X','V','I','D'),\
                                          vc.get(cv2.CAP_PROP_FPS),\
                                          frameSize=(int(vc.get(3)),int(vc.get(4))))
            clipI+=1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        i+=1
    print("finished!")

if __name__ == '__main__':
    
    videoPath="movie/antMan2.mp4"
    clipPath="movie"
    
    print("loading data ...")
    with open("structuredData/zeroOneArr.pkl","rb") as zeroOneArrFile:
        zeroOneArr=pkl.load(zeroOneArrFile)
        
    print("get cut point in the frame of fraction of the movie ...")
    cutPointList=[i/zeroOneArr.shape[0] for i in range(zeroOneArr.shape[0]-1) if isCutPoint(zeroOneArr[i:i+2],i/zeroOneArr.shape[0]-1)==True]
    cutPointList=[0]+cutPointList+[1]
    
    print(cutPointList)
    
    vc = cv2.VideoCapture('movie/antMan2.mp4')
    
    clipI=0
    i=0
    while 1:
        if cutPointList[clipI]==0:
            print("generaling the",clipI+1,"th clip ...")
            videoWriter = cv2.VideoWriter('movie/clip'+str(clipI)+'.mp4',\
                                          cv2.VideoWriter_fourcc('X','V','I','D'),\
                                          vc.get(cv2.CAP_PROP_FPS),\
                                          frameSize=(int(vc.get(3)),int(vc.get(4))))
            clipI+=1
        success,frame=vc.read()
        if not success:
            print("finished")
            break
        if i<int(cutPointList[clipI]*vc.get(7)):
            videoWriter.write(frame)
        else:
            videoWriter.write(frame)
            print("generaling the",clipI+1,"th clip ...")
            videoWriter = vc.VideoWriter('movie/clip'+str(clipI)+'.mp4',\
                                          cv2.VideoWriter_fourcc('X','V','I','D'),\
                                          vc.get(cv2.CAP_PROP_FPS),\
                                          frameSize=(vc.get(3),vc.get(4)))
            clipI+=1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        i+=1
    print("finished!")