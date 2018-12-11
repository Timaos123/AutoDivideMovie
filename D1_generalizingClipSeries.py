#coding:utf8
'''
Created on 2018年11月26日

@author: Administrator
'''

import pickle as pkl
import cv2
import moviepy
import win_unicode_console
win_unicode_console.enable()
from moviepy.video.io.VideoFileClip  import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

def isCutPoint(cmpedArr):
    if cmpedArr[0]-cmpedArr[1]==0:
        return False
    else:
        return True

def getVCItem(clipIndex,meanPreCutLen):
    vc = VideoFileClip("movie/clip"+str(clipIndex)+".mp4")
    vcLen=vc.duration
    startPoint=vcLen/2-meanPreCutLen
    endPont=vcLen/2
    vcItem = vc.subclip(int(startPoint), int(endPont))
    return vcItem
    
if __name__ == '__main__':
    
    totalTime=60#60 seconds
    cutClipList=[1,3,4,7,8,10,11,12,13]
    
    meanPreCutLen=totalTime/len(cutClipList)
    print("the average length of each clip:",meanPreCutLen)
    
    print("getting the preview's clips ...")
    preCutPointList=[getVCItem(cci,meanPreCutLen) for cci in cutClipList]
    
    print("combining ...")
    previewVideo=concatenate_videoclips(preCutPointList)
    
    print("saving")
    previewVideo.to_videofile("movie/preview.mp4", fps=24, remove_temp=True)