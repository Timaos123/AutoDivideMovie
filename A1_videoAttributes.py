#coding:utf8
'''
Created on 2018年116

@author: Administrator
'''
import cv2 

def main(videoPath='movie/antMan2.mp4'):
    '''
    description of the movie
    '''
    vc = cv2.VideoCapture(videoPath) 
    print("total numer of frames:",vc.get(7))
    print("frame rate:",vc.get(5))
    print("total time:",vc.get(7)/vc.get(5)/3600,"h")
    print("width",vc.get(3),"px")
    print("height:",vc.get(4),"px")

if __name__ == '__main__':
    vc = cv2.VideoCapture('movie/antMan2.mp4') 
    print("total numer of frames:",vc.get(7))
    print("frame rate:",vc.get(5))
    print("total time:",vc.get(7)/vc.get(5)/3600,"h")
    print("width",vc.get(3),"px")
    print("height:",vc.get(4),"px")