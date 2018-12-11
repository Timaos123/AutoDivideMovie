#coding:utf8
'''
Created on 2018年11月6日
reference:
https://www.jb51.net/article/135972.htm
@author: Administrator
'''
import cv2 
import os
import shutil
from PIL import Image
def main(videoPath,abstractMode="part",timeF=10,checkF=200,imgNum=-1):
    '''
    get data in image frame from the video
    ----------------------------------------
    videoPath: the path of the source video
    imgPath: the path of the images to be saved
    timeF: distinct between images
    checkF: check whether the programme is running every <checkF> frames
    imgNum: 
        if it is -1, then travel the whole video
        else get <imgNum> images and then stop
    
    '''
    imgPath="imageDatas"
    
    print("loading video ...")
    vc = cv2.VideoCapture(videoPath)
    
    print("checking whether is open ...")
    if vc.isOpened():
        rval , frame = vc.read() 
    else: 
        rval = False
    i=0    
    c=1
    
    print("developing images ...")
    if abstractMode=="part":
        print("distinct between images is",timeF)
        while rval:
            if i<0:
                continue
            rval, frame = vc.read() 
            if(c%timeF == 0):
                cv2.imwrite(imgPath+'/'+str(i) + '.jpg',frame)
                i=i+1
                if i%checkF==0:
                    print("the",i,"th photo")
            c = c + 1
            cv2.waitKey(1) 
            if imgNum!=-1:
                if i==imgNum:
                    break
        vc.release()
    elif abstractMode=="total":
        print("distinct between images is",int(vc.get(7)/imgNum))
        totalFrameNum=vc.get(7)
        timeF=int(totalFrameNum/imgNum)
        while rval:
            if i<0:
                continue
            rval, frame = vc.read() 
            if(c%timeF == 0):
                cv2.imwrite(imgPath+'/'+str(i) + '.jpg',frame)
                i=i+1
                if i%checkF==0:
                    print("the",i,"th photo")
            c = c + 1
            cv2.waitKey(1) 
            if imgNum!=-1:
                if i==imgNum:
                    break
        vc.release()
    else:
        raise NameError
        print("the divide mode's name is wrong")
    print("finished")
    
if __name__ == '__main__':
    
    main("movie/antMan2.mp4",abstractMode="part",timeF=77,checkF=200,imgNum=10)
#     
#     picNum=500
#     
#     print("clearing imageDatas ...")
#     if len(list(os.walk("imageDatas")))>0:
#         shutil.rmtree("imageDatas")
#         os.mkdir("imageDatas")
#     
#     timeF = 10
#     print("distinct between images is",timeF)
#     
#     print("loading video ...")
#     vc = cv2.VideoCapture('movie/antMan.mp4')
#     
#     print("checking whether is open ...")
#     if vc.isOpened(): #判断是否正常打开 
#         rval , frame = vc.read() 
#     else: 
#         rval = False
#       
#     
#     i=0    
#     c=1
#     while rval:
#         if i<0:
#             continue
#         rval, frame = vc.read() 
#         if(c%timeF == 0):
#             cv2.imwrite('imageDatas/'+str(i) + '.jpg',frame)
#             i=i+1
#             if i%100==0:
#                 print("the",i,"th photo")
#         c = c + 1
#         cv2.waitKey(1) 
#         if picNum!=-1:
#             if i==picNum:
#                 break
#     vc.release()
#     print("finished")