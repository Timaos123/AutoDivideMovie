#coding:utf8
'''
Created on 2018年11月7日

@author: Administrator
'''

from PIL import Image
import os
import tqdm

def main(resizedWidth=36,resizedHeight=36,cutSubtitle=True):
    '''
    zip the image data
    ----------------------------------------
    imgPath: the path of the images to be saved
    resizedWidth: the width that the images are going to be zipped
    resizedHeight: the height that the images are going to be ziped
    cutSubtitle:whether to cut subtitle (1/8 down part of the images)
    '''
    imgPath="imageDatas"
    i=0
    for _, _, files in os.walk(imgPath):
        for f in tqdm.tqdm(files):
            if i<0:
                continue
            fp = os.path.join(imgPath,f)
            img = Image.open(fp)
            
            if cutSubtitle==True:
                w, h = img.size
                img=img.crop((0,0,w,7/8*h))
            else:
                print("not cut subtitles")
                pass
            
            img.resize((resizedWidth, resizedHeight)).save(fp, "JPEG")
            img.close
            i=i+1

if __name__ == '__main__':
    i=0
    resizedWidth=36
    resizedHeight=36
    for _, _, files in os.walk("imageDatas"):
        for f in tqdm.tqdm(files):
            if i<0:
                continue
            fp = os.path.join("imageDatas",f)
            img = Image.open(fp)
            
            print("cutting subtitles ...")
            w, h = img.size
            img=img.crop((0,0,w,7/8*h))
            
            print("resizing ...")
            img.resize((resizedWidth, resizedHeight)).save(fp, "JPEG")
            img.close
            i=i+1
