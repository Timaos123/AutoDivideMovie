#coding:utf8
'''
Created on 2018年11月23日

@author: Administrator
'''

import os
import shutil

def main():
    '''
    configurate the programme, mainly build and clear data
    '''
    dirList=["figures","imageDatas","log","models","movie","structuredData"]
    
    print("making up dirs ...")
    for dirItem in dirList:
        if dirItem not in list(os.listdir(".")):
            os.mkdir(dirItem)
    
    print("clearing dirs ...")
    clearList=["imageDatas","log"]
    for dirItem in clearList:
        if len(list(os.walk(dirItem)))>0:
            shutil.rmtree(dirItem)
            os.mkdir(dirItem)
    
if __name__ == '__main__':
    main()