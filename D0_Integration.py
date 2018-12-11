#coding:utf8
'''
Created on 2018年11月23日

@author: Administrator
'''

import os
import shutil
import A0_configuration
import A1_videoAttributes
import A2_getPicData
import A3_zipPicData
import A4_structureDataStructure
import B1_integratedVarAna
import B2_Frame2VecDPLBuilding
import C1_getClusDiv
import C2_cutMovie
import C0_clusterModelBuilding


if __name__ == '__main__':
    videoPath='movie/antMan2.mp4'
    abstractMode="total"
    timeF=4
    checkF=200
    imgNum=2200
    resizedWidth=32
    resizedHeight=32
    cutSubtitle=True
    structureStep=1
    varWindowStartPoint=5
    varWindowEndPoint=20
    varWindowStep=2
    
    kernel_size = (3,3)
    validation_split=0.5
    epochs=50
    batch_size = 32
    nb_filters = 32
    img_num=9
    img_passes=3
    pool_size = (5,5)
    vecLen=150
    
    clusterVarWindowsSize=50
    clusterVarDivideMethod="mean"
    clusterVarShowFig=True
    
    clusterStarter=3
    clusterEnder=12
    clusterStep=2
    thresholdAdj=+1
    
    clearVarList=True
    if clearVarList==True:
        if "varListFile.pkl" in os.listdir("structuredData"):
            os.remove("structuredData/varListFile.pkl")
    
    bestWindowSizeByVarReOne=17
    A0_configuration.main()
    A1_videoAttributes.main(videoPath)
    A2_getPicData.main(videoPath,abstractMode, timeF, checkF, imgNum)
    A3_zipPicData.main(resizedWidth,resizedHeight,cutSubtitle)
    bestWindowSizeByVarReOne=B1_integratedVarAna.main(varWindowStartPoint,varWindowEndPoint,varWindowStep)
    A4_structureDataStructure.main(bestWindowSizeByVarReOne+1,structureStep)
    B2_Frame2VecDPLBuilding.main(resizedWidth,\
                                 resizedHeight,\
                                 kernel_size = kernel_size,\
                                 validation_split=validation_split,\
                                 epochs=epochs,\
                                 batch_size = batch_size,\
                                 nb_filters = nb_filters,\
                                 img_num=bestWindowSizeByVarReOne,\
                                 img_passes=img_passes,\
                                 pool_size = pool_size,\
                                 vecLen=vecLen)
    C0_clusterModelBuilding.main(clusterStarter,\
                                 clusterEnder,\
                                 clusterStep)
    C1_getClusDiv.main(varSize=clusterVarWindowsSize,\
                       divideMethod=clusterVarDivideMethod,\
                       thresholdAdj=thresholdAdj,\
                       showFig=clusterVarShowFig)
    C2_cutMovie.main()
