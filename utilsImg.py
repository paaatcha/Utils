# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This file contains auxiliary functions for problems involving CNNs and image classification

'''

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import itertools
from sklearn.utils import shuffle
import glob
import cv2
import os

def plotImg (img):
    plt.imshow(img)
    plt.show()
    

def loadImgInBatches (path, folders, posLabelPath=7, batchSize=50, norm=True, categ=True, verbose=False, resize=None):
    
    paths = loadPathFromFolders (path, folders, True, True)
    N = len(paths)
    
    while True:
        batchStart = 0
        batchEnd = batchSize
        
        for batchStart in xrange(0, N, batchSize):
            batchEnd = batchStart + batchSize
            
            if batchEnd > N:
                break
                
            XBatch, YBatch = loadImgsFromPaths (paths[batchStart:batchEnd], folders, posLabelPath, norm, categ, verbose, resize)
            yield (XBatch,YBatch)

     
def loadImgsFromPaths (paths, folders, norm=True, categ=True, verbose=False, resize=None, formatInput=False):
    
    folders.sort()
    imgList = list()
    labels = list()
    mapFoldNames = dict()
    
    k = 0
    for n in folders:
        mapFoldNames[n] = k
        k+=1
    
    N = len(paths)
    n = 1
    
    for pathImg in paths: 
        if verbose:
            print 'Loading ' + pathImg + '... [', n, ' of ', N, ']'
            n+=1
        
        img = cv2.imread(pathImg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if resize is not None:
            img = cv2.resize(img,resize,interpolation=cv2.INTER_LINEAR)            
                  
        if norm:
            imgFloat = img/255.0              
            imgList.append(imgFloat)                       
        else:
            imgList.append(img)            
        
        lb = pathImg.split('/')[-2]         
        labels.append(mapFoldNames[lb])        
              
        
    if categ:
        labels = formatOutputData(labels, len(folders))    
    
    return np.array(imgList), labels        
    
    
def loadImgsFromFolders (path, nSam=None, norm=True, categ=True, verbose=False, shuffle=True, resize=None, imgFormat='jpg'):           
    paths,foldNames = loadPathFromFolders (path, imgFormat, shuffle)
    
    if (nSam is None):
        return loadImgsFromPaths (paths, foldNames, norm, categ, verbose, resize)
    else:
        return loadImgsFromPaths (paths[0:nSam], foldNames, norm, categ, verbose, resize)

    
def loadPathFromFolders (path, imgFormat='jpg', shuf=True):
    paths = list()      
    foldNames = [nf for nf in os.listdir(path) if os.path.isdir(os.path.join(path, nf))]  
    
    if (len(foldNames) == 0):
        folders = glob.glob(path)
        foldNames = [path.split('/')[-1]]
    else:
        folders = glob.glob(path + '/*')
    
    for fold in folders:                   
        paths += (glob.glob(fold+'/*.'+imgFormat))
    
    if shuf:
        paths = shuffle(paths)    
    
    return paths, foldNames    


def getAugmentation (path, savePath, nSamAug, imgSize=(256,256), imgFormat ='jpg', saveFormat='jpg', savePrefix='aug_'):    
    
    imgs, lab = loadImgsFromFolders (path, norm=False, shuffle=True, imgFormat=imgFormat, resize=imgSize)    
    nImgs = len(imgs)
    print 'Number of images: ', nImgs
    print 'Number of augmented images: ', nImgs * nSamAug + nImgs   
    savePath = savePath + '/'
    
    ans = raw_input('Is this number ok? (Y,N): ')     
    if (ans == 'Y' or ans == 'y'):
        # Defining the data augmentation operations
        datagen = ImageDataGenerator (horizontal_flip=True,
                                      vertical_flip=True,                                      
                                      zoom_range=0.05,
                                      shear_range=0.05,
                                      height_shift_range=0.05,
                                      width_shift_range=0.05,
                                      rotation_range=0.2,
                                      fill_mode='nearest')
        
        data = datagen.flow(imgs, batch_size=nImgs, save_to_dir=savePath, save_format=saveFormat, save_prefix=savePrefix)        
        n = 0
        for batch in data:
            n+=1            
            if (n > nSamAug):
                break
            print 'Working on augmentation ', n, ' of ', nSamAug



def createDirs (path, folders=['A', 'B'], trainTest=False, doubleFoldLab=False):

    if (not os.path.isdir(path)):
        os.mkdir(path)
        
        if (trainTest):
            os.mkdir(path + '/' + 'TEST')
            os.mkdir(path + '/' + 'TRAIN')                
            os.mkdir(path + '/' + 'VAL')
            
        for folder in folders:            
            if (trainTest):
                if (doubleFoldLab):
                    os.mkdir(path + '/TRAIN/' + folder)
                    os.mkdir(path + '/TRAIN/' + folder + '/' + folder)
                else:
                    os.mkdir(path + '/TRAIN/' + folder)                                        
                    
                os.mkdir(path + '/TEST/' + folder)
                os.mkdir(path + '/VAL/' + folder)
            else:
                if (doubleFoldLab):
                    os.mkdir(path + '/' + folder)
                    os.mkdir(path + '/' + folder + '/' + folder)
                else:
                    os.mkdir(path + '/' + folder)

def splitFoldersTrainTestVal (pathIn, pathOut, tr=0.8, te=0.1, tv=0.1, verbose=False):
        
    if (tr+te+tv != 1.0):
        print ('tr, te and tv must sum up 1.0')
        raise ValueError
        
    folders = [nf for nf in os.listdir(pathIn) if os.path.isdir(os.path.join(pathIn, nf))]
    
#    print folders    
    
    for lab in folders:        
        pathImgs = glob.glob(pathIn + '/' + lab + '/*')        
        N = len(pathImgs)
        nTest = int(round(te*N))
        nVal = int(round(tv*N))
        nTrain = N - nTest - nVal        
        
        if (verbose):
            print 'Working on ', lab
            print 'Total: ', N, ' | Train: ', nTrain, ' | Test: ', nTest, ' | Val: ', nVal, '\n'
        
        pathTest = pathImgs[0:nTest]
        pathVal = pathImgs[nTest:(nTest+nVal)]
        pathTrain = pathImgs[(nTest+nVal):(nTest+nVal+nTrain)]
        
        for p in pathTest:
            os.system('cp ' + p + ' ' + pathOut + '/TEST/' + lab ) 
            
        for p in pathTrain:
            os.system('cp ' + p + ' ' + pathOut + '/TRAIN/' + lab)
            
        for p in pathVal:
            os.system('cp ' + p + ' ' + pathOut + '/VAL/' + lab )
            

def changeColorSpace (path, colorspace, savePath=None, trainTest=False, formatImg='jpg'):
    
    if (trainTest):
        L = ['TRAIN', 'TEST', 'VAL']
        folders = [nf for nf in os.listdir(path+'/TRAIN') if os.path.isdir(os.path.join(path+'/TRAIN', nf))]
    else:
        L = [colorspace]
        folders = [nf for nf in os.listdir(path) if os.path.isdir(os.path.join(path, nf))]
        
    if (savePath is None):
        savePath = '/'.join(path.split('/')[0:-1])        
        
    createDirs (savePath + '/' + colorspace, folders, trainTest)    
    
    for typ in L:        
        for fold in folders: 
            if (len(L) != 1):
                paths = glob.glob(path + '/' + typ + '/' + fold + '/*')                
            else:
                paths = glob.glob(path + '/' + fold + '/*')            
            
            print 'Working on ' + typ + ' - ' + fold
            
            for p in paths:   
                
                nameImg = p.split('/')[-1].split('.')[0]
                if (len(L) != 1):
                    savePathImg = savePath + '/' + colorspace + '/' + typ + '/' + fold + '/' + nameImg + '.' + formatImg 
                else:
                    savePathImg = savePath + '/'+ colorspace + '/' + fold + '/' + nameImg + '.' + formatImg 
                    
                img = cv2.imread(p)

                if (colorspace == 'HSV'):
                    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)           
                    cv2.imwrite (savePathImg, hsvImg)
                elif (colorspace == 'Lab'):
                    labImg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                    cv2.imwrite (savePathImg, labImg)            
                elif (colorspace == 'XYZ'):
                    xyzImg = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
                    cv2.imwrite (savePathImg, xyzImg)
                elif (colorspace == 'HLS'):
                    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                    cv2.imwrite (savePathImg, hlsImg)
                elif (colorspace == 'YUV'):
                    yuvImg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    cv2.imwrite (savePathImg, yuvImg)
                else:
                    print 'There is no ', colorspace, ' implemented in this code'
                    raise ValueError
                    

def formatInputData (x,rows,cols,depth=1):
    x = x.reshape(x.shape[0], rows, cols, depth)   
    x = x.astype('float32')    
    return x

def formatOutputData (y, nClass):
    return to_categorical (y,nClass)

def plotHistory (history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()

def splitData (X, Y, N, tr=0.80, te=0.10, val=0.10):
    
    if tr+te+val != 1.0:
        print 'ERROR in the splitdata values'
        raise ValueError
        
    ntr = int(round(tr*N))
    nval = int(round(te*N))
    nte = N-ntr-nval
    
    inTrain = X[0:ntr]
    outTrain = Y[0:ntr]    

    
    inVal = X[ntr:(ntr+nval)]
    outVal = Y[ntr:ntr+nval]
    
    inTest = X[ntr+nval:ntr+nval+nte]
    outTest = Y[ntr+nval:ntr+nval+nte]
        
    return inTrain, outTrain, inVal, outVal, inTest, outTest

def splitPaths (paths, tr=0.80, te=0.10, val=0.10):
    
    if tr+te+val != 1.0:
        print 'ERROR in the splitdata values'
        raise ValueError
    
    N = len(paths)
    ntr = int(round(tr*N))
    nval = int(round(te*N))
    nte = N-ntr-nval
    
    return paths[0:ntr], paths[ntr:ntr+nval], paths[ntr+nval:ntr+nval+nte]
    

def plotConfMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()