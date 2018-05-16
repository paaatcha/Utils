# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This file contains auxiliary function for classification problems.

If you find some bug, please, e-mail me

'''

import numpy as np
import matplotlib.pyplot as plt
import itertools

# This function binarizes a vector
# Example:
# In: v = [1,2,3]
# Out: v = [1,0,0;
#                 0,1,0;
#                 0,0,1]
def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if ind is None:
        return None
    
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)


# This function sets 1 to the the maximum label and 0 for the other ones
# The input is a matrix of lables, one per row
# Example:
#         [0.1 0.3  0.9
# In:     0.3 0.01 0.2
#          0.9 0.8  0.1]
#
#         [0 0 1
# Out:  1 0 0
#          1 0 0]
def getMaxLabel (vin):
      [m,n] = vin.shape
      vout = np.zeros([m,n])

      mx = vin.max(axis=1)
      for i in xrange(m):
            for j in xrange(n):
                  vout[i,j] = int (mx[i] == vin[i,j])                  

      return vout


# This function counts the number of the miss classification by comparing the label matrix obtaineg by a classifier
# and the real label matrix. Important: the vreal and vclass must be in this order!
def cont_error (vreal, vclass):
      # Getting the matrix binarized
      vclass = getMaxLabel (vclass)
      [m,n] = vreal.shape
      dif = vreal - vclass
      err = 0

      for i in xrange(m):
            flag = 0
            for j in xrange(n):
                  if dif[i,j] != 0:
                        flag = 1

            if flag == 1:
                  err = err + 1

      return err
  
def contError (vreal, vclass):
      # Getting the matrix binarized
      vclass = getMaxLabel (vclass)
      [m,n] = vreal.shape
      #dif = vreal - vclass
      err = abs(vreal - vclass).sum()
      return int(err/2)
  
def contErrorInt (real, pred):
    nSamples = real.shape[0]
    err = (real == pred).sum()
    
    return (1-(float(nSamples-err)/nSamples))*100   
    

# This function computes the sigmoid
def sigmoid (v):
    return 1/(1+np.exp(-v))

# This function computes the RMSE of set of data organized in a matrix. Each line is a sample
# and each column is an attribute. Important: if the np.array is in the format (n,), 
# it must be reshaped to (1,n)
def rmse (x,y):
    return np.sqrt(np.power((x-y),2)).mean(axis=1).mean() 
    
def mse (x,y):
    return np.power((x-y),2).mean(axis=1).mean()
    
# This function normalizes the data with zero mean and standart deviation 1
def normZeroMean (data):
    return (data - data.mean())/data.std()
    

# this function shuffles the data respecting its labels values
def shuffleData (dataIn, dataOut):
    n1 = len(dataIn)
    n2 = len(dataOut)
    if n1 != n2:
        raise ('ERROR: the length of dataIn and dataOut must be equal')
    
    pos = np.random.permutation(n1)    
    newIn = list()
    newOut = list()
   
    for i in xrange(n1):
        newIn.append (dataIn[pos[i]])        
        newOut.append(dataOut[pos[i]])

    return newIn, newOut

# This function flats a list of matrices
def flatList (l):
    n = len(l)
    ret = list()
    for i in xrange(n):
        ret.append (l[i].flatten())
    return ret
    
# This function split the dataIn and dataOut to dataIn_train, dataOut_train, 
# dataIn_test and dataOut_test. The % of the train and test set is determined
# by pctTrain. Ex: If pctTrain = 0.7 => 70% for training and 30% for test.
# t is the output type: linear or binary
def splitTrainTest (dataIn, dataOut, pctTrain,t = 'linear'):
    if pctTrain > 1 or pctTrain < 0:
        raise ('ERROR: the pctTrain must be in the [0,1] interval')
    
    nsp = len(dataIn) # number of samples
    sli = int(round(nsp*pctTrain)) # getting pctTrain% to trainning   
    dataIn_train = dataIn[0:sli]
    dataIn_test = dataIn[sli:nsp]    
    if t == 'linear':
        dataOut_train = (dataOut[0:sli])
        dataOut_test = (dataOut[sli:nsp])
    elif t == 'binary':
        dataOut_train = ind2vec(dataOut[0:sli])
        dataOut_test = ind2vec(dataOut[sli:nsp])
    
    return dataIn_train, dataOut_train, dataIn_test, dataOut_test
    
def softmax (x):
    x = np.asarray(x)
    e = np.exp(x)
    return (e / np.sum(e))

# This function returns the confusion matrix
def confusionMatrix (real, net):
    net = getMaxLabel (net)
    cNet = 0 # The estimated class
    cReal = 0 # The real classes
    m,n = real.shape
    mat = np.zeros([n,n])
    
    print m, n    
    
    for i in xrange(m):
        for j in xrange(n):            
            if real[i,j] == 1:
                cReal = j
            if net[i,j] == 1:
                cNet = j
        mat[cReal,cNet] = mat[cReal,cNet] + 1            
            
    return mat

# This function computes the recall and precision given a confusion matrix
# For multiclass label, the result is the recall and precision mean
def classificationMetrics (confMat, verbose=True):
    m = confMat.shape[0]
    recall = 0.0
    precision = 0.0
    totalLabel = confMat.sum(axis=0)
    totalPredicted = confMat.sum(axis=1)
    
    for k in xrange(m):        
        recall += confMat[k,k]/totalLabel[k]
        precision += confMat[k,k]/totalPredicted[k]

    recall = recall/m
    precision = precision/m    
    
    if verbose==True:
        print 'Recall: ', recall, ' - Precision: ', precision
        
    
        
    return recall, precision       
    
    
# This funtion remount an grayscale image, flatted in an array, into an image format.
# Data is the image flatted and res is the real image's resolution. In this case, the img
# must be square, e.g, res x res
# Ex: let p(i,j) a pixel, an image would be:
# Img = [p(1,1) ... p(1,res)
#          .           .
#          .           .
#          .           .
#       p(res,1) ... p(1,res)]
#
# data = [p(1,1), p(1,2)...p(res,1)...p(res,res)]    
def remountImg (data, res):    
    newImg = np.zeros([res,res])        
    for i in xrange(res):
        for j in xrange(res):
            newImg[i,j] = data[i*res + j]
    
    return newImg


# This function writes a file using one or more arrays. There are two demands: the
# last parameter must be the file's name (w/ or w/o the whole path), and the second
# to last must be an array (or a simple list) naming all the previous arrays that 
# will be writed on the file.
# Ex: writeResults (array1, array2, array3, ['Data1', 'Data2', 'Data3'], 'fileName')
def saveResults (*args):
    nArrays = len(args)
    fileName = args[nArrays-1] # The file name must be the last parameter
    paramNames = args[nArrays-2]
    
    if nArrays-2 != len(paramNames):
        raise ('ERROR: the second to last parameter must be the name of all previous arrays')
    
    F = open (fileName+'.txt', 'w')
    
    for i in range(nArrays-2):
        F.write('The paramenter '+ paramNames[i] +':\n')
        F.write(str(args[i])+'\n')
        F.write('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##'+'\n\n')   
    
    F.close()
    
# This function plots the confusion matriz using the matplotlib    
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
  
    

    
    
    
    
    
