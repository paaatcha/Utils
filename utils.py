#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

This file contains some utils fuctions that I use in different situations

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find some bug, please email-me

"""

import numpy as np
import matplotlib
import itertools
from textwrap import wrap
import re
import tfplot
from sklearn.metrics import confusion_matrix


'''
    This function binarizes a vector (one hot enconding)
    For example:
        Input: v = [1,2,3]
        Output: v = [1,0,0;
                     0,1,0;
                     0,0,1]
    
        Input:
            ind: a array 1 x n
            N: the number of indices. If None, the code get is from the shape
        Output:
            The one hot enconding array n x N    
'''
def one_hot_encoding(ind, N=None):
    ind = np.asarray(ind)
    if ind is None:
        return None
    
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)


"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Input: 
        cm: the confusion matrix array
        classes: the list of labels' name
        normalize: set it true if you wann normalize the confusion matrix
        title: the title that will appear in the plot
        cmap: the color's pallet
"""
def plot_conf_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=matplotlib.pyplot.cm.Blues):

    plt = matplotlib.pyplot
    
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
    

''' 
    This function plots a confusion matrix in tensorboard. I get this from this post in stackoverflow:
    https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
    
    Input:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Output:
        summary: TensorFlow summary 

    Other itema to note:
    - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
    - Currently, some of the ticks dont line up due to rotations.
    
    Need to call in the main code  confusion matrix summaries:
        img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
        img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
        img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='dev/cm')
        img_d_summary_writer.add_summary(img_d_summary, current_step)
    
'''
def confusion_matrix_tf(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):

    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')
    
    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()
    
    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')
    
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]
    
    tick_marks = np.arange(len(classes))
    
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    
    return summary    


#cm = np.array([[56,1,2,0,0,2,0,0,5],
#[2,36,3,0,4,3,1,0,18],
#[0,0,70,0,0,0,0,0,0],
#[1,0,0,71,0,1,0,0,0],
#[0,1,0,0,59,0,0,0,2],
#[0,7,0,0,0,58,0,0,3],
#[2,0,0,0,0,0,59,0,1],
#[0,0,0,0,0,0,0,63,0],
#[1,9,3,0,2,2,0,0,72]])
#
#diag = ['Nevo', 'cera_act', 'Ceratoacantoma', 'corno', 'CEC', 'cera_seborreica', 'lentigo', 'melanoma', 'CBC']     
#
#plot_conf_matrix(cm, diag)       
            