#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns

# allData: a list with all datas to plot
# nPlots: the number of plots on the subplot
# posSubPlot: the axes1 values for the subplot
# names: a list with all algorithms' name, e.g., ['alg1', 'alg2']
# xLabel and yLabel: the labels for the axes
# saveName: if you wanna save the figure, just set the name or the whole path
# figSize: the figure's size
def boxplot (allData, nPlots=1, posSubPlot=[1,1], names=None, xLabel=None, yLabel=None, saveName=None, figSize=[9,4]):
    
    # Getting the number of plots and boxes in each plot
    if nPlots > 1:
        nBoxes = np.asarray(allData[0]).shape[0]

    else:
        nBoxes = np.asarray(allData).shape[0]    
     
    fig, axes = plt.subplots(nrows=posSubPlot[0], ncols=posSubPlot[1], figsize=(figSize[0],figSize[1]))     
    
    if nPlots > 1:
        bplot = list()
        for i in range(nPlots):
            bplot.append (axes[i].boxplot (allData[i], vert=True))
    else:
        bplot = axes.boxplot (allData, vert=True)
    
    # Adding the horizontal grid lines and the x/y labels
    if nPlots > 1:
        for ax in axes:
            ax.yaxis.grid(True)#, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        if xLabel is not None and yLabel is not None:
            axes[int(ceil(nPlots/2.0))-1].set_xlabel(xLabel)
            axes[0].set_ylabel(yLabel)            
    else:
        axes.yaxis.grid(True)
        #axes.yaxis.grid(True)#, linestyle='-', which='major', color='lightgrey', alpha=0.5)        
        #axes.set_axis_bgcolor('white')
        
        
        
        if xLabel is not None and yLabel is not None:
            axes.set_xlabel(xLabel)
            axes.set_ylabel(yLabel)
    
    # Adding the algorithms' name on the figure
    if names is not None:
        plt.setp(axes, xticks=[y+1 for y in range (nBoxes)], xticklabels=names)
        
    # Saving the figure
    if saveName is not None:
        fig.savefig(saveName+'.png', bbox_inches='tight')
        
    
    plt.show()
    


def boxplotSNS (allData, nPlots=1, posSubPlot=[1,1], names=None, xLabel=None, yLabel=None, saveName=None, figSize=[9,4], pal=None):
    # Getting the number of plots and boxes in each plot
    if nPlots > 1:
        nBoxes = np.asarray(allData[0]).shape[0]

    else:
        nBoxes = np.asarray(allData).shape[0]        
    
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    if pal is None:
        pal = 'BuGn_d'
    
    ax = sns.boxplot(data=allData, palette=pal, linewidth=1.8)
        
    if xLabel is not None and yLabel is not None:
        ax.set_xlabel(xLabel, fontsize=22)
        ax.set_ylabel(yLabel, fontsize=22)
        
    if names is not None:
        plt.setp(ax, xticks=[y for y in range (nBoxes)], xticklabels=names)
        
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)         
      
        
    
        
    
    fig = ax.get_figure()   
    
    fig.set_size_inches(figSize[0], figSize[1])
    
    if saveName is not None:                
        fig.savefig(saveName+'.pdf', format='pdf')    
        
    plt.show()

        
# EXAMPLE:
#data1 = [np.random.normal(0, std, 100) for std in range(1,4)]
#data2 = [np.random.normal(2, std, 100) for std in range(1,4)]
#data3 = [np.random.normal(3, std, 100) for std in range(1,4)]
#data4 = [np.random.normal(-1, std, 100) for std in range(1,4)]
#data = [data1, data2, data3, data4]
#boxplotSNS (data, xLabel='Xis', yLabel='Episolon', names=['a1', 'a1', 'a3', 'a4'], saveName='Teste1', figSize=[8,7], pal=["#6b8ba4", "#b1d1fc", "#1fa774", "#c79fef"])
#boxplot (data)
#boxplot(data1, nPlots=1,  posSubPlot=[1,1], names=['a1', 'a1', 'a3'], xLabel='Xis', yLabel='Episolon', saveName='Teste1', figSize=[8,7])
