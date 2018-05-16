#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find any bug, please email me =)
"""
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations



# This function performs the Friedman's test. If pv returned by the friedman test
# is less than the pvRef, the Wilcoxon test is also performed
# input: 
#       data arrays - the data that the test will perform. Algorithms x Samples
#       for example a matrix 3 x 10 contains 3 algorithms with 10 samples each
#       pvRef - pvalue ref
#       names - a list with database names. Ex: n = ['X', 'Y', 'Z']
def statisticalTest (data, names, pvRef, verbose=True):    
    data = np.asarray(data)
    if data.shape[0] != len(names):
        raise ('ERROR: the size of the data row must be the same of the names')

    out = 'Performing the Friedman\'s test...'
    sFri, pvFri = friedmanchisquare(*[data[i,:] for i in range(data.shape[0])])
    out += 'Pvalue = ' + str(pvFri) + '\n'

    if pvFri > pvRef:
        out += 'There is no need to pairwise comparison because pv > pvRef'
    else:
        out += '\nPerforming the Wilcoxon\'s test...\n'
        combs = list(combinations(range(data.shape[0]),2))
        for c in combs:           
            sWil, pvWill = wilcoxon (data[c[0],:], data[c[1],:])
            out += 'Comparing ' + names[c[0]] + ' - ' + names[c[1]] + ': pValue = ' + str(pvWill) + '\n'
            
    if verbose:
        print out
        
    return out  
    

# This function returns the gaussian PDF
def gaussianPDF(x, mean, stdev):
    p1 = 1 / (np.sqrt(2*np.pi*np.power(stdev,2)))
    p2 = np.exp(-(np.power((x-mean),2))/2*np.power(stdev,2))
    return p1*p2
   

# EXAMPLE
#vals = np.array( [[0.00349, 0.00273, 0.00002, 0.00009],
#        [0.00053, 0.00056, 0.00008, 0.00013],
#        [0.00171, 0.00142, 0.00030, 0.00071],
#        [0.00972, 0.00922, 0.00222, 0.00534],
#        [0.03090, 0.02980, 0.06570, 0.02630],
#        [0.06638, 0.07131, 0.05182, 0.04387],
#        [0.04890, 0.05402, 0.03953, 0.03946],
#        [0.02460, 0.02410, 0.03120, 0.02220],
#        [0.05088, 0.05011, 0.03469, 0.03459],
#        [0.10506, 0.10275, 0.03853, 0.03842],
#        [0.04228, 0.06492, 0.02752, 0.02749]] ).T
#
#
#stdVals = np.array([[0.00186, 0.00417, 0.00001, 0.00004],
#           [0.00026, 0.00025, 0.00002, 0.00004], 
#           [0.00078, 0.00055, 0.00012, 0.00041], 
#           [0.00291, 0.00364, 0.00060, 0.00148],
#           [0.00517, 0.00499, 0.00060, 0.00063],
#           [0.03292, 0.02353, 0.02639, 0.01757],
#           [0.01768, 0.00982, 0.00015, 0.00022],
#           [0.00195, 0.00217, 0.00019, 0.00013],
#           [0.01393, 0.01712, 0.00122, 0.00083],
#           [0.06068, 0.07357, 0.00014, 0.00012],
#           [0.02847, 0.03455, 0.00018, 0.00030]]).T
#
#
#statisticalTest (vals, ['OS-ELM','KLM','OR-ELM','OR-KLM'],0.05)


