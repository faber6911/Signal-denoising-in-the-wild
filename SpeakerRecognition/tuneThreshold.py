#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    # need to add accuracy score based on threshold
    threshold = thresholds[idxE]
    y_pred = numpy.where(scores >= threshold, 1, 0)
    acc = metrics.accuracy_score(labels, y_pred)*100
    
    return (tunedThreshold, eer, acc, fpr, fnr);
