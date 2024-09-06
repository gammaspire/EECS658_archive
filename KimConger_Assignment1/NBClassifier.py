#!/usr/bin/env python
# coding: utf-8

#aim: randomly shuffle dataset into two sets of equal size. train on first & test on second (k=1 folding...). Print output confusion matrix and associated score summary.

#load all relevant (and potentially relevant) libraries
import numpy as np
import scipy
from pandas import read_csv
import os
#import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

def read_table(homedir):
    #define column names of .csv table
    names = ['sepal-length','sepal-width','petal-length','petal-width','class']
    #read table with astropy.table.Table
    iris_dat = read_csv(homedir+'/Downloads/iris.csv',names=names)
    return(iris_dat)
    
def isol_array(dat):
    array = dat.values
    #isolate flower features in line-matched rows
    x = array[:,0:4]
    #isolate flower classes
    y = array[:,4]
    return(x,y)

def train_data(xfold,yfold,xfold2):
    model = GaussianNB()
    model.fit(xfold,yfold)
    pred1 = model.predict(xfold2)
    return(pred1)

if __name__ == "__main__":
    
    homedir = os.getenv("HOME")
    iris_dat = read_table(homedir=homedir)
    x,y = isol_array(iris_dat)
    
    #random state --> splits 'randomly' in a way that is reproducible for all who use the same integer (in this case, 1)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(x, y,
test_size=0.50, random_state=1)
    pred1 = train_data(X_Fold1,y_Fold1,X_Fold2)
    print('Overall Accuracy: ',accuracy_score(y_Fold1,pred1))
    print(' ')
    print('Confusion Matrix: ')
    print(confusion_matrix(y_Fold1,pred1))
    print(' ')
    print('Score Summary: ')
    print(classification_report(y_Fold1,pred1))
