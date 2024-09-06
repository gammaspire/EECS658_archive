'''
2-Fold Cross-Validation:
-randomly shuffle dataset into two sets of equal size. train on first & test
-on second, then train on second and test on first. training forms a model.
-model is discarded after evaluation
-skill scores are collected for each model and summarized for use

============

This python program compares the accuracy scores of Neural Network model classifications (neural_network.MLPClassifier) for imbalanced data, where the data is either:
-----not treated
-----oversampled
-----undersampled

-- This program displays, for each 'iteration', the:
- Confusion matrix
- Accuracy metric
- Class-Balanced Accuracy
- Balanced Accuracy
- Sklearn Balanced Accuracy

Example: in python notebook, run:
%run path/to/program/ImbalancedDataSampling.py
'''

import os
homedir = os.getenv("HOME")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

from pandas import read_csv
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier

from imblearn import over_sampling
from imblearn import under_sampling


def read_table(homedir):
    #read_table
    iris_dat = read_csv(homedir+'/Desktop/KimConger_Assignment5/imbalanced_iris.csv')
    return(iris_dat)
    

def isol_array(iris_dat):
    #separate data array from header information
    array = iris_dat.values
    #isolate flower features in line-matched rows
    x = array[:,0:4]
    #isolate flower classes
    y = array[:,4]
    return(x,y)


def fold_data(features_x,classes_y,class_int_list=False,deg=1):
    
    #set class_int_list=True if want to convert string classes to integer classes
    
    #if deg=1, then it is the case that x=features_x. no change.
    #if deg=2, then shape becomes (150,14), with the additional 10 columns to account for the additional 10 polynomial coefficients...x1x2, x1x3, x1x4, x2x3, x2x4, x3x4, x1x1, x2x2, x3x3, x4x4. 
    #if deg=3, same as above but with more columns. recall that polynomial fit represented as a matrix.
    
    #LinearRegression is somehow able to parse initial number of features (x_n) and the remaining polynomial coefficients...
    
    x = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(features_x)
    
    if class_int_list == True:
        f = LabelEncoder()
        f.fit(classes_y)
        classes_y=f.transform(classes_y)
    
    #random state --> splits 'randomly' in a way that is reproducible for all instances with the same integer (in this case, 1)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(x, classes_y, test_size=0.50, random_state=1)
    return(X_Fold1, X_Fold2, y_Fold1, y_Fold2)


def train_data_neuralnet(xfold1,xfold2,yfold1,yfold2,var=0):
    #composed of input layer, 1+ hidden layers, output layer
    #activation function --> y=wx+b
    #each node in any layer sums its inputs (x), multiplies the sum (x) by weight (w), then adds bias (b). feeds results f(x) into next mode. each node has its own weight and bias.
    #MANY, MANY, MANY PARAMETERS --> "requires tuning a number of hyperparameters such as the number of hidden neurons, layers, and iterations"
    #uses RELU activation function (rectified linear unit function, returns f(x) = max(0,x))
    #hidden_layer_sizes parameter, governs number of hidden layers and number of nodes per layer
    model=MLPClassifier(max_iter=5000)
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_nn = np.concatenate([yfold2,yfold1])
    pred_nn = np.concatenate([pred1,pred2])

    print('Neural Network (MLPClassifier)')
    print('Overall Accuracy: ','%.3f'%accuracy_score(actual_nn,pred_nn))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_nn,pred_nn))
    print(' ')
    if int(var) == 1:
        return(confusion_matrix(actual_nn,pred_nn), pred_nn, actual_nn)
    print(' ')


def calculate_accuracies(con_matrix,actual_nn,pred_nn):
    #class balanced accuracy = mean(min(P,R) per class)
    #setosa, versicolor, virginica
    CBA_list = []
    #for each of the 3 classes, calculate precision and recall; append minimum of these to CBA_list
    for i in range(3): 
        precision_i = con_matrix[i][i]/np.sum(con_matrix[i])
        recall_i = con_matrix[i][i]/np.sum(con_matrix[:,i])
        if (precision_i <= recall_i):
            CBA_list.append(precision_i)
        if (precision_i > recall_i):
            CBA_list.append(recall_i)

    print('Class Balanced Accuracy:','%.3f'%np.mean(CBA_list))
    
    #balanced accuracy = average(average(S,R) per class)
    BA_list = []
    
    #for TN, all non-N instances that are NOT classified as class N
    TN = [np.sum([con_matrix[1][1],con_matrix[1][2],con_matrix[2][1],con_matrix[2][2]]),
         np.sum([con_matrix[0][0],con_matrix[0][2],con_matrix[2][0],con_matrix[2][2]]),
         np.sum([con_matrix[0][0],con_matrix[0][1],con_matrix[1][0],con_matrix[1][1]])]
    #for FP, all non-N instances that ARE classified as class N
    FP = [np.sum([con_matrix[0][1],con_matrix[0][2]]),
         np.sum([con_matrix[1][0],con_matrix[1][2]]),
         np.sum([con_matrix[2][0],con_matrix[2][1]])]
    
    for i in range(3):
        S_i = TN[i]/(TN[i]+FP[i])
        recall_i = con_matrix[i][i]/np.sum(con_matrix[:,i])

        BA_list.append(np.mean([S_i,recall_i]))
    
    print('Balanced Accuracy:','%.3f'%np.mean(BA_list))
    
    print('Sklearn Balanced Accuracy:','%.3f'%balanced_accuracy_score(actual_nn,pred_nn))


if __name__ == "__main__":
    
    
    homedir = os.getenv("HOME")
    iris_dat = read_table(homedir)
    x,y = isol_array(iris_dat) #separate data into x (features) and y (classes) variables
    
    #PART ONE:    
    #print and label confusion matrix and accuracy score
    #print and label Class Balanced Accuracy
    #print and label Balanced Accuracy
    #print and label sklearn's function balanced_accuracy_score as described in the Imbalanced Datasets lecture
    
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x,y,deg=1) #deg=1, which leaves x effectively untouched. note that deg argument is a relic of lin reg models from assignments 2-3
    print('~~~~~~~~~~~~~PART ONE~~~~~~~~~~~~~')
    print()
    con_matrix, pred_nn, actual_nn = train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2, var=1)
    calculate_accuracies(con_matrix,actual_nn,pred_nn)
    print()
    
    
    #PART TWO
    #balance the imbalanced iris_dat with random oversampling and print/label the confusion matrix and accuracy score
    #balance the imbalanced iris_dat with SMOTE oversampling and print/label the confusion matrix and accuracy score
    #balance the imbalanced iris_dat with ADASYN oversampling and print/label the confusion matrix and accuracy score; use the sampling_strategy = 'minority' parameter
    #NOTES: don’t need to balanced accuracy scores from Part 1 because the set is balanced now; resulting dataset may still not be balanced, which is fine.
    
    print('~~~~~~~~~~~~~PART TWO~~~~~~~~~~~~~')
    print()
    #over-sample (the 42 is because I can)
    print('RANDOM OVERSAMPLING')
    ros = over_sampling.RandomOverSampler(random_state=42)
    x_ros, y_ros = ros.fit_resample(x,y)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x_ros,y_ros,class_int_list=False,deg=1)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    print('SMOTE OVERSAMPLING')
    sm = over_sampling.SMOTE(random_state=42)
    x_ros, y_ros = sm.fit_resample(x,y)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x_ros,y_ros,class_int_list=False,deg=1)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    print('ADASYN OVERSAMPLING')
    ada = over_sampling.ADASYN(random_state=42,sampling_strategy='minority')
    x_ros, y_ros = ada.fit_resample(x,y)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x_ros,y_ros,class_int_list=False,deg=1)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    print()
    
    
    #PART THREE
    #balance the imbalanced iris_dat with random undersampling and print/label the confusion matrix and accuracy score
    #balance the imbalanced iris_dat with Cluster undersampling and print/label the confusion matrix and accuracy score
    #balance the imbalanced iris_dat with Tomek Links undersampling and print/label the confusion matrix and accuracy score
    #NOTES: don’t need to balanced accuracy scores from Part 1 because the set is balanced now; resulting dataset may still not be balanced, which is fine.
    
    
    print('~~~~~~~~~~~~PART THREE~~~~~~~~~~~~')
    print()
    print('RANDOM UNDERSAMPLING')
    rus = under_sampling.RandomUnderSampler(random_state=42)
    x_rus, y_rus = rus.fit_resample(x,y)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x_rus,y_rus,class_int_list=False,deg=1)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    print('Cluster Undersampling')
    rus = under_sampling.ClusterCentroids()
    x_rus, y_rus = rus.fit_resample(x,y)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x_rus,y_rus,class_int_list=False,deg=1)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    print('Tomek Links Undersampling')
    rus = under_sampling.TomekLinks()
    x_rus, y_rus = rus.fit_resample(x,y)
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x_rus,y_rus,class_int_list=False,deg=1)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
        
    
    
    
