'''
2-Fold Cross-Validation:
-randomly shuffle dataset into two sets of equal size. train on first & test
-on second, then train on second and test on first. training forms a model.
-model is discarded after evaluation
-skill scores are collected for each model and summarized for use

============

This python program executes the following:
-- Uses 2-fold cross-validation to produce a test set of 150 samples of the iris 
data set with the following ML models: 
- Linear regression (LinearRegression)
- Polynomial of degree 2 regression (LinearRegression)
- Polynomial of degree 3 regression (LinearRegression)
- Naïve Baysian (GaussianNB) 
- kNN (KNeighborsClassifier) 
- LDA (LinearDiscriminantAnalysis) 
- QDA (QuadraticDiscriminantAnalysis)
- SVM (LinearSVC)
- Decision Tree (DecisionTreeClassifier)
- Random Forect (RandomForestClassifier)
- ExtraTrees (ExtraTreesClassifier)
- NN (neural_network.MLPClassifier)

-- This program displays, along with the model label, the:
- Confusion matrix
- Accuracy metric

Example: in python notebook, run:
%run path/to/program/CompareMLModelsV2.py
'''

#load all relevant (and potentially relevant) libraries
import numpy as np
import scipy
from pandas import read_csv
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier



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
    

def train_data_gaussianNB(xfold1,xfold2,yfold1,yfold2):
    model = GaussianNB()
    model.fit(xfold1,yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_NB = np.concatenate([yfold2,yfold1])
    pred_NB = np.concatenate([pred1,pred2])
    
    print('Naive Bayesian')
    print('Overall Accuracy: ',accuracy_score(actual_NB,pred_NB))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_NB,pred_NB))
    print(' ')
    print(' ')


def train_data_kNN(xfold1,xfold2,yfold1,yfold2):
    #used p=2, corresponding to Euclidean distance --> length, width used as features
    #assume uniform weight, I suppose...
    #n_neighbors encodes k; three classes, so try even k value.
    #k=10 gives 97.3% accuracy; k=13 gives 98% accuracy
    #did not normalize data, since scales are not different (all are in cm)
    model=KNeighborsClassifier(p=2)
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_kNN = np.concatenate([yfold2,yfold1])
    pred_kNN = np.concatenate([pred1,pred2])

    print('kNN (p=2)')
    print('Overall Accuracy: ',accuracy_score(actual_kNN,pred_kNN))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_kNN,pred_kNN))
    print(' ')
    print(' ')
    
    
def train_data_LDA(xfold1,xfold2,yfold1,yfold2):
    #a more involved version of NB, does not assume features are independent
    #(fits class conditional densities to the data and uses Bayes' rule.)
    #assumes, however, that all classes share the same covariance matrix (i.e., each feature varies around the mean by the same amount on average)
    model=LinearDiscriminantAnalysis(solver='lsqr')
    model=KNeighborsClassifier(n_neighbors=13,p=2)
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_lda = np.concatenate([yfold2,yfold1])
    pred_lda = np.concatenate([pred1,pred2])
    
    print('LDA')
    print('Overall Accuracy: ',accuracy_score(actual_lda,pred_lda))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_lda,pred_lda))
    print(' ')
    print(' ')


def train_data_QDA(xfold1,xfold2,yfold1,yfold2):
    #similar to LDA, but does not assume all classes share the same covariance matrix
    #moreover, I ultimately did not tweak any QDA parameters, as I am unsure what they represent; for instance, I could improve accuracy to 0.98 if I let reg_param=0.05, but the meaning of this parameter is unknown to me.
    #model=QuadraticDiscriminantAnalysis(reg_param=0.05)
    model=QuadraticDiscriminantAnalysis()
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_qda = np.concatenate([yfold2,yfold1])
    pred_qda = np.concatenate([pred1,pred2])

    print('QDA (reg_param=0.05)')
    print('Overall Accuracy: ',accuracy_score(actual_qda,pred_qda))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_qda,pred_qda))
    print(' ')
    print(' ')
    
    
def train_data_linreg(xfold1,xfold2,yfold1,yfold2,deg):
    
    model=LinearRegression()
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_lin = np.concatenate([yfold2,yfold1])
    pred_lin = np.concatenate([pred1,pred2])
    #convert values into integers, since the classes are integers!
    pred_lin = pred_lin.round()

    #note that my integer classes are 0, 1, and 2. Rounding, however, can yield a class >2. This is no bueno, since
    #1. there is no class associated with the integer 3
    #2. due to the above, the confusion matrix will add a column with zeros for the diagonal term
    #to fix, make values <0 a 0, and values >2 a 2. That is all.
    #see 'for' loop below.
    #I complete this loop following the accuracy statement, as doing so before will affect the printed score.
    
    if int(deg)==1:
        print('Linear Regression (deg=1)')
    if int(deg)==2:
        print('Polynomial Regression (deg=2)')
    if int(deg)==3:
        print('Polynomial Regression (deg=3)')
    print('Overall Accuracy: ',accuracy_score(actual_lin,pred_lin))
    for i in range(0,len(pred_lin)):
        if pred_lin[i] < 0.:
            pred_lin[i] = 0.
        if pred_lin[i] > 2.:
            pred_lin[i] = 2.
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_lin,pred_lin))
    print(' ')
    print(' ')
    
    
def train_data_SVM(xfold1,xfold2,yfold1,yfold2):
    #the Support Vector Machine!
    #draws line between clusters of classes (plotted in feature space) then using this line to predict the class of some configuration of features.
    #finds "optimal division line," where "extreme points" of each cluster appear to be supporting, or holding up, the optimal line.
    #SVC --> Linear Support Vector Classification
    #uses linear kernal by default, but lacks parameters of SVC/NuSVC...CORRECT KERNAL FUNCTION IS *CRITICAL* TO THE PERFORMANCE OF THIS CLASSIFIER, so if scattering of clusters is nonlinear then we have a problem-o.
    #set dual=False for this scenario where #samples > #features
    model=LinearSVC(dual=False)
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_svc = np.concatenate([yfold2,yfold1])
    pred_svc = np.concatenate([pred1,pred2])

    print('Linear_SVC (dual=False)')
    print('Overall Accuracy: ',accuracy_score(actual_svc,pred_svc))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_svc,pred_svc))
    print(' ')
    print(' ')
    
    
def train_data_dectree(xfold1,xfold2,yfold1,yfold2):
    #overfitting tendency
    #unstable; small variations in data might resultin completely different tree
    #training algorithms do not guarantee globally optimal decision trees
    #random_state=1:
        #first iteration, 0.953
        #second iteration, 0.927
        #third iteration, 0.92
        #fourth iteration, 0.953
    #anyhow.
    model=DecisionTreeClassifier()
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_dec = np.concatenate([yfold2,yfold1])
    pred_dec = np.concatenate([pred1,pred2])

    print('Decision Tree (iteration results may vary)')
    print('Overall Accuracy: ',accuracy_score(actual_dec,pred_dec))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_dec,pred_dec))
    print(' ')
    print(' ')
    

def train_data_ranforest(xfold1,xfold2,yfold1,yfold2):
    #mitigates but does not eliminate the weaknesses of decision trees
    #generates MULTIPLE decision trees (a forest), then final classification is taken as the mode of different classifications
    #for n_estimators=100 (default number of decision trees in forest), highest after 100 iterations: 0.96.
    #with n_estimators=1000, score is more frequently 0.96; after 100 iterations, seldom below 0.947
    model=RandomForestClassifier()
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_rain = np.concatenate([yfold2,yfold1])
    pred_rain = np.concatenate([pred1,pred2])

    print('Random Forest (iteration results may vary)')
    print('Overall Accuracy: ',accuracy_score(actual_rain,pred_rain))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_rain,pred_rain))
    print(' ')
    print(' ')


def train_data_RANforest(xfold1,xfold2,yfold1,yfold2):
    #extremely randomized trees (hence the capitalization of 'RAN')
    #adds additional step to random forest --> splitting rules in decision trees determined after drawing thresholds at random for each candidate feature then selecting the "best"
    #after 100 iterations with n_estimators=100, appears that average accuracy score is highest for this classifier as compared with randomforest and decisiontree
    model=ExtraTreesClassifier()
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_RAIN = np.concatenate([yfold2,yfold1])
    pred_RAIN = np.concatenate([pred1,pred2])

    print('ExtraTrees (iteration results may vary)')
    print('Overall Accuracy: ',accuracy_score(actual_RAIN,pred_RAIN))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_RAIN,pred_RAIN))
    print(' ')
    print(' ')


def train_data_neuralnet(xfold1,xfold2,yfold1,yfold2):
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
    print('Overall Accuracy: ',accuracy_score(actual_nn,pred_nn))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual_nn,pred_nn))
    print(' ')
    print(' ')



    
if __name__ == "__main__":
    
    
    homedir = os.getenv("HOME")
    iris_dat = read_table(homedir=homedir) #read iris data table
    x,y = isol_array(iris_dat) #separate data into x (shape --> 150,4) and y (shape --> 150,1) columns
    
    #the following models do not require integer classes, so I begin with this set
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x,y,deg=1) #deg=1, which leaves x effectively untouched
    train_data_gaussianNB(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_kNN(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_LDA(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_QDA(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    
    
    #"new models" for V2 of this python routine
    train_data_SVM(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_dectree(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_ranforest(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_RANforest(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    train_data_neuralnet(X_Fold1, X_Fold2, y_Fold1, y_Fold2)
    
    
    #switching to the regression models...
    #Linear first, so deg = 1
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x,y,class_int_list=True,deg=1)
    train_data_linreg(X_Fold1, X_Fold2, y_Fold1, y_Fold2,deg=1)
    
    #now deg=2
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x,y,class_int_list=True,deg=2)
    train_data_linreg(X_Fold1, X_Fold2, y_Fold1, y_Fold2,deg=2)
    
    #lastly, deg=3
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(x,y,class_int_list=True,deg=3)
    #note --> as deg rises, over-fitting appears to become a problem-o.
    train_data_linreg(X_Fold1, X_Fold2, y_Fold1, y_Fold2,deg=3)
   
    
