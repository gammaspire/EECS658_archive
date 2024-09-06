'''
2-Fold Cross-Validation:
-randomly shuffle dataset into two sets of equal size. train on first & test
-on second, then train on second and test on first. training forms a model.
-model is discarded after evaluation
-skill scores are collected for each model and summarized for use

============

This python program executes the following:
-- Uses 2-fold cross-validation to produce a test set of 150 samples of the iris 
data set with the Decision Tree ML model; features determined using:
---- default four Iris features
---- PCA feature transformation
---- Simulated Annealing
---- Genetic Algorithm
-- This program displays, along with the model label, the:
---- Confusion matrix
---- Accuracy metric
---- List of features used to obtain the final confusion matrix and accuracy metric

Example: in python notebook, run:
%run path/to/program/CompareFeatureSelectionMethods.py
'''

#load all relevant libraries
import numpy as np
import scipy
from pandas import read_csv
import pandas as pd
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA


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
    

def pca_feature_selection(x_data):
    #aim of PCA --> reduce number of features while simultaneously preserving as much information as possible
    #PCA converts the features matrix (n rows = number of samples, m columns = number of features) to a centered matrix, meaning all elements in a column are subtracted by the mean of the column; calculates the covariance matrix of the centered matrix (covariance is a measure of the LINEAR correlation between two features), where the diagonal gives the variance of each feature; computes the eigenvalues and eigenvectors of this covariance matrix; THEN determines the transformed parameters Z = XW^T, where X is the X_centered matrix of the original features and W is the eigenvectors matrix.
    #this function will:
        #fit to all original features x, generate transformed matrix Z
        #find where the PoV array > 90%, then save that index in a variable
        #determine Z column of principle components corresponding to the
        
    #perform PCA
    pca = PCA(n_components=4)
    pca.fit(x)

    #get eigenvectors and eigenvalues
    eigenvecs = pca.components_
    eigenvals = pca.explained_variance_
    print('eigenvectors:')
    print(eigenvecs)
    print(' ')
    print('eigenvalues:')
    print(eigenvals)
    print(' ')
    
    #make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = list(zip(eigenvals,eigenvecs))

    #transform data
    principleComponents = pca.transform(x) #transformed matrix

    #calculate PoVs (proportion of variance)
    sumvariance = np.cumsum(eigenvals)
    #c /= a is equivalent to c = c / a
    #so we are scaling all elements with the last element in sumvariance (100)
    #first element contains largest delta_PoV; the rest only rise by a few percent; is also > 0.90, the
    #'typical' PoV that defines which component feature(s) to use
    sumvariance /= sumvariance[-1]
    print('PoV (decimals):')
    print(sumvariance)
    print(' ')
    
    index = np.where(sumvariance>0.90)[0][0]   #isolates first index at which element >.90, or 90%
    
    #this section is a bit puzzling to me. From lecture, Z = XW^T, where X is the centered matrix. However, the principleComponents matrix supposedly represents already the transformed matrix Z, so there is no need to perform a dot product with W. Instead, I need only isolate the first element of every column from principleComponents in order to prepare the row of first principle features for training/testing, no?
    #but alas, I must follow what is given as the example in the lecture notes, which indeed uses Z = principleComponents.dot(W)
    
    #if index>0, then there would be more than one principle component to include, so I would have to edit the definition of W to accommodate more than one column. Not necessary for this project, but I add this note as a precaution if I ever wish to generalize this work.
    
    #DR. JOHNSON HAS CONFIRMED THAT I MAY GO AHEAD AND SIMPLY DEFINE Z AS THE FIRST COLUMN IN THE PRINCIPLECOMPONENTS MATRIX. As such, I will comment out this W and replace it with the...proper W.
    #W = eigen_pairs[int(index)][1].reshape(4,1)   #converts eigenvector at index to from a row to column vector
    W = np.array([1,0,0,0]).reshape(4,1)
    Z = principleComponents.dot(W)
    
    return(principleComponents,eigen_pairs,Z)


def train_data_dectree(xfold1,xfold2,yfold1,yfold2,print_dat=1):
    #overfitting tendency
    #unstable; small variations in data might result in completely different tree
    #training algorithms do not guarantee globally optimal decision trees
    #anyhow.
    model=DecisionTreeClassifier()
    model.fit(xfold1, yfold1) #first fold training
    pred1 = model.predict(xfold2) #first fold testing
    model.fit(xfold2,yfold2) #second fold training
    pred2 = model.predict(xfold1) #second fold testing
    actual_dec = np.concatenate([yfold2,yfold1])
    pred_dec = np.concatenate([pred1,pred2])
    
    if print_dat==1:
        print('Decision Tree (iteration results may vary)')
        print('Overall Accuracy: ',accuracy_score(actual_dec,pred_dec))
        print('Confusion Matrix: ')
        print(confusion_matrix(actual_dec,pred_dec))
        print(' ')
        print(' ')
    return(accuracy_score(actual_dec,pred_dec))
    

def simulated_annealing(x,y,principleComponents):
    #use simulated annealing to select the best set of features from the 4 original features (SL, SW, PL, PW) plus the four transformed features (z1, z2, z3, z4) from Part 2 (for a total of 8 features)
    #let number of iterations = 100
    #perturb with randomly selected 1 or 2 parameters (since 1-5% of 8 is <1)
    #(that is, number of features is too few to use percentage)
    #c in Pr[accept] = 1
    #Use restart value (x) of 10
    #for each iteration, print
        #the subset of features
        #accuracy
        #Pr[accept]
        #Random Uniform
        #Status: Improved, Accepted, Discarded, or Repeat
        
    #the steps consist of the following:
        #for each iteration,
            #perturb the current feature subset
            #fit model and estimate performance
            #if performance > previous subset,
                #accept new subset
            #else:
                #calculate acceptance probability
                #if random uniform variable > probability,
                    #reject new subset
                #else:
                    #accept new subset
    
    #firstly, we define z1, z2, z3, and z4
    w1 = np.array([1,0,0,0]).reshape(4,1)
    w2 = np.array([0,1,0,0]).reshape(4,1)
    w3 = np.array([0,0,1,0]).reshape(4,1)
    w4 = np.array([0,0,0,1]).reshape(4,1)
    
    z1 = principleComponents.dot(w1)
    z2 = principleComponents.dot(w2)
    z3 = principleComponents.dot(w3)
    z4 = principleComponents.dot(w4)
    
    #isolate the x features
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    x4 = x[:,3]
    
    #define number of iterations, constant C, restart value
    max_restart_counter = 10
    n_iterations = 100
    C = 1
    
    #define restart counter, best_accuracy as beginning at zero
    restart_counter = 0
    
    #defining set variables
    feature_set = np.column_stack((x1,x2,x3,x4,z1,z2,z3,z4))  #total set
    #feature_set = pd.DataFrame(feature_set, columns = ['x1','x2','x3','x4','z1','z2','z3','z4'])
    feature_set_labels = ['x1','x2','x3','x4','z1','z2','z3','z4']
    #feature_set_labels = feature_set.columns
    feature_subset = np.column_stack((x1,x2,x3,x4,z1,z2,z3,z4))  #starting set (all features)
    
    feature_subset_labels =  ['x1','x2','x3','x4','z1','z2','z3','z4']
    
    best_subset = np.column_stack((x1,x2,x3,x4,z1,z2,z3,z4))  #best feature set
    best_subset_labels = ['x1','x2','x3','x4','z1','z2','z3','z4']
    best_features_not_used = []
    
    #determine accuracy_score using full feature_set
    print(' ')
    print('iteration 0')
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(feature_subset,y)
    new_accuracy = train_data_dectree(X_Fold1,X_Fold2,y_Fold1,y_Fold2)
    first_accuracy = new_accuracy
    best_accuracy = new_accuracy
    
    #lastly, create empty features list to keep a record of features *not* in current set (begins empty since all 8 features comprise subset)
    features_not_used = []
    
    #off we go.
    for n in range(n_iterations):
        feature_set_labels = ['x1','x2','x3','x4','z1','z2','z3','z4']
        print('iteration',n+1)
        #reset name of i-1 accuracy to be old_accuracy (or, if i=0, old_accuracy is the calculated accuracy with all features)
        old_accuracy = new_accuracy
        #do likewise for the feature subsets
        old_feature_subset = np.ndarray.tolist(feature_subset)
        old_feature_subset = np.array(feature_subset)
        #annnnd, the label lists
        features_not_used_old = np.array(features_not_used)
        features_not_used_old = np.ndarray.tolist(features_not_used_old)
        #the following two lines prevent the changing of feature_subset_labels to in turn change feature_subset_labels_old
        feature_subset_labels_old = np.array(feature_subset_labels)
        feature_subset_labels_old = np.ndarray.tolist(feature_subset_labels_old)
                
        #number of feature lists to add or delete --> m = 1 or 2
        #determine randomly.
        r = random.random()
        if r <= 0.5:
            m = 1
        if r > 0.5:
            m = 2
        
        #if all features in subset (number of columns is 8), delete only
        if (np.shape(old_feature_subset)[1] == 8) | (len(features_not_used_old) == 0):
            print('old feature subset is full')
            print('old feature subset:',feature_subset_labels)
            random_choice_array = np.arange(0,8,1)
            feature_index = random.choice(random_choice_array)
            feature_subset = np.delete(old_feature_subset,[feature_index],1)
            #append feature label to used_features list
            features_not_used.append(feature_subset_labels[feature_index])
            feature_subset_labels.remove(feature_subset_labels[feature_index])
            if m == 2:
                print('m=2')
                random_choice_array2 = np.arange(0,len(feature_subset_labels),1)
                feature_index2 = random.choice(random_choice_array2)
                feature_subset = np.delete(feature_subset,[feature_index2],1)
                #append feature label to used_features list
                features_not_used.append(feature_subset_labels[feature_index2])
                feature_subset_labels.remove(feature_subset_labels[feature_index2])
            print('new feature subset:',feature_subset_labels)
            print('list of features unused',features_not_used)
            
        #if subset is empty (number of columns is 0), add only
        elif np.shape(old_feature_subset)[0] == 0:
            print('old feature subset is empty')
            print('old feature subset:',feature_subset_labels)
            random_choice_array = np.arange(0,8,1)
            feature_index = random.choice(random_choice_array)
            #new feature subset becomes the single feature set column
            feature_subset = feature_set[:,feature_index]
            #in this case, all 8 feature labels should be in features_not_used. remove that/those corresponding to added features to subset
            #update features_not_used list
            features_not_used.remove(feature_set_labels[feature_index])
            #print old feature subset, then add new feature!
            print('old feature subset:',feature_subset_labels)
            feature_subset_labels.append(feature_set_labels[feature_index])
            
            if m == 2:
                print('m=2')
                random_choice_array2 = np.arange(0,7,1)  #removed 1, 7 left
                feature_index2 = random.choice(random_choice_array2)
                #if...else statement is to avoid duplicates
                if features_not_used[feature_index2] in feature_subset_labels:
                    feature_index2 = np.where(np.array(feature_subset_labels) == np.array(features_not_used)[feature_index2])[0]
                    feature_index2 = feature_index2[0]  #isolate the integer!
                    if feature_index2==7:  #choose a new integer, since previous feature is already in subset. if 7, subtract; otherwise, add
                        feature_index2 -= 1
                    else:
                        feature_index2 += 1
                else:
                    feature_index2 = np.where(np.array(feature_set_labels) == np.array(features_not_used)[feature_index2])[0]
                    feature_index2 = feature_index2[0]
            
                feature_subset = np.column_stack((feature_subset,feature_set[:,feature_index2]))
           
            #update features_not_used list
            features_not_used.remove(feature_set_labels[feature_index2])
                
            feature_subset_labels.append(feature_set_labels[feature_index2])
            print('new feature subset:',feature_subset_labels)
            print('list of features unused',features_not_used)
            
        else:
            r2 = random.random()
            if (r2 <= 0.5) & (len(feature_subset_labels_old) != 1):
                print('r2<=0.5, delete m features')
                #list of integers corresponding to each element in the old_feature_subset
                random_choice_array = np.arange(0,len(feature_subset_labels_old),1)
                #choose one of these integers 'at random'
                feature_index = random.choice(random_choice_array)
                #delete the appropriate feature, define as the new feature subset
                feature_subset = np.delete(old_feature_subset,[feature_index],1)
                print('old feature subset:',feature_subset_labels)
                #add the deleted label to 'features not used'
                features_not_used.append(feature_subset_labels_old[feature_index])
                #now delete appropriate feature label
                feature_subset_labels.remove(feature_subset_labels_old[feature_index])
                
                #only remove two feature if m=2 and the updated features_subset list contains 1 element...

                if (m == 2) & (len(feature_subset_labels) != 1):
                    print('m=2')
                    random_choice_array2 = np.arange(0,len(feature_subset_labels),1)
                    feature_index2 = random.choice(random_choice_array2)
                    feature_subset = np.delete(feature_subset,[feature_index2],1)
                    features_not_used.append(feature_subset_labels[feature_index2])
                    feature_subset_labels.remove(feature_subset_labels[feature_index2])
                    
            else:
                if len(features_not_used_old) == 1:
                    print('r2>0.5, len = 1')
                    #if only one feature not in subset, then we must...add that feature to the subset.
                    #(note that if m=2, we cannot add a ninth feature...so I ignore that statement here.)
                    #this one feature is located, of course, at the zeroth features_not_used index
                    #find at what index the feature_set_labels contains the features_not_used label
                    feature_index = np.where(np.array(feature_set_labels) == np.array(features_not_used)[0])[0]
                    #use that index to extract the feature column, append with old_feature_subset
                    feature_subset = np.column_stack((old_feature_subset,feature_set[:,feature_index]))
                    print('old feature subset:',feature_subset_labels)
                    #isolate integer
                    feature_index = feature_index[0]
                    features_not_used.remove(feature_set_labels[feature_index])
                    feature_subset_labels.append(feature_set_labels[feature_index])
                    
                if len(features_not_used_old) > 1:
                    print('r2>0.5, len > 1')
                    #this section is a bit trickier, as I must accommodate the m value as well as ensure that there are no duplicate features
                    #first, find random element corresponding to a feature not used (so program knows what it can add)
                    random_choice_array = np.arange(0,len(features_not_used),1)
                    feature_index = random.choice(random_choice_array)
                    while features_not_used[feature_index] in feature_subset_labels:  #while this condition is true,
                            feature_index = random.choice(random_choice_array)  #re-select random index
                            if features_not_used[feature_index] not in feature_subset_labels:  #if this condition is now true
                                break  #break 'while' loop
                    #determine index at which the randomly selected feature_not_used appears in the full feature set list (will then use this index to extract feature column from feature_set)
                    feature_index = np.where(np.array(feature_set_labels)==np.array(features_not_used)[feature_index])[0]
                    #isolate integer
                    feature_index = feature_index[0]
                    #combine...
                    feature_subset = np.column_stack((old_feature_subset,feature_set[:,feature_index]))
                    print('old feature subset:',feature_subset_labels)
                    #update 'features not used' list
                    #features_not_used_old = features_not_used
                    features_not_used.remove(feature_set_labels[feature_index])
                    feature_subset_labels.append(feature_set_labels[feature_index])

                    if m == 2:
                        print('m=2')
                        random_choice_array2 = np.arange(0,len(features_not_used),1)
                        feature_index2 = random.choice(random_choice_array2)
                        while features_not_used[feature_index2] in feature_subset_labels:  #while this condition is true,
                            feature_index2 = random.choice(random_choice_array2)  #re-select random index
                            if features_not_used[feature_index2] not in feature_subset_labels:  #if this condition is now true
                                break  #break 'while' loop
                        feature_index2 = np.where(np.array(feature_set_labels)==np.array(features_not_used)[feature_index2])[0]
                        feature_index2 = feature_index2[0]
                        feature_subset = np.column_stack((feature_subset,feature_set[:,feature_index2]))
                        #features_not_used_old = features_not_used
                        features_not_used.remove(feature_set_labels[feature_index2])
                        feature_subset_labels.append(feature_set_labels[feature_index2])
                    
            print('new feature subset:',feature_subset_labels)
            print('list of features unused:',features_not_used)
    
        #now check decisiontree model accuracy!
        #use a previously written function to perform 2-fold cross-validation, with new feature_subset
        X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(feature_subset,y)
        new_accuracy = train_data_dectree(X_Fold1,X_Fold2,y_Fold1,y_Fold2,print_dat=0)
        print('new accuracy:',new_accuracy)
        
        #returning to the algorithm,
        if new_accuracy > old_accuracy:
            print('STATUS: IMPROVED')
        #if new_accuracy is lower than the current best accuracy, compare a calculated random uniform variable with the acceptance probability
        new_acc = new_accuracy  #use new_accuracy in the 'restart' condition further below, but I change new_accuracy's value if new<best. this reserves that new_accuracy value.
        
        #if new_accuracy is lower than the previous accuracy, compare a calculated random uniform variable with the acceptance probability
        if (new_accuracy <= old_accuracy):
            p_accept = np.exp(-n/C * ((old_accuracy-new_accuracy)/old_accuracy))  #as n increases, the chance that random_uniform_variable > p_accept increases (and, as such, the chance of valley hopping decreases)
            print('p_accept:',p_accept)
            random_uniform_variable = random.random()
            print('random uniform variable:',random_uniform_variable)
            if random_uniform_variable > p_accept:
                #print('p_accept < R.U.V., so reverting to n-1 values.')
                print('STATUS: DISCARDED')
                feature_subset = old_feature_subset
                new_accuracy = old_accuracy
                #also have to reset the features_not_used and feature_subset_labels lists...ARGH.
                features_not_used = features_not_used_old
                feature_subset_labels = feature_subset_labels_old
                
            if random_uniform_variable < p_accept:
                #print('p_accept > R.U.V., so accepting new n values.')
                print('STATUS: ACCEPTED')
                
        #a modification called 'restarts' provides an additional layer of protection from lingering in inauspicious locales. If a new optimal solution has not been found within x=10 iterations, then the search resets to the last known optimal solution and proceeds again with n being the number of iterations since the restart.
        #that is...is new accuracy>best_accuracy, then replace subset and accuracy with these best versions; if new<best, then increment restart counter. when restart_counter = max_restart_counter, then set set feature_subset to be the current best_subset and set return restart_counter to zero.
        
        '''
        NO:
        b = [2]
        b = a
        a.append(1)
        print(b) --> [2,1]
        #################################
        YES:
        a = [2]
        b = np.array(a)
        b = np.ndarray.tolist(b)
        a.append(1)
        print(b) --> [2]
        '''
        
        if (new_acc > best_accuracy):

            if round(new_acc) == round(best_accuracy):
                if len(feature_subset_labels) < len(best_subset_labels):
                    print('same as best accuracy, but feature list is smaller than that of the current subset. update best feature subset.')
                    print('new acc',new_acc)
                    print('best accuracy',best_accuracy)
                    best_subset = np.ndarray.tolist(feature_subset)
                    best_subset = np.array(best_subset)
                    best_accuracy = new_accuracy
                    best_subset_labels = np.array(feature_subset_labels)
                    best_subset_labels = np.ndarray.tolist(best_subset_labels)
                    best_features_not_used = np.array(features_not_used)
                    best_features_not_used = np.ndarray.tolist(best_features_not_used)
                    restart_counter = 0
                else:
                    continue
                    
            if round(new_acc) != round(best_accuracy):
                print('new acc',new_acc)
                print('best accuracy',best_accuracy)
                best_subset = np.ndarray.tolist(feature_subset)
                best_subset = np.array(best_subset)
                best_accuracy = new_accuracy
                best_subset_labels = np.array(feature_subset_labels)
                best_subset_labels = np.ndarray.tolist(best_subset_labels)
                best_features_not_used = np.array(features_not_used)
                best_features_not_used = np.ndarray.tolist(best_features_not_used)
                restart_counter = 0
                print('updated best_accuracy',best_accuracy)
                print('updated best feature subset:',best_subset_labels)
                
        if (new_acc < best_accuracy):
            restart_counter+=1
            if restart_counter == max_restart_counter:
                print('STATUS: RESTART')
                feature_subset = np.ndarray.tolist(best_subset)
                feature_subset = np.array(feature_subset)
                feature_subset_labels = np.array(best_subset_labels)
                feature_subset_labels = np.ndarray.tolist(feature_subset_labels)
                features_not_used = np.array(best_features_not_used)
                features_not_used = np.ndarray.tolist(features_not_used)
                restart_counter = 0
        print(' ')
        print('best feature subset',best_subset_labels)
        print(' ')
        
    #run decisiontree model classifier on best feature set, which corresponds to the highest found accuracy!
    print(' ')
    print('first accuracy:',first_accuracy)
    print('best accuracy:',best_accuracy)
    print('best feature set:',best_subset_labels)
    print(' ')
    print('note: did not re-run classification model for this best subset, as the result is not consistent by nature of the decisiontree variability (decisiontree models may not converge on an optimum decision tree for every iteration!')
    #print('Results for best_accuracy feature set:')
    #X_Fold1, X_Fold2, y_Fold1, y_Fold2 = fold_data(best_subset,y)
    #new_accuracy = train_data_dectree(X_Fold1,X_Fold2,y_Fold1,y_Fold2,print_dat=1)


def genetic_algorithm(iris_dat,n_gen,principleComponents):
    
    #create table with all iris features, including transformed features
    iris_features_only = iris_dat.loc[:, 'sepal-length':'petal-width']
    iris_features_only.insert(4,'z1',principleComponents[:,0])
    iris_features_only.insert(5,'z2',principleComponents[:,1])
    iris_features_only.insert(6,'z3',principleComponents[:,2])
    iris_features_only.insert(7,'z4',principleComponents[:,3])
    
    #isolate data
    array = iris_features_only.values
    x_new = array[:,0:8]
    
    #isolate flower classes
    y = iris_dat.values[:,4]
    
    #5 sets of features (with 5 features each) as the initial generation of individuals
    A = iris_features_only[['z1','sepal-length','sepal-width','petal-length','petal-width']].copy()
    B = iris_features_only[['z1','z2','sepal-width','petal-length','petal-width']].copy()
    C = iris_features_only[['z1','z2','z3','sepal-width','petal-length']].copy()
    D = iris_features_only[['z1','z2','z3','z4','sepal-width']].copy()
    E = iris_features_only[['z1','z2','z3','z4','sepal-length']].copy()
    
    #for accounting purposes, since ideally the best individual of all generations will have both the highest accuracy AND the lowest number of features, I could extract one high-scoring individual from the five highest per generation (somewhat contrived, since there is a subset with .953 accuracy and only two features! --> otherwise, program tends toward individuals with 6 or 7 features, but comparable accuracy) and append them to a list. then, the individual in this list with the smallest number of features would reign. mwuhahaha and such. Alas, time constrains preclude this venture.
        
    for n in range(n_gen):
    
        original = [A,B,C,D,E]
        
        #to expand first generation, combine the features of each individual through crossover --> union or intersection
        #union --> 10 new sets
        AUB = pd.concat([A,B], axis=1)
        AUB = AUB.loc[:,~AUB.columns.duplicated()]

        AUD = pd.concat([A,D], axis=1)
        AUD = AUD.loc[:,~AUD.columns.duplicated()]

        BUC = pd.concat([B,C], axis=1)
        BUC = BUC.loc[:,~BUC.columns.duplicated()]

        BUE = pd.concat([B,E], axis=1)
        BUE = BUE.loc[:,~BUE.columns.duplicated()]

        CUE = pd.concat([C,E], axis=1)
        CUE = CUE.loc[:,~CUE.columns.duplicated()]

        AUC = pd.concat([A,C], axis=1)
        AUC = AUC.loc[:,~AUC.columns.duplicated()]

        AUE = pd.concat([A,E], axis=1)
        AUE = AUE.loc[:,~AUE.columns.duplicated()]

        BUD = pd.concat([B,D], axis=1)
        BUD = BUD.loc[:,~BUD.columns.duplicated()]

        CUD = pd.concat([C,D], axis=1)
        CUD = CUD.loc[:,~CUD.columns.duplicated()]

        DUE = pd.concat([D,E], axis=1)
        DUE = DUE.loc[:,~DUE.columns.duplicated()]

        union = [AUB,AUD,BUC,BUE,CUE,AUC,AUE,BUD,CUD,DUE]

        #intersection --> 10 new sets
        AnB = pd.concat([A,B], axis=1)
        AnB = AnB.loc[:,AnB.columns.duplicated()]

        AnD = pd.concat([A,D], axis=1)
        AnD = AnD.loc[:,AnD.columns.duplicated()]

        BnC = pd.concat([B,C], axis=1)
        BnC = BnC.loc[:,BnC.columns.duplicated()]

        BnE = pd.concat([B,E], axis=1)
        BnE = BnE.loc[:,BnE.columns.duplicated()]

        CnE = pd.concat([C,E], axis=1)
        CnE = CnE.loc[:,CnE.columns.duplicated()]

        AnC = pd.concat([A,C], axis=1)
        AnC = AnC.loc[:,AnC.columns.duplicated()]

        AnE = pd.concat([A,E], axis=1)
        AnE = AnE.loc[:,AnE.columns.duplicated()]

        BnD = pd.concat([B,D], axis=1)
        BnD = BnD.loc[:,BnD.columns.duplicated()]

        CnD = pd.concat([C,D], axis=1)
        CnD = CnD.loc[:,CnD.columns.duplicated()]

        DnE = pd.concat([D,E], axis=1)
        DnE = DnE.loc[:,DnE.columns.duplicated()]

        intersection = [AnB,AnD,BnC,BnE,CnE,AnC,AnE,BnD,CnD,DnE]

        #now have 25 individuals for this first generation (5 original, 10 unions, 10 intersections)
        #randomly mutate each of the 25 sets by either adding another feature, deleting a feature, or replacing
        #one feature with another. Each is mutated differently. :-)

        #let 0 = add, 1 = delete, and 2 = replace

        possible_features = ['z1','z2','z3','z4','sepal-length','sepal-width','petal-length','petal-width']
        all_individuals = original+union+intersection
        all_mutated_individuals = []


        #for each individual of the 25 'untampered'
        for i in range(len(all_individuals)):
            
            #create new instance of the table, so creating A_mut does not affect the original A variable
            ind_mut = all_individuals[i].copy()
            
            #determine whether to add, delete, or replace
            mutation = random.choice([0,1,2])
            mutation=0
            
            #if mutation == 0 & individual list is *not* full -- OR the individual list is empty -- add feature
            if ((mutation == 0) & (ind_mut.shape[1] != 8)) | (ind_mut.shape[1] == 0):
                
                #first convert column names of all_individuals[i] to list
                f = (ind_mut.columns).to_numpy()
                f = np.ndarray.tolist(f)

                #to find the features in possible_features but not all_individuals[i],
                features_not_in_individual = list(set(possible_features) - set(f))
                
                #choose one at random
                
                #if the features_not_in_individual list is not empty,
                if len(features_not_in_individual) != 0:
                    feature_name = random.choice(features_not_in_individual)
                
                #if ind_mut is empty, there is a problem-o. this list should not be empty in this loop
                if len(features_not_in_individual) == 0:
                    print('features_not_in_individual list is empty! Bzzzrt.')
                    break
                
                #add feature to the individual!
                mutated_individual = (pd.concat([ind_mut,iris_features_only[feature_name]], axis=1)).copy()
                
                #append to full mutated set
                all_mutated_individuals.append(mutated_individual)
            
            
            #if individual list is full, or mutation == 1, delete
            if (ind_mut.shape[1] == 8) | (mutation == 1):
                    
                #first convert column names of all_individuals[i] to list
                f = (ind_mut.columns).to_numpy()
                f = np.ndarray.tolist(f)

                #choose feature in ind_mut (or, in this case, f) at random
                feature_name = random.choice(f)
                
                #delete column from ind_mut
                mutated_individual = (ind_mut.drop(columns=[feature_name])).copy()
                
                #append to full mutated set
                all_mutated_individuals.append(mutated_individual)
                    
            
            #if mutation == 2, replace
            if mutation == 2:
                
                #first convert column names of all_individuals[i] to list
                f = (ind_mut.columns).to_numpy()
                f = np.ndarray.tolist(f)
                
                #determine features not already in individual
                #(a shame it would be to replace one feature with the same feature)
                features_not_in_individual = list(set(possible_features) - set(f))
            
                #select feature not in individual to use as replacement
                feature_replacement = random.choice(features_not_in_individual)
                
                #select feature at random to replace with feature_name!
                feature_to_replace = random.choose(f)
                
                #replacing...
                ind_mut[feature_to_replace] = iris_features_only[feature_replacement]
                mutated_individual = ind_mut.copy()
            
                #add mutated result to the full set
                all_mutated_individuals.append(mutated_individual)

        #I now have the original (all_individuals) and mutated sets (all_mutated_individuals).
        #NOW, we must evaluate! calculate the accuracy of the 50 sets using 2-fold cross-validation
        #remember: print five best feature sets and their corresponding accuracies, as well as the generation number

        #first, combine the sets (notice how I capitalized 'ALL' to emphasize that I merge both lists)
        ALL_individuals = all_individuals+all_mutated_individuals
        best_individual_list = []
        accuracy_list = []

        #for each individual,
        #   -perform 2-fold cross-validation
        #   -perform decisiontree classification model to determine accuracy
        #   -append accuracy values to accuracy_list
        for i in range(len(ALL_individuals)):

            #isolate data arrays, class array
            num_columns = ALL_individuals[i].shape[1]
            array = ALL_individuals[i].values
            x = array[:,0:num_columns]
            y = iris_dat.values[:,4]

            #create x and y folds
            X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(x, y, test_size=0.50, random_state=1)

            model=DecisionTreeClassifier()
            model.fit(X_Fold1, y_Fold1) #first fold training
            pred1 = model.predict(X_Fold2) #first fold testing
            model.fit(X_Fold2,y_Fold2) #second fold training
            pred2 = model.predict(X_Fold1) #second fold testing

            actual_classes = np.concatenate([y_Fold2,y_Fold1])
            predicted_classes = np.concatenate([pred1,pred2])
            accuracy_list.append(accuracy_score(actual_classes,predicted_classes))

        #we want the first five largest accuracies; one option is to create a zipped list with accuracies
        #and the associated indices, then sort this list (which rearranges from least to greatest accuracy)
        #and isolate the five zipped elements at the end.

        #create indices list from 0 to 50 (length of accuracy_list)
        indices = np.ndarray.tolist(np.arange(0,50,1))

        #zip lists
        a_zip = zip(accuracy_list,indices)
        zipped_list = list(a_zip)

        #rearrange from least to greatest accuracy
        zipped_list.sort()

        #last five elements are [45:50]
        largest_accuracies = zipped_list[45:50]

        #isolate index, append corresponding individual to best_individual_list
        for i in largest_accuracies:
            #isolate index
            index = i[1]
            best_individual_list.append(ALL_individuals[index])

        print('Generation %s'%str(n+1))
        print()
        for i in range(len(best_individual_list)):
            print('Individual %s'%str(i+1),(best_individual_list[i].columns).tolist())
            print('Accuracy:','%.3f'%largest_accuracies[i][0])
            
        #now, define the new A, B, C, D, and E
        A = best_individual_list[0]
        B = best_individual_list[1]
        C = best_individual_list[2]
        D = best_individual_list[3]
        E = best_individual_list[4]
        
        #if first generation, then best individual will be that with highest accuracy of the lot
        if n == 0:
            best_overall_individual = E.copy()
            best_overall_accuracy = largest_accuracies[4][0]
        
        #otherwise, compare individuals among generations to determine best 'fitness' of the lot
        else:
            if largest_accuracies[4][0] > best_overall_accuracy:
                
                best_overall_accuracy = largest_accuracies[4][0]
                best_overall_individual = (E.columns).tolist()
        
        print()
    
    print('After %s generations,'%str(n_gen))
    print('Best feature set:',best_overall_individual)
    print('Best accuracy:',best_overall_accuracy)



if __name__ == "__main__":
        
    homedir = os.getenv("HOME")
    iris_dat = read_table(homedir=homedir) #read iris data table
    
    #Part One: execute decisiontree model classification using the default features
    print('Part One')
    x,y = isol_array(iris_dat) #separate data into x (shape --> 150,4) and y (shape --> 150,1) columns
    X_Fold1,X_Fold2,y_Fold1,y_Fold2 = fold_data(x,y)
    train_data_dectree(X_Fold1, X_Fold2, y_Fold1, y_Fold2,print_dat=1)
    print()
    
    #Part Two: apply PCA feature transformation, then run decisiontree model classification
    print('Part Two')
    principleComponents,eigen_pairs,Z = pca_feature_selection(x_data=x)  #perform PCA
    Z_Fold1,Z_Fold2,y_Fold1,y_Fold2 = fold_data(Z,y)  #two-fold cross-validation
    train_data_dectree(Z_Fold1,Z_Fold2,y_Fold1,y_Fold2,print_dat=1)  #train/test on Z data
    print()
    
    #Part Three: apply simulated annealing feature selection, then run decisiontree model classification
    print('Part Three')
    simulated_annealing(x,y,principleComponents)
    print()
    
    print('Part Four')
    genetic_algorithm(iris_dat,50,principleComponents)
