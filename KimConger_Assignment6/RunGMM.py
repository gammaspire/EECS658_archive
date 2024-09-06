'''
(Note: 2-fold cross validation not used here --> k-Means Clustering is an unsupervised model!)

============

This python program executes the following:
--reads iris-data table; respectively assigns features and classes to variables x and y
--shuffles data such that classes are not stratified (i.e., iris-setosa are the first 50 rows, iris-versicolor are the next 50 rows, usw.)
--runs GMM (Gaussian Mixture Model) 20 times, plots AIC/BIC vs. k value and algorithmically identifies the elbow (kneed pkg)
--isolates model label column (integers) and compares with 'true' class column (strings) --> 
            # Create a DataFrame with labels and varieties as columns: df
            df = pd.DataFrame({'Labels': labels, 'Clusters': k_labels})
            # Create crosstab: ct
            ct = pd.crosstab(df['Labels'], df['Clusters'])
--ct is then the relevant confusion matrix. For accuracy_score, calculate the sum of the diagonal and divide by total number of samples

============

OUTPUT:
---- Confusion matrix (printed)
---- Accuracy metric (printed)
-----k clusters representing the elbow (printed)
-----.png of reconstruction_err vs. k_value plot

EXAMPLE: 
in python notebook, run
%run path/to/program/RunKMeansClustering.py
'''


from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks
import numpy as np
import pandas as pd
from pandas import read_csv
import os

from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
from kneed import KneeLocator


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


def shuffle_data(x,y):
    return shuffle(x,y,random_state=42)


def find_elbows(x,y,homedir,k_min=1,k_max=20):
    
    #location to save inertia v. k-value plot
    path_to_plot = homedir+'/Desktop/KimConger_Assignment6/'
    figname = 'GMM_plot.png'
    
    #create possible k-values
    k_values = np.arange(k_min,k_max+1,1)
    
    #create AIC, BIC arrays
    AIC = np.zeros(k_values.shape)
    BIC = np.zeros(k_values.shape)
    
    #populate inertia and k_values lists
    for k in k_values:
        GMM_model = GaussianMixture(n_components=k,covariance_type='diag').fit(x)
        AIC[k-1] = GMM_model.aic(x)
        BIC[k-1] = GMM_model.bic(x)
    
    kneedle_aic = KneeLocator(k_values, AIC, S=1.0, curve="convex", direction="decreasing")
    kneedle_bic = KneeLocator(k_values, BIC, S=1.0, curve="convex", direction="decreasing")
    elbow_k_aic = kneedle_aic.elbow
    elbow_k_bic = kneedle_bic.elbow
    
    #create, save inertia plot
    plt.figure(figsize=(10,6))
    
    #AIC
    plt.plot(k_values,AIC,'black',alpha=0.4)
    plt.scatter(k_values,AIC,color='purple',s=80,label='AIC')
    
    plt.plot(elbow_k_aic, AIC[elbow_k_aic-1], marker='o', markersize=20, markeredgecolor='blue', markerfacecolor='none', linestyle='none', label='Kneed Elbow (AIC)')
    
    #BIC
    plt.plot(k_values,BIC,'black',alpha=0.4)
    plt.scatter(k_values,BIC,color='green',s=80,label='BIC')
    
    plt.plot(elbow_k_bic, BIC[elbow_k_bic-1], marker='o', markersize=20, markeredgecolor='red', markerfacecolor='none', linestyle='none', label='Kneed Elbow (BIC)')
    
    plt.xlabel('k-value',fontsize=20)
    plt.ylabel('Information Criterion',fontsize=20)
    
    #change ticks to integer values
    locs,labels=xticks()
    xticks(np.arange(k_min,k_max+1,step=1))
    plt.grid(alpha=0.2,color='purple')
    plt.legend(loc='upper right',fontsize=15)

    plt.savefig(path_to_plot+figname,dpi=300)
    plt.close()
    
    return(elbow_k_aic, elbow_k_bic)
    
    
def GMM_predict(x,y,k_value):
    
    #define model
    model = GaussianMixture(n_components=k_value,covariance_type='diag').fit(x)

    #predict() method to classify the entire iris dataset with integers from 0 to k-1
    k_clusters = model.predict(x)
    
    #create DataFrame with labels and varieties as columns (df)
    df = pd.DataFrame({'Labels': y, 'Clusters': k_clusters})
    
    #create crosstab ct (effectively serves as the confusion matrix)
    ct = pd.crosstab(df['Labels'], df['Clusters'])
    
    #Occasionally when run, model output will be such that the resulting ct has a column-row mismatch. For instance, if the model assigns integer 1 most frequently to Iris-setosa, then the correct predictions will be the second element of the first row (as opposed to the first element of the first row --> the diagonal). I simply create a new matrix to correct any inconsistencies.
    #EXAMPLE: We can assume that the majority of samples that have a true classification "A" will belong to the same cluster "1". For row "A" in the confusion matrix, I find the column corresponding to the largest number; then rearrange that column such that the column index of that largest number equals its row index (that is, the largest number will occupy the diagonal). If row_index = column_index already, then no changes occur.
    
    #isolate k x k numpy array with the original ct values
    ct_vals = (ct.values).copy()
    
    #if k>3, then append a row of zeros (converts 3x4 to kx4 matrix)
    if k_value > 3:
        ct_vals = np.append(ct_vals,np.zeros(k_value)).reshape(k_value,k_value)
    
    #create empty k x k matrix that will serve as the canvas for the new matrix
    new_ct = np.zeros(ct_vals.shape)
    
    #extract the column index of the largest value in rows 1 (iris-setosa), 2 (iris-versicolor), and 3 (iris-virginica)...or some cluster class 4
    #begin with creating empty indices array
    indices = np.zeros(k_value)
    
    for i in range(k_value):
        
        index_i = np.where(ct_vals[i] == np.max(ct_vals[i]))
        index_i = index_i[0]
        indices[i] = int(index_i[0])
        #in case k=4, then define index for the fourth row of ct_vals matrix (whichever index is remaining after rows 0, 1, 2 are arranged). if k != 4, then this variable will ultimately be ignored
        if i == 3:
            #create list of all possible indices
            k_values = np.arange(0,k_value,1)
            #isolate whichever index has not yet been used
            index_i = list(set(k_values) - set(indices))
            indices[i] = int(index_i[0])
        
    #place nth column with whichever column's largest value lies at row_index = col_index
    
    for i in range(k_value):
        new_ct[:,i] = ct_vals[:,int(indices[i])]
        
    print('Confusion Matrix')
    print(new_ct)
    
    #calculate accuracy score (only of number of clusters does not exceed the number of classes (np.unique(y).shape[0], or len(np.unique(y)))
    if k_value > len(np.unique(y)):
        print('# clusters > # classes, so conventional accuracy_score not available.')
    else:
        diagonal_sum = np.trace(new_ct)
        tot_num_samples = len(y)
        accuracy = diagonal_sum / tot_num_samples
        print('accuracy score:',accuracy)
    

if __name__ == "__main__":
        
    homedir = os.getenv("HOME")
    iris_dat = read_table(homedir=homedir) #read iris data table
    
    #Part One: Gaussian Mixture Model
    print('~~~~~~~~~~Part Two~~~~~~~~~~')
    print()
    x,y = isol_array(iris_dat) #separate data into x (shape --> 150,4) and y (shape --> 150,1) columns
    x,y = shuffle_data(x,y) #shuffle data!
    elbow_k_aic, elbow_k_bic = find_elbows(x,y,homedir=homedir)
    print('GMM run with AIC elbow_k =',elbow_k_aic, '(found algorithmically)')
    GMM_predict(x,y,elbow_k_aic)
    print()
    print('GMM run with BIC elbow_k =',elbow_k_bic, '(found algorithmically)')
    GMM_predict(x,y,elbow_k_bic)
    print()
    print('GMM run with k = 3')
    GMM_predict(x,y,3)
    print()

