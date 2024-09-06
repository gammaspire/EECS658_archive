'''
(Note: 2-fold cross validation not used here --> k-Means Clustering is an unsupervised model!)

============

This python program executes the following:
--reads iris-data table; respectively assigns features and classes to variables x and y
--shuffles data such that classes are not stratified (i.e., iris-setosa are the first 50 rows, iris-versicolor are the next 50 rows, usw.)
--runs k-means clustering 20 times, plots reconstruction err vs. k value and algorithmically identifies the elbow (kneed pkg)
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

from sklearn.cluster import KMeans
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


def find_elbow(x,y,homedir,k_min=1,k_max=20):
    
    #location to save inertia v. k-value plot
    path_to_plot = homedir+'/Desktop/KimConger_Assignment6/'
    figname = 'inertia_plot.png'
    
    #create empty inertia list
    inertia=[]
    
    #create possible k-values
    k_values = np.arange(k_min,k_max+1,1)
    
    #populate inertia and k_values lists
    for k in k_values:
        kmeans_inertia = (KMeans(n_clusters=k, init='k-means++').fit(x)).inertia_
        inertia.append(kmeans_inertia)
    


    #I initially attempted to find the elbow algorithmically through the use of slopes calculated between k = 20 and k != 20 (starting from 19, then 18, ...) points, and in particular find the slope started to deviate *significantly* from that between k=20 up to k~8. The trouble was that I was not able to define this transition point so rigorously that I could systematically apply it to any curve for which I wanted to extract the elbow. I then tried R^2 values and std_errs with this same comparison technique, but it was a bit tricky to convincingly claim that k=3 was more gooder better good than k=4.
    
    #in lieu of this sorrowful effort, I found a good ol' python package. 
    kneedle = KneeLocator(k_values, inertia, S=1.0, curve="convex", direction="decreasing")
    elbow_k = kneedle.elbow
    
    #create, save inertia plot
    plt.figure(figsize=(10,6))
    plt.plot(k_values,inertia,'black',alpha=0.4)
    plt.scatter(k_values,inertia,color='purple',s=80)
    #add elbow marker
    plt.plot(elbow_k, inertia[elbow_k-1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="none",linestyle="none",label='Kneed Elbow')
    #change ticks to integer values
    locs,labels=xticks()
    xticks(np.arange(k_min, k_max+1, step=1))
    plt.xlabel('K Value',fontsize=20)
    plt.ylabel('Reconstruction Error',fontsize=20)
    plt.grid(alpha=0.2, color='purple')
    plt.legend(loc='upper right',fontsize=18)
    plt.savefig(path_to_plot+figname,dpi=300)
    plt.close()
    
    return(elbow_k)
    
    
def KMeans_predict(x,y,k_value):
    
    #define model
    model = KMeans(n_clusters=k_value, init='k-means++').fit(x)

    #predict() method to classify the entire iris dataset
    #(integers 0<i<k-1)
    #note --> in this case, prediction = model.labels_
    #(but I use the same syntax as the lecture slides for consistency purposes)
    prediction = model.predict(x)
    k_labels = prediction
    
    #create DataFrame with labels and varieties as columns (df)
    df = pd.DataFrame({'Labels': y, 'Clusters': k_labels})
    
    #create crosstab: ct (effectively serves as the confusion matrix)
    ct = pd.crosstab(df['Labels'], df['Clusters'])
    
    #Occasionally when run, model output will be such that the resulting ct has a column-row mismatch. For instance, if the model assigns integer 1 most frequently to Iris-setosa, then the correct predictions will be the second element of the first row (as opposed to the first element of the first row --> the diagonal). I simply create a new matrix to correct any inconsistencies.
    #EXAMPLE: We can assume that the majority of samples that have a true classification "A" will belong to the same cluster "1". For row "A" in the confusion matrix, I find the column corresponding to the largest number; then rearrange that column such that the column index of that largest number equals its row index (that is, the largest number will occupy the diagonal). If row_index = column_index already, then no changes occur.
    
    #isolate 3x3 numpy array with the original ct values
    ct_vals = (ct.values).copy()
    
    #create empty 3x3 matrix that will serve as the canvas for the new matrix
    new_ct = np.zeros([3,3])

    #extract the column index of the largest value in rows 1 (iris-setosa), 2 (iris-versicolor), and 3 (iris-virginica)
    index_row1 = np.where(ct_vals[0] == np.max(ct_vals[0]))
    index_row2 = np.where(ct_vals[1] == np.max(ct_vals[1]))
    index_row3 = np.where(ct_vals[2] == np.max(ct_vals[2]))

    #place nth column with whichever column's largest value lies at row_index = col_index
    new_ct[:,0] = ct_vals[:,index_row1[0][0]]
    new_ct[:,1] = ct_vals[:,index_row2[0][0]]
    new_ct[:,2] = ct_vals[:,index_row3[0][0]]
        
    print('Confusion Matrix')
    print(new_ct)
    
    #calculate accuracy score
    diagonal_sum = np.trace(new_ct)
    tot_num_samples = len(y)
    accuracy = diagonal_sum / tot_num_samples
    print('accuracy score:',accuracy)
    

if __name__ == "__main__":
        
    homedir = os.getenv("HOME")
    iris_dat = read_table(homedir=homedir) #read iris data table
    
    #Part One: K-Means Clustering
    print('~~~~~~~~~~Part One~~~~~~~~~~')
    print()
    x,y = isol_array(iris_dat) #separate data into x (shape --> 150,4) and y (shape --> 150,1) columns
    x,y = shuffle_data(x,y) #shuffle data!
    elbow_k = find_elbow(x,y,homedir=homedir)
    print('KMeans run with elbow_k =',elbow_k, '(found algorithmically)')
    KMeans_predict(x,y,elbow_k)
    print()
    print('KMeans run with k = 3')
    KMeans_predict(x,y,3)
    print()

