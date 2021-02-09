# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:18:52 2020

@author: akshay
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score 
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp as ind
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score 

from keras.models import Sequential 
from keras.layers import Dense

data = pd.read_csv('./Data/pp5i_train.gr.csv') 
test_data = pd.read_csv('./Data/pp5i_test.gr.csv')
serial_nos = data.loc[:, "SNO"]

# Thresholding train data between 20 and 16000
data = data.drop("SNO", axis = 1)
data[data < 20] = 20
data[data > 16000] = 16000
mod_data = data

# Thresholding test data between 20 and 16000
test_data = test_data.drop("SNO", axis = 1)
test_data[test_data < 20] = 20
test_data[test_data > 16000] = 16000

# Extracting min max values of each row in the given dataset
max_vals = np.max(mod_data, axis = 1)
min_vals = np.min(mod_data, axis = 1)
print("\n Max values: ", max_vals)
print("\n Min values: ", min_vals)
size = data.shape[0]

# Removing samples with fold difference less than 2
count = 0
for row in np.arange(0,size):
    if max_vals[row]/min_vals[row] < 2:
        mod_data = mod_data.drop(row, axis = 0)
        serial_nos = serial_nos.drop(row)
        count+=1
        
# Extracting Classes from pp5i_train_class.txt
file = open('./Data/pp5i_train_class.txt')
file_data = file.read()
classes = file_data.splitlines()
classes.pop(0)
class_list = set(classes)

print(mod_data, serial_nos)    
mod_data.to_csv('folded.csv', index = False)
mod_data = pd.read_csv('./folded.csv') 


# Dividing list of data based on classes and storing them as a list with class
# as variable name
count = 0
for item in class_list:
    count+=1
    indices = [i for i, s in enumerate(classes) if item in s]
    var = 'data' + str(count)
    globals()[var] = mod_data.iloc[:,indices]
    print("Data of "+item+": \n", globals()[var])
    cls_mean = abs(globals()[var].mean(axis = 1))
    pop_mean = abs(cls_mean.mean(axis = 0))
    t_val = ind(globals()[var], pop_mean, axis = 1)
    t_val = t_val.statistic
    globals()[item+"_ind"] = np.argsort(t_val)[-30:]
    globals()[item] = globals()[var].loc[globals()[item+"_ind"],:]
    print("Indices of top 30 Gene data for class ",item,":---------------------\n", globals()[item+"_ind"], "\n---------------------")    


# Prepares a list of classes for top N best genes per class data
def prep_cls_list_top():
    cls = []
    for item in class_list:
        for i in np.arange(0,globals()[item].shape[1]): cls.append(item)
    return cls

# Prepares the list of top N best genes per class data
def prep_top_list_mod_data(mod_data, lp_range, class_list):
    indices = np.concatenate((RHB_ind[:lp_range].ravel(), MGL_ind[:lp_range].ravel(), JPA_ind[:lp_range].ravel(), MED_ind[:lp_range].ravel(), EPD_ind[:lp_range].ravel()), axis = None)
    indices = list(dict.fromkeys(indices))
    top = mod_data.iloc[indices,:].to_numpy()
    cls = prep_cls_list_top()
    top = np.append(top, [cls], axis = 0)
    final_top = pd.DataFrame(top)
    final_top.rename({final_top.index[-1]: "Class"}, inplace=True)
    return final_top

# Prepares the test data based on the cross validation error of the best gene train set
def prep_top_list_test_mod_data(test_data, lp_range, class_list):
    indices = np.concatenate((RHB_ind[:lp_range].ravel(), MGL_ind[:lp_range].ravel(), JPA_ind[:lp_range].ravel(), MED_ind[:lp_range].ravel(), EPD_ind[:lp_range].ravel()), axis = None)
    indices = list(dict.fromkeys(indices))
    top = test_data.iloc[indices,:]
    return top




# Prepares a list of top N values
def prep_top_list(lp_range, class_list) :
    top = []
    cls = []
    for i in np.arange(0,lp_range) :
        RHB_dat = RHB.iloc[i,:].ravel()
        MGL_dat = MGL.iloc[i,:].ravel()
        JPA_dat = JPA.iloc[i,:].ravel()
        MED_dat = MED.iloc[i,:].ravel()
        EPD_dat = EPD.iloc[i,:].ravel()
        prep = np.concatenate((RHB_dat, MGL_dat, JPA_dat, MED_dat, EPD_dat), axis=None)
        top.append(prep)
    cls = prep_cls_list_top()
    top.append(cls)
    final_top = pd.DataFrame(top)
    final_top.rename({final_top.index[-1]: "Class"}, inplace=True)
    return final_top

acc_obj = {'GNB': [], 'DTC': [], 'KNN2': [], 'KNN3': [], 'KNN4': [], 'ABC': [], 'MLP': [], 'cvrows': [2, 4, 6, 8, 10, 12, 15, 20,
25, 30]}    

# Extracts data from the corresponding filepath and feeds it to the classifiers.
def extract_data(filepath) :
    data = pd.read_csv(filepath)
    X = data.loc[:, data.columns != 'Class'].to_numpy()
    Y = data.loc[:, 'Class'].to_numpy()
    
    gnb = GaussianNB();
    acc_score = work_on_classifier(gnb, X, Y)
    acc_obj['GNB'].append(acc_score)
    
    dtc = tree.DecisionTreeClassifier();
    acc_score = work_on_classifier(dtc, X, Y)
    acc_obj['DTC'].append(acc_score)
    
    for i in np.arange(0,3) :
        n = 2 + i
        knc = KNeighborsClassifier(n_neighbors= n);
        acc_score = work_on_classifier(knc, X, Y)
        acc_obj['KNN'+str(n)].append(acc_score)
    
    rfc = AdaBoostClassifier()
    acc_score = work_on_classifier(rfc, X, Y)
    acc_obj['ABC'].append(acc_score)
    
    mlp = MLPClassifier()
    acc_score = work_on_classifier(mlp, X, Y)
    acc_obj['MLP'].append(acc_score)
    
# Works with classifier on the data and sends its accuracy    
def work_on_classifier(obj, X, y) :
    obj.fit(X,y)
    accuracy = cross_val_score(obj, X, y, cv=5, scoring="accuracy")
    return np.round(accuracy.std(), 3)
    
def work_on_neural_net(X_train, X_test, y_train, y_test, shape) :
    
    model = Sequential([Dense(32, activation='relu', input_shape=(shape,)), Dense(32, activation='relu'), Dense(1, activation='sigmoid'),])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
    out = model.evaluate(X_test, y_test)[1]
    print("Accuracy of neural network: ", out)
    
    
# Prepares top N genes transposed csv from the extracted top 30 genes for each class.
top2 = prep_top_list_mod_data(mod_data, 2, class_list).T
top2.to_csv('pp5i_train.top2.gr.csv', index = False)
extract_data('./pp5i_train.top2.gr.csv')

top4 = prep_top_list_mod_data(mod_data, 4, class_list).T
top4.to_csv('pp5i_train.top4.gr.csv', index = False)
extract_data('./pp5i_train.top4.gr.csv')

top6 = prep_top_list_mod_data(mod_data, 6, class_list).T
top6.to_csv('pp5i_train.top6.gr.csv', index = False)
extract_data('./pp5i_train.top6.gr.csv')

top8 = prep_top_list_mod_data(mod_data, 8, class_list).T
top8.to_csv('pp5i_train.top8.gr.csv', index = False)
extract_data('./pp5i_train.top8.gr.csv')

top10 = prep_top_list_mod_data(mod_data, 10, class_list).T
top10.to_csv('pp5i_train.top10.gr.csv', index = False)
extract_data('./pp5i_train.top10.gr.csv')

top12 = prep_top_list_mod_data(mod_data, 12, class_list).T
top12.to_csv('pp5i_train.top12.gr.csv', index = False)
extract_data('./pp5i_train.top12.gr.csv')

top15 = prep_top_list_mod_data(mod_data, 15, class_list).T
top15.to_csv('pp5i_train.top15.gr.csv', index = False)
extract_data('./pp5i_train.top15.gr.csv')

top20 = prep_top_list_mod_data(mod_data, 20, class_list).T
top20.to_csv('pp5i_train.top20.gr.csv', index = False)
extract_data('./pp5i_train.top20.gr.csv')

top25 = prep_top_list_mod_data(mod_data, 25, class_list).T
top25.to_csv('pp5i_train.top25.gr.csv', index = False)
extract_data('./pp5i_train.top25.gr.csv')

top30 = prep_top_list_mod_data(mod_data, 30, class_list).T
top30.to_csv('pp5i_train.top30.gr.csv', index = False)
extract_data('./pp5i_train.top30.gr.csv')     

cvrows = acc_obj['cvrows']

cv_errs = pd.DataFrame(acc_obj, index = cvrows)
cv_errs = cv_errs.drop('cvrows', axis = 1)
print("Cross validation erros for all the classifiers with different best gene datasets\n", cv_errs)
for col in cv_errs:   
    cv_errs[col].plot(figsize = (10,10), legend = col,marker='o')
    plt.xticks([2,4,6,8,10,12,15,20,25,30])
    name = str(col) + '.png'
    plt.savefig(name)
    plt.show()

print("\n---------------- Considering KNN with 3 n_neighbors as the best classifier as it has got the lowest cross validation error ----------------")

print("\nPreparing the test data and predicting it using KNN classifier\n")

final_test_data = prep_top_list_test_mod_data(test_data, 25, class_list).T
final_test_data.to_csv('pp5i_test.top2.gr.csv', index = False)

# extracting best train data and preparing x_train and y_train
train_data = pd.read_csv('./pp5i_train.top25.gr.csv')
X_train = train_data.loc[:, train_data.columns != 'Class'].to_numpy()
y_train = train_data.loc[:, 'Class'].to_numpy()

# extracting best test data
test_data = pd.read_csv('./pp5i_test.top2.gr.csv')

# Predicting the classes for all the 23 rows in the test data
clf = KNeighborsClassifier(n_neighbors= 3, algorithm = 'ball_tree', 
                           weights = 'distance', leaf_size = 20, n_jobs = 10, 
                           p = 1, metric = 'euclidean');
clf.fit(X_train, y_train)
pred = clf.predict(test_data)
print("Prediction for the test Data: \n", pred)

