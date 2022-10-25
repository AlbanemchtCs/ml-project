# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:35:39 2022

@author: Admin
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('../')
os.chdir('preprocessing/data_cleaning')
from data_cleaning import DataCleaning
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression



class SpecialKNN:
    def __init__(self,n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.model = XGBRegressor()
        
    def fit(self,X_knn,X_tree,y):
        
        
        X_knn = np.array(X_knn)
        self.X_tree = np.array(X_tree)
        self.y = np.array(y)
        self.model_knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='brute').fit(X_knn)
        
        
        distances, indices = self.model_knn.kneighbors(X_knn)
        
        new_X = np.empty((len(self.X_tree),self.n_neighbors,len(self.X_tree[0])+1))
        new_y = np.empty((len(self.X_tree),self.n_neighbors))
        nb_final = len(self.X_tree)*self.n_neighbors
        for i in range(len(self.X_tree)):
            new_X[i] = np.concatenate((self.X_tree[i] - self.X_tree[indices[i]],self.y[indices[i]].reshape((self.n_neighbors,1))),axis=1)
            new_y[i] = self.y[i]# - self.y[indices[i]])/self.y[indices[i]]
        self.model.fit(new_X.reshape((nb_final,len(self.X_tree[0])+1)),new_y.reshape((nb_final)))
        self.feature_importances_ = self.model.feature_importances_
    def predict(self,X_knn,X_tree):
        X_knn = np.array(X_knn)
        X_tree = np.array(X_tree)
        distances, indices = self.model_knn.kneighbors(X_knn)
        
        new_X = np.empty((len(X_tree),self.n_neighbors,len(X_tree[0])+1))
        nb_final = len(X_tree)*self.n_neighbors
        
        for i in range(len(X_tree)):
            new_X[i] =np.concatenate((X_tree[i] - self.X_tree[indices[i]],self.y[indices[i]].reshape((self.n_neighbors,1))),axis=1)
        y = self.model.predict(new_X.reshape((nb_final,len(X_tree[0])+1)))
        y = y.reshape((len(X_tree),self.n_neighbors))
        final_y  = np.zeros(len(y))
        
        for i in range(len(y)):
            y[i] = y[i]#*self.y[indices[i]]+self.y[indices[i]]
            pond = 1/(np.sum(abs(new_X[i]/self.model.feature_importances_),axis=1)+1)
            final_y[i] = sum(pond*y[i])/sum(pond)
            #final_y[i] = sum(y[i])/self.n_neighbors
        return final_y

    
    

# model = SpecialKNN(50)

# data_cleaner = DataCleaning(path="C:/Users/Admin/Documents/ML project/ml-project/data/train_airbnb_berlin.csv")
# data_cleaner.data_cleaning(csv_name="train_airbnb_berlin_cleaned.csv")
# df = data_cleaner.df

# train,test = train_test_split(df,random_state=0)
# F = ['Beds','Bathrooms','Bedrooms','Accomodates','Guests Included','Room Type_Entire home/apt','Property Type_Apartment']
# B = ['Longitude','Latitude']

# model.fit(train[B],train[F],train['Price'])

# y = model.predict(test[B],test[F])

# er= y - test['Price']
# error = abs(er)
# print(np.mean(error))
# print(np.mean(error/test['Price'])*100)
# #plt.hist(er,bins=40)

# plt.bar(F+['Price'],model.feature_importances_)
# plt.xticks(rotation=90)