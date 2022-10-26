# **Airbnb Berlin Price Prediction Projet**

The project aims to predict the price of a night's accommodation offered on the short-term rental website Airbnb in the city of Berlin. This project is derived from a data challenge proposed by dphi.tech.

<center><img src="media/berlin.jpg" alt="Berlin" width="600"/><center>

**Table of contents:**
1. Folders organization
2. Data analysis & Preprocessing
3. Models

## **1. Folders organization**

- [first approach model](https://gitlab.com/charlotte_sasson/ml-project/-/tree/main/first%20approach%20model) ⇒ folder for our first approach with the model XGBoost. Our first test approach doesn't use the same preprocessing. The folder contains the model xgb.ipynb, two files of utilities and a README.md 
    * we decided to perform a first model (XGBoost), as we know from experience, in tabular competitions, XGB generally performed better than other models (e.g. RandomForest)

    * this first approach enables us to have a first look on our data 
    
    * we have then performed a new preprocessing, due to our discoveries (folder utilities) that will be used for every models we are going to run
<br>

- [media](https://gitlab.com/charlotte_sasson/ml-project/-/tree/main/media) ⇒ contains all photos and graphs


- [models](https://gitlab.com/charlotte_sasson/ml-project/-/tree/main/models) ⇒ contains four files (trees & boosting, k-nn, k-nn & tree and special_knn) with all the models we ran


- [preprocessing](https://gitlab.com/charlotte_sasson/ml-project/-/tree/main/preprocessing) ⇒ data_cleaning and dimensionality_reduction processes
    

- [utilities](https://gitlab.com/charlotte_sasson/ml-project/-/tree/main/utilities) ⇒ data and results analysis

## **2. Data analysis & Preprocessing**

<img src="media/correlation.png" alt="Pearson's correlation" width="600"/>

We looked at the Pearson’s Coefficients of all features separately regarding their correlation with the feature *Price*. What we obtained is that *Accomodates* seems to be the most linearly correlated with *Price*.

### **Preprocessing Strategy**

<img src="media/preprocessing.png" alt="Preprocessing" width="1000"/>

### **Dimensionality Reduction**

| Fowards Selection |  | 
|---|---|---|---|
| Bedrooms | 54.616082 | 
| Room Type_Entire home/apt | 25.079846 | 
| Bathrooms | 83.915373 | 
| Accomodates | 74.884845 | 
| dist_to_center | -43.948291 | 
| Property Type_Loft | 27.289468 | 
| Guests included | 42.534146 | 
| Property Type_Hotel | 90.443688 |
| Property Type_Serviced apartment | 43.127653 | 
| Host Response Time | -6.122382 | 
| Neighborhood Group_Charlottenburg-Wilm. | 13.186426 | 

<img src="media/variance_pca.png" alt="Variance PCA" width="400"/>

* We have studied Best/Forward/Backward selection & PCA for dimensionality reduction. After studying the four methods, we decide to
to use **Forward and Backward** stepwise selections and to reduce our dataset to **11 variables**
* PCA has allowed us to justify our dimensional reduction to a number of variables because these variables allow us to justify more than 80% of the variance

## **3. Models**

| Models | MSE | MAE | RAE | 
|---|---|---|---|
| Decision Tree | X | X |  X |  
| Random Forest | X | X |  |  
| Bagging | X | X |  |  
| Gradient Boosting | X | X |  |  
| AdaBoost | X | X | X |  
| XGBoost | X | X | X | 
| K-NN | X | X | X | 
| K-NN & Tree | X | X |  
| K-NN & XGBoost | X | X |  
