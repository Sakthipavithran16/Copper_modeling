# Copper Modeling Predictions

## Introduction

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions.

## Problem Statement 
A machine learning regression model can address these above issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 
A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer.

## Approach

### Data Understanding
Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null. Treat reference columns as categorical variables.

### Data Preprocessing
Handle missing values with mean/median/mode.
Treat Outliers using IQR or Isolation Forest from sklearn library.
Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation for continuous variables.
Encode categorical variables using suitable techniques

### EDA
Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot.

### Feature Engineering
Drop highly correlated columns using SNS HEATMAP.

### Model Building and Evaluation
Split the dataset into training and testing/validation sets. 
Train and evaluate different classification models, such as RandomForestClassifier, XGBClassifier and Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. 
Optimize model hyperparameters using techniques such as grid search to find the best-performing model.
Use pickle module to dump and load models such as encoder, scaling models, ML models.
Perform same steps for Regression modelling.

### Streamlit UI
Using streamlit module,create an input field where you can enter each column value except ‘Selling_Price’ for regression model and  except ‘Status’ for classification model. 
Perform the same feature engineering, scaling factors, log transformation steps using loaded models which is used for training ml model  
Predict the new data from streamlit and display the output.

### Tools used
1. Python
2. Numpy
3. Pandas
4. Streamlit
5. Seaborn
6. Matplotlib
7. Sklearn

### Key skills
1. Data preprocessing techniques
2. EDA
3. Machine learning techniques such as Regression and Classification
4. Hyper parameter tuning to optimize ML models
5. Web application using the Streamlit 

### Streamlt Overview


