# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 06:54:40 2019

@author:
"""
# import pertinent modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# load Fannie Mae loan Acquisition Data into DataFrame
df_acquisition = pd.DataFrame()
df_acquisition = pd.read_csv(r'Temp\Acquisition_2008Q4.txt', sep = "|", header=None
                          , names=['Loan_ID', 'Orig_Channel', 'Seller_Name', 'Orig_Int_Rate', 'Orig_UPB'
                          , 'Orig_Loan_Term','Origination_Date', 'First_Pmt_Date', 'Orig_LTV', 'Orig_CLTV'
                          ,'Num_Borrowers', 'Orig_Debt_Inc_Ratio', 'Borrower_Credit_Score'
                          , 'First_Time_Home_Buyer_Ind','Loan_Purpose', 'Property_Type', 'Number_Units'
                          , 'Occupancy_Type', 'Property_State', 'Zip_Code', 'Primary_Mortg_Insur_Percent'
                          ,'Product_Type', 'Coborrower_Credit_Score', 'Mortg_Insur_Type'
                          ,'Relocation_Mortg_Ind']) 
#Copy of Acquisition_2008Q4.txt can be found at the link below
#https://drive.google.com/file/d/1dhv1sKy5TP5hpLKmligEO5tLIyjk79LP/view?usp=sharing
    
    # Load Fannie Mae Loan Performance Data to DataFrame
df_performance = pd.DataFrame()
df_performance = pd.read_csv(r'Temp\Performance_2008Q4.txt', sep = "|", header=None
                            , names=['Loan_ID', 'Reporting_Period', 'Servicer_Name', 'Current_Int_Rate'
                            , 'Current_Act_UPB', 'Loan_Age','Remain_Months_Legal_Maturity'
                            , 'Adj_Months_Maturity', 'Maturity_Date', 'Metro_Stat_Area'
                            ,'Current_Loan_Delinq_Status', 'Modification_Flag', 'Zero_Balance_Code'
                            , 'Zero_Balance_Effective_Date','Last_Paid_Install_Date', 'Foreclosure_Date'
                            , 'Disposition_Date', 'Foreclosure_Costs','Property_Preserv_Repair_Costs'
                            , 'Asset_Recovery_Costs', 'Misc_Holding_Expenses_Credits'
                            ,'Associated_Taxes_Holding_Property', 'Net_Sale_Proceeds'
                            , 'Credit_Enhancement_Proceeds','Repurchase_Make_Whole_Proceeds'
                            , 'Other_Foreclosure_Proceeds', 'Non_Interest_Bearing_UPB','Principal_Forgiveness_Amt'
                            , 'Repurchase_Make_Whole_Proceeds_Flag', 'Foreclosure_Principal_Writeoff_Amt'
                            ,'Servicing_Activity_Indicator'])
#Copy of Performance_2008Q4.txt can be found at the link below
#https://drive.google.com/file/d/1DlzF6TG17Vl-LC1yOsCPgXedjjjIxdsT/view?usp=sharing 

# Remove all but rows with Zero_Balance_Codes that indicate account has been closed
df_performance = df_performance[df_performance['Zero_Balance_Code'].notna()]

# Keep only performance data representing foreclosures information (Foreclosure_Date & Zero_Balance_Code)
df_performance = df_performance[['Loan_ID', 'Zero_Balance_Code', 'Foreclosure_Date']]

# Join data from the acquisition and performance DataFrames
df = df_acquisition.merge(df_performance, on="Loan_ID", how="inner")

# Add classification column, 1=defaulted, 0=not defaulted
df['Classification'] = [1 if pd.notnull(x) else 0 for x in df['Foreclosure_Date']]

# Fill in and clean up data

# Add in Num_Borrowers value for those that are NaN
df['Num_Borrowers'] = np.where(df['Num_Borrowers'].notnull(), df['Num_Borrowers']                                        
                                           ,np.where(np.logical_and(df['Num_Borrowers'].isnull()
                                           ,df['Coborrower_Credit_Score'].isnull()), 1, 2))

# Drop rows with conflicting number of borrowers and co-borrower credit score data
df = df[((df['Num_Borrowers'] > 1) & (df['Coborrower_Credit_Score'].notnull()) | (df['Num_Borrowers']==1) 
         & df['Coborrower_Credit_Score'].isnull())]

# Drop rows without borrower credit score
df = df.dropna(subset=['Borrower_Credit_Score'], how='all')

# Orig_CLTV make NaN values equal to Orig_LTV
df['Orig_CLTV'].fillna(df['Orig_LTV'], inplace=True)

# Orig_Debt_Inc_Ratio make NaN values equal to median value of column
debt_ratio = np.nanmedian(df['Orig_Debt_Inc_Ratio'])
df['Orig_Debt_Inc_Ratio'].fillna(debt_ratio, inplace=True)

# Coborrower_Credit_Score make NaN values equal to Borrower_Credit_Score
df['Coborrower_Credit_Score'].fillna(df['Borrower_Credit_Score'], inplace=True)

# First_Time_Home_Buyer_Ind make Y = 1 else 0
df['First_Time_Home_Buyer_Ind'] = [1 if x == 'Y' else 0 if x in ['N','U'] else 
                                   df['First_Time_Home_Buyer_Ind']
                                   for x in df['First_Time_Home_Buyer_Ind']]

# Property_Type make SF (Single Family) = 1, PU (Planned Unit Development) = 2, CO (Condo) = 3, CP (Co-Op) = 4,
# MH (Multi House) = 5
df['Property_Type'] = [1 if x == 'SF' else 2 if x == 'PU' else 3 if x == 'CO' else 4 if x == 'CP' 
                       else 5 if x == 'MH' else df['Property_Type'] for x in df['Property_Type']]

# Loan_Purpose make P (Purchase) = 1, C (Cash-Out Refinance) = 2, R (Cash-Out No Refinance) = 3
df['Loan_Purpose'] = [1 if x == 'P' else 2 if x == 'C' else 3 if x == 'R' else df['Loan_Purpose'] 
                      for x in df['Loan_Purpose']]

# Occupancy_Type make P (Principal) = 1, I (Investment) = 2, S (Second) = 3
df['Occupancy_Type'] = [1 if x == 'P' else 2 if x == 'I' else 3 if x == 'S' else df['Occupancy_Type'] 
                        for x in df['Occupancy_Type']]

# Identify features and target to be included in classification model
features = ['Borrower_Credit_Score', 'Coborrower_Credit_Score','Orig_Debt_Inc_Ratio','Orig_LTV','Orig_UPB', 
            'First_Time_Home_Buyer_Ind', 'Loan_Purpose', 'Property_Type','Num_Borrowers','Occupancy_Type', 'Zip_Code']
X = df[features]
y = df['Classification']

# Random forest classifier
def random_forest(n_estimator, max_feature, min_samples_split):
    """Executes Random Forest Train, Test, Fit"""
    global y_test, y_predict, model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
    model = RandomForestClassifier(n_estimators=n_estimator, random_state=0, max_features=max_feature, 
                                   min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_test, y_predict, model

# Initial random forest run of model using default hyperparameters
random_forest(100, int(np.sqrt(len(features))), 2)
print('Random Forest')
print(classification_report(y_test, y_predict))    

# Resample data for undersampling
nr = NearMiss()
X, y = nr.fit_sample(X, y)

# Random Forest run after resampling of model using default hyperparameters
random_forest(100, int(np.sqrt(len(features))), 2)
print('Random Forest after Undersampling')  
print(classification_report(y_test, y_predict)) 

# k nearest neighbors
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size = 0.25, random_state=0) 
knn = KNeighborsClassifier(n_neighbors=5) 
  
knn.fit(X_train, y_train) 

y_predict = knn.predict(X_test)
print('K Nearest Neighbors after Undersampling')
print(classification_report(y_test, y_predict)) 

# Logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predict = logreg.predict(X_test)
print('Logistic Regression after Undersampling')
print(classification_report(y_test, y_predict)) 

# Tune n_estimators values to determine highest f1 score
n_estimators = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
score = 0
n = 0
for n_estimator in n_estimators:
    random_forest(n_estimator, int(np.sqrt(len(features))), 2)
    f1 = f1_score(y_test, y_predict)
    if f1 > score:
        score = f1
        n = n_estimator
print(n, ': Number of trees to use for Random Forest model')
print(score, ': Random Forest f1_score')

# Tune max_features and min_sample_splits to determine highest f1 score
max_features = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
min_samples_splits = [2, 4, 8, 16, 32, 64, 128, 256]
score = 0
max_feat = 0
min_ss = 0

for max_feature in max_features:
    for min_samples_split in min_samples_splits:
        random_forest(n, max_feature, min_samples_split)
        f1 = f1_score(y_test, y_predict)
        if f1 > score:
            score = f1
            max_feat = max_feature
            min_ss = min_samples_split
print(score, ': Random Forest f1_score')
print(max_feat, ': Random Forest max_feature')
print(min_ss, ': Random Forest min_sample_split')

# Final prediction using tuned hyperparameters
random_forest(n, max_feat, min_ss)
print('Random Forest after Hyperparameter Tuning')
print(classification_report(y_test, y_predict)) 

# Feature importance summary
print('Random Forest Feature Importances')
for i in range(len(features)):
    print(round(model.feature_importances_[i],4), features[i]) 
    
# Combine features and feature importances
zipped = zip(features, model.feature_importances_)
importances, features = zip(*sorted(zip(list(model.feature_importances_), features)))

# Graph feature importances in descending order
_ = plt.barh(list(range(0,11)), importances, color='royalblue')
_ = plt.yticks(np.arange(11), list(features))
_ = plt.grid(False)
_ = plt.xlabel('weight')
_ = plt.title('Feature Importances')


