import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import sys


from modules import correlation_approach, rbf_kernel_approach, xgboost_approach, split_and_evaluate, gpr_approach, knn_approach


test_data = pd.read_csv('test_set.csv', index_col=0)
train_data = pd.read_csv('train_set.csv', index_col=0)


selected_features =['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',  'feat_6','feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22',  'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29']

train_data= train_data.fillna(0)
train_data= train_data[['target'] +selected_features]
test_data= test_data[selected_features]

X_train = train_data.drop(columns=['target'])
y_train = train_data['target']



correlation_matrix = train_data.corr()
correlation = correlation_matrix['target'].iloc[1:]

result_array = gpr_approach(test_data, X_train, y_train)

result_df_rbf = pd.DataFrame(result_array, columns=["ID", "target"])
result_df_rbf.to_csv("results.csv", index=False)


print("Results for RBF kernel have been saved to 'results.csv'")

#split_and_evaluate(X_train, y_train, selected_features)

