import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(data, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination)
    outliers = iso_forest.fit_predict(data)
    outlier_points = [i for i, x in enumerate(outliers) if x == -1]
    return outlier_points

train_data= pd.read_csv('fixed_train_set.csv', index_col=0)

test_data = pd.read_csv('test_set.csv', index_col=0)
data= train_data.drop('target', axis=1)

outlier=detect_outliers_isolation_forest(train_data)
print(outlier)

#Compare what the correlation would be without the outliers

correlation_matrix = train_data.corr()
correlation= correlation_matrix['target']


outliers_df = train_data.iloc[outlier]
outlier_correlation_matrix = outliers_df.corr()
outliers_correlation= outlier_correlation_matrix['target']
print (outliers_correlation)
x=0
while(x<30):

    print(x, correlation[x], outliers_correlation[x])
    x=x+1


