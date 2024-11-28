import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
train_data = pd.read_csv('fixed_train_set.csv', index_col=0)

# Separate features and target
X = train_data.drop(columns=['target'])
y = train_data['target']

# Step 1: Calculate pairwise correlations
def find_highly_correlated_features(X, threshold=0.95):
    correlation_matrix = X.corr()
    correlated_features = set()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:  # Highly correlated
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    return correlated_features

# Step 2: Use a model to find low importance features
def find_low_importance_features(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    feature_importance = model.feature_importances_
    low_importance_features = X.columns[feature_importance < 0.01].tolist()
    return low_importance_features, feature_importance

# Step 3: Linear Regression to find approximate feature combinations (e.g., X = a*feat_1 + b*feat_2 + c)
def find_feature_combinations(X):
    feature_combinations = []
    
    for col in X.columns:
        # Try to fit a linear regression to predict the feature based on the others
        X_temp = X.drop(columns=[col])
        reg = LinearRegression().fit(X_temp, X[col])
        
        # If the model explains a significant amount of variance, this feature may be a combination of others
        mse = mean_squared_error(X[col], reg.predict(X_temp))
        
        # Threshold can be adjusted based on the amount of variance explained (e.g., MSE below 0.01 could indicate combination)
        if mse < 0.01:
            feature_combinations.append(col)
    
    return feature_combinations

# Find highly correlated features
correlated_features = find_highly_correlated_features(X, threshold=0.95)
print("Highly Correlated Features (Threshold 0.95):", correlated_features)

# Find low importance features using Random Forest
low_importance_features, feature_importance = find_low_importance_features(X, y)
print("Low Importance Features:", low_importance_features)

# Find features that may be combinations of others using Linear Regression
combinations = find_feature_combinations(X)
print("Potential Feature Combinations (based on regression):", combinations)

# Visualizing feature importance for better understanding
plt.figure(figsize=(12, 6))
plt.bar(X.columns, feature_importance)
plt.title('Feature Importance from RandomForestClassifier')
plt.xticks(rotation=90)
plt.ylabel('Importance')
plt.show()
