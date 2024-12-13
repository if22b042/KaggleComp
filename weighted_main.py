import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load train and test datasets
test_data = pd.read_csv('data/test_set.csv', index_col=0)
train_data = pd.read_csv('data/train_set.csv', index_col=0)

# Selected features (you can adjust this based on your needs)
selected_features = [
'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',  'feat_6','feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22',  'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29'
]

# Preprocess the data
train_data = train_data.fillna(0)
test_data = test_data[selected_features]
train_data = train_data[['target'] + selected_features]

X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data)

# Calculate feature importance using XGBoost
def calculate_feature_importance(X, y):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    return importance

# Adjust features based on importance
def adjust_features(X, feature_importance, threshold=0.01):
    normalized_importance = feature_importance / np.max(feature_importance)
    scaling_factors = np.where(normalized_importance > threshold, normalized_importance, 0.01)
    X_adjusted = X * scaling_factors
    return X_adjusted

# Calculate and adjust features
feature_importance = calculate_feature_importance(X_train_scaled, y_train)
X_train_adjusted = adjust_features(X_train_scaled, feature_importance)
X_test_adjusted = adjust_features(X_test_scaled, feature_importance)

# Train a Gaussian Process Classifier on adjusted features
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
gpr_model = GaussianProcessClassifier(kernel=kernel, random_state=42)
gpr_model.fit(X_train_adjusted, y_train)

# Make predictions on test data
y_pred = gpr_model.predict(X_test_adjusted)

# Prepare results for saving
result_array = [[i, bool(pred)] for i, pred in enumerate(y_pred, start=0)]
result_df = pd.DataFrame(result_array, columns=["ID", "target"])
result_df.to_csv("results.csv", index=False)

print("Results have been saved to 'results.csv'")
