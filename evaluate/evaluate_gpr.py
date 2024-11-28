import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def evaluate_gpr(X_train, X_val, y_train, y_val, selected_features):
    # Subset the features
    X_train_subset = X_train[selected_features]
    X_val_subset = X_val[selected_features]

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')  # Impute with the mean value
    X_train_imputed = imputer.fit_transform(X_train_subset)
    X_val_imputed = imputer.transform(X_val_subset)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Train the GPR model with RBF kernel
    kernel = RBF(length_scale=1.0)  # You can tune this parameter
    gpr_model = GaussianProcessClassifier(kernel=kernel)
    gpr_model.fit(X_train_scaled, y_train)

    # Predict on validation set
    y_pred = gpr_model.predict(X_val_scaled)

    # Calculate F1 score
    f1 = f1_score(y_val, y_pred)
    return f1
