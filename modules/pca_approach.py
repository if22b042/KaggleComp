import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def pca_approach(test_data, X_train, y_train, n_components=10, kernel="rbf", C=1.0):
    """
    Use PCA for dimensionality reduction followed by an SVM for classification.

    Parameters:
        test_data (DataFrame): Test data for prediction.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        n_components (int): Number of principal components to retain.
        kernel (str): Kernel type for the SVM.
        C (float): Regularization parameter for the SVM.

    Returns:
        result_array (list): A list of [index, prediction] for the test data.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train an SVM
    svm_model = SVC(kernel=kernel, C=C, random_state=42)
    svm_model.fit(X_train_pca, y_train)

    # Predict on the test data
    test_predictions = svm_model.predict(X_test_pca)
    result_array = [[i, bool(pred)] for i, pred in enumerate(test_predictions)]

    return result_array
