from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def poly_kernel_approach(test_data, X_train, y_train, degree=3, C=1.0):
    """
    Uses an SVM with a polynomial kernel to classify the data.

    Parameters:
        test_data (DataFrame): The test data to predict.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        degree (int): The degree of the polynomial kernel.
        C (float): Regularization parameter.
    
    Returns:
        result_array (list): A list of [index, prediction] for the test data.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    # Train SVM with polynomial kernel
    poly_svm = SVC(kernel="poly", degree=degree, C=C, random_state=42)
    poly_svm.fit(X_train_scaled, y_train)

    # Predict on test data
    test_predictions = poly_svm.predict(X_test_scaled)
    result_array = [[i, bool(pred)] for i, pred in enumerate(test_predictions)]
    
    return result_array
