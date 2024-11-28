from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def rbf_kernel_approach(test_data, X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    svm_model = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    predictions = svm_model.predict(X_test_scaled)
    result_array = [[i, bool(pred)] for i, pred in enumerate(predictions)]
    
    return result_array
