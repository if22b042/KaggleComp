from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def knn_approach(test_data, X_train, y_train, n_neighbors=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    # Initialize k-NN classifier with specified number of neighbors
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_scaled, y_train)

    predictions = knn_model.predict(X_test_scaled)
    result_array = [[i, bool(pred)] for i, pred in enumerate(predictions)]
    
    return result_array
