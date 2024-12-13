from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

def gpr_approach(test_data, X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)


    gpr_model = GaussianProcessClassifier(kernel=RBF(length_scale=1.0)+WhiteKernel(noise_level=0.1))
    gpr_model.fit(X_train_scaled, y_train)

    predictions = gpr_model.predict(X_test_scaled)
    result_array = [[i, bool(pred)] for i, pred in enumerate(predictions)]
    
    return result_array
