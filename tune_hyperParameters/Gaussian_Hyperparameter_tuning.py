import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

train_data = pd.read_csv('data/train_set.csv', index_col=0)

selected_features =['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',  'feat_6','feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22',  'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29']
train_data = train_data.fillna(0)
train_data = train_data[['target'] + selected_features]

X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_1_scaled = scaler.fit_transform(X_train_1)
X_train_2_scaled = scaler.transform(X_train_2)

param_grid = {
    'length_scale': [ 0.1, 0.3, 0.5, 1.0, 2.0, 5.0], # Length scale values to test
    'noise_level' : [ 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
}

# Function to evaluate the Gaussian Process Classifier with different length_scale values
def evaluate_gpr(X_train_1, y_train_1, X_train_2, y_train_2, param_grid):
    results = []
    
    # Iterate over all combinations of hyperparameters in the grid
    for params in ParameterGrid(param_grid):
        length_scale_value = params['length_scale'],
        noise_level_value= params['noise_level']
        
        # Use the RBF kernel with the current length_scale
        kernel = RBF(length_scale=length_scale_value)+WhiteKernel(noise_level=1.0)
        gpr_model = GaussianProcessClassifier(kernel=kernel)
        gpr_model.fit(X_train_1, y_train_1)
        
        # Predict on the second half of the data (X_train_2)
        y_pred = gpr_model.predict(X_train_2)
        
        # Calculate the F1 score based on the target (True/False)
        f1 = f1_score(y_train_2, y_pred)
        
        # Store the results with the current hyperparameters and the F1 score
        results.append((length_scale_value, noise_level_value, f1))
        
        # Print the current attempt and the corresponding F1 score
        print(f"Tested length_scale={length_scale_value, noise_level_value} => F1 Score: {f1}")
    
    return results

# Run the evaluation
results = evaluate_gpr(X_train_1_scaled, y_train_1, X_train_2_scaled, y_train_2, param_grid)

best_result = max(results, key=lambda x: x[1])

print("\nBest Hyperparameters and F1 Score:")
print(f"length_scale={best_result[0]} => F1 Score: {best_result[1]}")
