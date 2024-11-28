import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

# Load the dataset
train_data = pd.read_csv('train_set.csv', index_col=0)

selected_features = ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_10', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_19', 'feat_20', 'feat_21', 'feat_22', 'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29']
train_data= train_data.fillna(0)
train_data= train_data[['target'] +selected_features]

# Separate features and target from training data
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Split the training data into two halves
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_train_1_scaled = scaler.fit_transform(X_train_1)
X_train_2_scaled = scaler.transform(X_train_2)

# Define the hyperparameters grid for testing
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.01, 0.1, 1, 10],  # Kernel width
}

# Function to evaluate the SVM with RBF kernel for different hyperparameters
def evaluate_svm_rbf(X_train_1, y_train_1, X_train_2, y_train_2, param_grid):
    results = []
    
    # Iterate over all combinations of hyperparameters in the grid
    for params in ParameterGrid(param_grid):
        C_value = params['C']
        gamma_value = params['gamma']
        
        # Create and train the SVM model with the current hyperparameters
        svm_model = SVC(C=C_value, gamma=gamma_value, kernel='rbf')
        svm_model.fit(X_train_1, y_train_1)
        
        # Predict on the second half of the data (X_train_2)
        y_pred = svm_model.predict(X_train_2)
        
        # Calculate the F1 score based on the target (True/False)
        f1 = f1_score(y_train_2, y_pred)
        
        # Store the results with the current hyperparameters and the F1 score
        results.append((C_value, gamma_value, f1))
        
        # Print the current attempt and the corresponding F1 score
        print(f"Tested C={C_value}, gamma={gamma_value} => F1 Score: {f1}")
    
    return results

# Run the evaluation
results = evaluate_svm_rbf(X_train_1_scaled, y_train_1, X_train_2_scaled, y_train_2, param_grid)

# Find the best hyperparameters based on F1 score
best_result = max(results, key=lambda x: x[2])

print("\nBest Hyperparameters and F1 Score:")
print(f"C={best_result[0]}, gamma={best_result[1]} => F1 Score: {best_result[2]}")

