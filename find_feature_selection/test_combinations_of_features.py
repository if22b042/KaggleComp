import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import random

# Load the dataset
train_data = pd.read_csv('train_set.csv', index_col=0)

# Define the order of features
feature_order = [
    'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10',
    'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20',
    'feat_21', 'feat_22', 'feat_23', 'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29'
]

# Feature sets
selected_features_1 = ['feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22', 'feat_25', 'feat_26', 'feat_27', 'feat_28', 'feat_29']
selected_features_2 = ['feat_0', 'feat_1', 'feat_4', 'feat_6', 'feat_7', 'feat_10', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_19', 'feat_20', 'feat_21', 'feat_22', 'feat_24', 'feat_26', 'feat_27', 'feat_28']

# Fill missing data and prepare the dataset
train_data = train_data.fillna(0)  
train_data = train_data[['target'] + selected_features_1 + selected_features_2]  # Include target and selected features

# Separate features and target
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Split the training data into two halves
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_train_1_scaled = pd.DataFrame(scaler.fit_transform(X_train_1), columns=X_train_1.columns, index=X_train_1.index)
X_train_2_scaled = pd.DataFrame(scaler.transform(X_train_2), columns=X_train_2.columns, index=X_train_2.index)

# Hyperparameter grid for testing
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.01, 0.1, 1, 10],  # Kernel width
}

# Function to evaluate the SVM with random feature sets and hyperparameters
def evaluate_svm_rbf(X_train_1, y_train_1, X_train_2, y_train_2, selected_features, param_grid):
    results = []
    
    # Sort the features based on the predefined feature order
    selected_features_sorted = sorted(selected_features, key=lambda x: feature_order.index(x))
    
    # Select the features from the dataset
    X_train_1_selected = X_train_1[selected_features_sorted]
    X_train_2_selected = X_train_2[selected_features_sorted]

    # Iterate over all combinations of hyperparameters in the grid
    for params in ParameterGrid(param_grid):
        C_value = params['C']
        gamma_value = params['gamma']
        
        # Create and train the SVM model
        svm_model = SVC(C=C_value, gamma=gamma_value, kernel='rbf')
        svm_model.fit(X_train_1_selected, y_train_1)

        # Predict on the second half of the data (X_train_2)
        y_pred = svm_model.predict(X_train_2_selected)

        # Calculate the F1 score
        f1 = f1_score(y_train_2, y_pred)

        # Store the results with the current hyperparameters and the F1 score
        results.append((C_value, gamma_value, f1))

    return results

# Function to generate random feature combinations (with no duplicates)
def generate_random_feature_combinations(features_1, features_2, n_combinations=10):
    all_features = features_1 + features_2
    combinations = []
    
    for _ in range(n_combinations):
        num_features_1 = random.randint(1, len(features_1))
        num_features_2 = random.randint(1, len(features_2))
        
        # Randomly select features without duplication
        selected_from_1 = random.sample(features_1, num_features_1)
        selected_from_2 = random.sample(features_2, num_features_2)
        
        # Combine the features from both sets
        combined_features = selected_from_1 + selected_from_2
        
        # Ensure no duplicates by converting to a set and then back to a list
        combined_features = list(set(combined_features))
        
        combinations.append(combined_features)
    
    return combinations

# Generate random feature combinations
random_combinations = generate_random_feature_combinations(selected_features_1, selected_features_2, n_combinations=50)

# Evaluate the performance for each combination and track the best one
best_f1_score = -1
best_features = None

for features in random_combinations:
    results = evaluate_svm_rbf(X_train_1_scaled, y_train_1, X_train_2_scaled, y_train_2, features, param_grid)

    # Find the best result for this combination of features
    best_result = max(results, key=lambda x: x[2])
    f1 = best_result[2]

    # Track the best feature set and F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_features = features

# Sort the best feature set according to the predefined feature order
best_features_sorted = sorted(best_features, key=lambda x: feature_order.index(x))

# Print best feature set and score
print("\nBest Feature Set and F1 Score:")
print(f"Best Features: {best_features_sorted}")
print(f"Best F1 Score: {best_f1_score}")

# Now run hyperparameter tuning using the best feature set
print("\nRunning hyperparameter tuning with the best feature set:")

# Now, we use the best feature set found to run the hyperparameter tuning
results = evaluate_svm_rbf(X_train_1_scaled, y_train_1, X_train_2_scaled, y_train_2, best_features_sorted, param_grid)

# Find the best hyperparameters based on F1 score
best_result = max(results, key=lambda x: x[2])

# Print the best hyperparameters and the corresponding F1 score
print("\nBest Hyperparameters and F1 Score:")
print(f"C={best_result[0]}, gamma={best_result[1]} => F1 Score: {best_result[2]}")
