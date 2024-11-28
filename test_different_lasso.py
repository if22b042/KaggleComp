import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the train data
train_data = pd.read_csv('train_set.csv', index_col=0)
New_train_data = pd.read_csv('archive/unused_train_set.csv', index_col=0)
New_X_train = New_train_data.drop(columns=['target'])

# Separate features and target from training data
train_data = train_data.fillna(0)
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Function to evaluate SVM with RBF kernel and calculate F1 score
def evaluate_svm_rbf(X_train, X_val, y_train, y_val, selected_features, C=10, gamma=0.01):
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

    # Train the SVM model
    svm_model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_pred = svm_model.predict(X_val_scaled)
    
    # Calculate F1 score
    f1 = f1_score(y_val, y_pred)
    return f1

# Function for Lasso-based Feature Selection
def lasso_feature_selection(X_train, y_train, alpha):
    lasso = LassoCV(alphas=[alpha], cv=5, max_iter=10000)
    lasso.fit(X_train, y_train)
    selected_features = X_train.columns[lasso.coef_ != 0].tolist()
    return selected_features

# Function to test different Lasso alpha values
def test_lasso_feature_sizes(X_train, y_train, X_val, y_val):
    alphas = np.logspace(-4, 1, 50)  # Different alpha values (regularization strengths)
    f1_scores = []
    best_selected_features = []

    for alpha in alphas:
        print(f"Testing alpha={alpha}")
        selected_features = lasso_feature_selection(X_train, y_train, alpha)
        if selected_features:
            f1 = evaluate_svm_rbf(X_train, X_val, y_train, y_val, selected_features)
            f1_scores.append(f1)
            if f1 == max(f1_scores):
                best_selected_features = selected_features  # Update with the best feature set
        else:
            f1_scores.append(0)  # If no features are selected, assign a score of 0

    # Find the best alpha based on F1 score
    best_alpha_idx = np.argmax(f1_scores)
    best_alpha = alphas[best_alpha_idx]
    best_f1 = f1_scores[best_alpha_idx]

    print(f"\nBest Alpha: {best_alpha} | Best F1 Score: {best_f1:.4f}")
    return alphas, f1_scores, best_alpha, best_f1, best_selected_features

# Split the data into training and validation sets
X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Run Lasso feature selection for different alpha values and evaluate performance
alphas, f1_scores, best_alpha, best_f1, best_selected_features = test_lasso_feature_sizes(X_train_data, y_train_data, X_val_data, y_val_data)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(alphas, f1_scores, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('F1 Score')
plt.title('Lasso Feature Selection - F1 Score vs Alpha')
plt.grid(True)
plt.show()

# Output the best selected features
print("\nBest selected features based on Lasso feature selection:")
print(best_selected_features)