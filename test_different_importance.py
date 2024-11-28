import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('train_set.csv', index_col=0)
train_data = train_data.fillna(0)

# Define the selected features (all features)
X = train_data.drop(columns=['target'])
y = train_data['target']

# Split the training data into two halves (same as in hyperparameter testing code)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X, y, test_size=0.5, random_state=42)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_train_1_scaled = scaler.fit_transform(X_train_1)
X_train_2_scaled = scaler.transform(X_train_2)

# Function to determine feature importance using XGBoost
def feature_importance(X_train_1, y_train_1):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_1, y_train_1)
    
    # Get the feature importances
    importance = model.feature_importances_
    
    return importance

# Evaluate feature importance and print the results
def test_feature_importance(X_train_1_scaled, y_train_1, X_train_2_scaled, y_train_2):
    importance = feature_importance(X_train_1_scaled, y_train_1)
    
    # Iterate over different thresholds to select features and evaluate performance
    results = []
    thresholds = [i * 0.001 for i in range(1, 200)]

    for threshold in thresholds:
        selected_features = X_train_1.columns[importance > threshold].tolist()
        
        if not selected_features:
            continue
        
        # Train an SVM model with RBF kernel using the selected features
        svm_model = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
        svm_model.fit(X_train_1_scaled[:, importance > threshold], y_train_1)
        
        # Predict on the second half of the data
        y_pred = svm_model.predict(X_train_2_scaled[:, importance > threshold])
        
        # Calculate the F1 score
        f1 = f1_score(y_train_2, y_pred)
        
        results.append({
            'threshold': threshold,
            'f1_score': f1
        })

    # Find the best result based on F1 score
    best_result = max(results, key=lambda x: x['f1_score'])

    # Plot the results
    f1_scores = [result['f1_score'] for result in results]
    thresholds = [result['threshold'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='o', label='F1 Score')
    plt.axvline(x=best_result['threshold'], color='red', linestyle='--', label=f"Best Threshold: {best_result['threshold']}")
    plt.title("F1 Score vs Feature Importance Threshold")
    plt.xlabel("Feature Importance Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output the best result
    print("\nBest Result:")
    print(f"Threshold={best_result['threshold']} => F1 Score: {best_result['f1_score']:.4f}")

# Run the feature importance evaluation
test_feature_importance(X_train_1_scaled, y_train_1, X_train_2_scaled, y_train_2)
