import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('train_set.csv', index_col=0)
train_data = train_data.fillna(0)

# Separate features and target
X = train_data.drop(columns=['target'])
y = train_data['target']

# Function to test RFE with Gaussian Process Regression
def test_rfe_with_gpr(X, y):
    results = []
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for n_features in range(1, X_train.shape[1] + 1):
        # Use a simple linear model for feature selection
        model_for_rfe = LinearRegression()
        rfe = RFE(model_for_rfe, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        
        selected_features = X_train.columns[rfe.support_].tolist()
        
        # Fit the Gaussian Process model on the selected features
        gpr = GaussianProcessRegressor(random_state=42)
        gpr.fit(X_train[selected_features], y_train)
        
        # Predict on the validation set
        y_pred = gpr.predict(X_val[selected_features])
        
        # Calculate F1 score for regression task (by thresholding the predictions)
        threshold = 0.5  # Set an appropriate threshold based on your target distribution
        y_pred_bin = (y_pred > threshold).astype(int)
        f1 = f1_score(y_val, y_pred_bin)
        
        results.append({
            'n_features': n_features,
            'features': selected_features,
            'f1_score': f1
        })

    return results

# Run the RFE process and get the results
results = test_rfe_with_gpr(X, y)

# Find the best result (maximum F1 score)
best_result = max(results, key=lambda x: x['f1_score'])

# Plot the results
f1_scores = [result['f1_score'] for result in results]
n_features = [result['n_features'] for result in results]

plt.figure(figsize=(10, 6))
plt.plot(n_features, f1_scores, marker='o', label='F1 Score')
plt.axvline(x=best_result['n_features'], color='red', linestyle='--', label=f"Best: {best_result['n_features']} features")
plt.title("F1 Score vs Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.savefig("rfe_feature_test_plot.png")
plt.show()

# Print and save the best result
print(f"Best Result: {best_result['n_features']} features")
print(f"Selected Features: {best_result['features']}")
print(f"F1 Score: {best_result['f1_score']:.4f}")
