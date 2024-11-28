import pandas as pd
from modules import rbf_kernel_approach, xgboost_approach

def split_and_evaluate(X, y, features):
  
    # Split the dataset into two halves
    split_idx = len(X) // 2
    X_1, X_2 = X.iloc[:split_idx], X.iloc[split_idx:]
    y_1, y_2 = y.iloc[:split_idx], y.iloc[split_idx:]

    # Initialize result DataFrames
    result_columns = ["ID", "true_target", "rbf_prediction", "xgb_prediction"]
    results_1 = pd.DataFrame(columns=result_columns)
    results_2 = pd.DataFrame(columns=result_columns)

    # Train on the first half, predict on the second
    rbf_predictions_2 = rbf_kernel_approach(X_2, X_1, y_1)
    xgb_predictions_2 = xgboost_approach(X_2, X_1, y_1)
    results_2["ID"] = X_2.index
    results_2["true_target"] = y_2.values
    
    rbf_predictions_2=[row[1] for row in rbf_predictions_2]
    results_2["rbf_prediction"] = rbf_predictions_2
    # Train on the second half, predict on the first
    rbf_predictions_1 = rbf_kernel_approach(X_1, X_2, y_2)
    xgb_predictions_1 = xgboost_approach(X_1, X_2, y_2)
    results_1["ID"] = X_1.index
    results_1["true_target"] = y_1.values
    
    rbf_predictions_1=[row[1] for row in rbf_predictions_1]
    results_1["rbf_prediction"] = rbf_predictions_1

    # Combine the results
    combined_results = pd.concat([results_1, results_2]).sort_values(by="ID")

    # Save the results to a CSV file
    combined_results.to_csv("results_cross_validation.csv", index=False)
    print("Cross-validation results saved to 'results_cross_validation.csv'")
