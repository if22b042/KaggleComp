import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_predictions(result_file):
    """
    Evaluates the predictions in the provided result file using F1-score, precision, and recall.

    Args:
        result_file (str): Path to the CSV file containing true targets and predictions.

    Returns:
        None: Prints the evaluation results to the console.
    """
    # Load the results file
    results = pd.read_csv(result_file)

    # Ensure required column0s are present
    required_columns = ["true_target", "rbf_prediction", "xgb_prediction"]
    if not all(col in results.columns for col in required_columns):
        raise ValueError(f"The file {result_file} must contain the following columns: {required_columns}")

    # Evaluate RBF predictions
    true = results["true_target"]
    pred = results["rbf_prediction"]
    f1 = f1_score(true, pred, average="binary")
    precision = precision_score(true, pred, average="binary")
    recall = recall_score(true, pred, average="binary")

    # Print evaluation results
    print("\n--- Evaluation Results ---\n")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}\n")



if __name__ == "__main__":
    # Specify the result file to evaluate
    result_file = "results_cross_validation.csv"
    try:
        evaluate_predictions(result_file)
    except Exception as e:
        print(f"Error: {e}")
