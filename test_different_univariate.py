import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from evaluate import evaluate_svm_rbf, evaluate_gpr

train_data = pd.read_csv('train_set.csv', index_col=0)

# Separate features and target from training data
train_data = train_data.fillna(0)
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Function to evaluate SVM with RBF kernel and calculate F1 score


# Split the data into training and validation sets
X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Function for Univariate Feature Selection (ANOVA F-test)
def univariate_feature_selection(X_train, y_train, k):
    selector = SelectKBest(score_func=f_classif, k=k)  # Select top k features
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features

# Function to evaluate different sizes of feature sets (1 to 30 features) and different RBF hyperparameters
def evaluate_univariate_feature_sets(X_train, X_val, y_train, y_val):
    f1_scores = []
    selected_features_all = {}

    # Test feature sizes from 1 to 30
    for k in range(1, 31):
        selected_features = univariate_feature_selection(X_train, y_train, k)

    

        best_f1 = evaluate_gpr(X_train, X_val, y_train, y_val, selected_features)


        # Store the best F1 score for this feature size and the corresponding hyperparameters
        
        print(f"Evaluating top {k} features...  ", selected_features, "   f1 Score: ", best_f1)
        f1_scores.append(best_f1)
        selected_features_all[k] = {
            'f1_score': best_f1,
            'selected_features': selected_features
        }

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), f1_scores, marker='o', label='F1 Score')
    plt.title("F1 Score vs Number of Selected Features (Univariate Feature Selection with Hyperparameter Tuning)")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the F1 scores and selected features with the best hyperparameters
    return f1_scores, selected_features_all

# Run the evaluation and plotting
f1_scores, selected_features_all = evaluate_univariate_feature_sets(X_train_data, X_val_data, y_train_data, y_val_data)

# Print the best result
best_k = f1_scores.index(max(f1_scores)) + 1  # Get the feature size with the best F1 score
best_result = selected_features_all[best_k]
print(f"Best number of features: {best_k} with F1 Score: {best_result['f1_score']:.4f}")
print(f"Best Lengt Scale: {best_result['C']}")
print(f"Selected Features for Best Result: {best_result['selected_features']}")
