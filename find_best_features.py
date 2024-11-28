import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from evaluate import evaluate_svm_rbf, evaluate_gpr

train_data = pd.read_csv('train_set.csv', index_col=0)

# Separate features and target from training data
train_data = train_data.fillna(0)
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Function for Variance Thresholding
def variance_thresholding(X_train):
    selector = VarianceThreshold(threshold=0.1)
    X_selected = selector.fit_transform(X_train)
    return X_train.columns[selector.get_support()].tolist()

# Function for Correlation Filtering
def correlation_filtering(X_train, threshold=0.1):
    correlation_matrix = X_train.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    return [col for col in X_train.columns if col not in correlated_features]

# Function for Recursive Feature Elimination (RFE)
def recursive_feature_elimination(X_train, y_train):
    model = SVC(kernel="linear")
    selector = RFE(model, n_features_to_select=10)
    selector = selector.fit(X_train, y_train)
    return X_train.columns[selector.get_support()].tolist()

# Function for Feature Importance (using XGBoost)
def feature_importance(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    selected_features = X_train.columns[importance > 0.01].tolist()
    return selected_features

# Function for Lasso-based Feature Selection
def lasso_feature_selection(X_train, y_train):
    lasso = LassoCV()
    lasso.fit(X_train, y_train)
    selected_features = X_train.columns[lasso.coef_ != 0].tolist()
    return selected_features

# Function for Random Forest-based Feature Selection
# Function for Random Forest-based Feature Selection with hyperparameter tuning
def random_forest_feature_selection(X_train, y_train):
    # Tune hyperparameters
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances and select features
    importances = rf.feature_importances_
    selected_features = X_train.columns[importances > 0.01].tolist()  # Select features with importance > 0.01
    return selected_features

# Function for Univariate Feature Selection (ANOVA F-test)
def univariate_feature_selection(X_train, y_train):
    selector = SelectKBest(score_func=f_classif, k=20)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features

# Function to evaluate SVM with RBF kernel and calculate F1 score


# Split the data into training and validation sets
X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Applying each feature selection method
variance_features = variance_thresholding(X_train)
print("Selected Features after Variance Thresholding:", variance_features)

correlation_features = correlation_filtering(X_train, threshold=0.1)
print("Selected Features after Correlation Filtering:", correlation_features)

rfe_features = recursive_feature_elimination(X_train, y_train)
print("Selected Features after RFE:", rfe_features)

importance_features = feature_importance(X_train, y_train)
print("Selected Features after Feature Importance:", importance_features)

lasso_features = lasso_feature_selection(X_train, y_train)
print("Selected Features after Lasso Feature Selection:", lasso_features)

rf_features = random_forest_feature_selection(X_train, y_train)
print("Selected Features after Random Forest Feature Selection:", rf_features)

univariate_features = univariate_feature_selection(X_train, y_train)
print("Selected Features after Univariate Feature Selection:", univariate_features)

# Add a default case with all features
all_features = X_train.columns.tolist()
print("Using all features:", all_features)

# Evaluate each feature set and return the F1 scores
def evaluate_feature_sets(X_train, X_val, y_train, y_val):
    feature_sets = {
        'Variance Thresholding': variance_features,
        'Correlation Filtering': correlation_features,
        'RFE': rfe_features,
        'Feature Importance': importance_features,
        'Lasso': lasso_features,
        'Random Forest': rf_features,
        'Univariate': univariate_features,
        'All Features': all_features,
    }

    # Dictionary to store F1 scores for each feature set
    f1_scores = {}

    for method, features in feature_sets.items():
        print(f"Evaluating {method}...")

        # Evaluate using SVM with RBF kernel
        f1_svm = evaluate_svm_rbf(X_train, X_val, y_train, y_val, features)
        f1_scores[method + ' (SVM)'] = f1_svm
        print(f"F1 Score for {method} using SVM: {f1_svm:.4f}")

        # Evaluate using GPR
        f1_gpr = evaluate_gpr(X_train, X_val, y_train, y_val, features)
        f1_scores[method + ' (GPR)'] = f1_gpr
        print(f"F1 Score for {method} using GPR: {f1_gpr:.4f}")
    
    return f1_scores

# Run the evaluation
f1_scores = evaluate_feature_sets(X_train_data, X_val_data, y_train_data, y_val_data)

# Output the best feature set based on F1 score
best_method = max(f1_scores, key=f1_scores.get)
print(f"\nBest Feature Set and Model: {best_method} | F1 Score: {f1_scores[best_method]:.4f}")
