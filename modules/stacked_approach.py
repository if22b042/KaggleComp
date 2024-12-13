from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def stacked_approach(test_data, X_train, y_train):
    # Standardize the dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    # Define base learners
    base_learners = [
        ('gpr', GaussianProcessClassifier(kernel=RBF(length_scale=1.0))),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ]

    # Create the stacking classifier
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )

    # Fit the stacked model
    stack_model.fit(X_train_scaled, y_train)

    # Predict using the stacked model
    predictions = stack_model.predict(X_test_scaled)
    result_array = [[i, bool(pred)] for i, pred in enumerate(predictions)]

    return result_array

