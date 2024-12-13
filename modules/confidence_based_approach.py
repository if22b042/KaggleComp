import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

def confidence_based_approach(test_data, X_train, y_train):
    # Standardize the dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    # Define the models
    models = {
        'gpr': GaussianProcessClassifier(kernel=RBF(length_scale=1.0)),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'logreg': LogisticRegression(max_iter=1000, random_state=42)
    }

    # Fit the models
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

    # Generate predictions with confidence scores
    results = []
    for i, x_test in enumerate(X_test_scaled):
        highest_confidence = -1
        best_prediction = None

        for name, model in models.items():
            # Predict probabilities if available; otherwise, use the decision function
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([x_test])
                confidence = max(probs[0])  # Confidence is the highest probability
                prediction = np.argmax(probs)
            elif hasattr(model, "decision_function"):
                decision = model.decision_function([x_test])
                confidence = abs(decision[0])  # Use magnitude as confidence
                prediction = int(decision[0] > 0)
            else:
                confidence = 0  # Default confidence if not available
                prediction = model.predict([x_test])[0]

            # Update if this model is more confident
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_prediction = prediction

        results.append([i, bool(best_prediction)])

    return results