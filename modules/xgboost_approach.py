import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def xgboost_approach(test_data, X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_data)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    predictions = xgb_model.predict(X_test_scaled)
    result_array = [[i, bool(pred)] for i, pred in enumerate(predictions)]
    
    return result_array
