# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# Load training and test datasets
train_data = pd.read_csv('train_set.csv')  # Replace with the path to your train set
test_data = pd.read_csv('test_set.csv')   # Replace with the path to your test set



X_train_full = train_data.drop(columns=['target'])  
y_train_full = train_data['target']

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(test_data)

# Split training data for validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Use 'mse' for regression problems
    metrics=['accuracy']
)

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping and learning rate scheduling
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Make predictions on the test set
predictions = model.predict(X_test_scaled)
predicted_classes = (predictions > 0.5)  # Convert probabilities to binary classes

# Save predictions to a CSV file
output = pd.DataFrame({
    'ID': np.arange(0, len(predicted_classes)),  # Assuming IDs start from 1
    'Predicted': predicted_classes.flatten()
})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")

# Plot training history
# Plot loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
