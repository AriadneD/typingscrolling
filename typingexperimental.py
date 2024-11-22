import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load the data
file_path = 'combined_typing_data.csv'
data = pd.read_csv(file_path)

# Select the features and target
features = [
    't', 'accel_change_x', 'accel_change_y', 'accel_change_z', 'gyro_change_x', 
    'gyro_change_y', 'gyro_change_z', 'accel_magnitude', 'gyro_magnitude', 
    'accel_mean_x', 'accel_mean_y', 'accel_mean_z', 'gyro_mean_x', 'gyro_mean_y', 
    'gyro_mean_z', 'accel_std_x', 'accel_std_y', 'accel_std_z', 'gyro_std_x', 
    'gyro_std_y', 'gyro_std_z', 'accel_energy', 'gyro_energy'
]

# Ensure 'keystrokes' column exists in data as target variable
if 'keystrokes' not in data.columns:
    data['keystrokes'] = 464  # Placeholder if 'keystrokes' doesn't exist

# Prepare feature (X) and target (y) data
X = data[features]
y = data['keystrokes']

# Preprocess the data: drop timestamp column, scale features
X = X.drop(columns=['t'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X to 3D for RNN [samples, time steps, features]
time_steps = 5  # Define the length of sequences (time steps)
X_scaled_seq = []
y_seq = []

# Creating sequences of length `time_steps`
for i in range(len(X_scaled) - time_steps + 1):
    X_scaled_seq.append(X_scaled[i:i + time_steps])
    y_seq.append(y[i + time_steps - 1])

X_scaled_seq = np.array(X_scaled_seq)
y_seq = np.array(y_seq)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Track metrics across folds
final_loss, final_mae, final_mse = [], [], []
history_dict = {'loss': [], 'mae': [], 'mse': []}

for train_index, val_index in kf.split(X_scaled_seq):
    # Split data
    X_train, X_val = X_scaled_seq[train_index], X_scaled_seq[val_index]
    y_train, y_val = y_seq[train_index], y_seq[val_index]

    # Define the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(time_steps, X_train.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    # Train the model and capture the history
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Append metrics for each epoch
    history_dict['loss'].append(history.history['loss'])
    history_dict['mae'].append(history.history['mae'])
    history_dict['mse'].append(history.history['mse'])

    # Evaluate on validation data and store final metrics
    loss, mae, mse = model.evaluate(X_val, y_val, verbose=0)
    final_loss.append(loss)
    final_mae.append(mae)
    final_mse.append(mse)

# Print final average metrics across all folds
print(f"Average Final Loss: {np.mean(final_loss):.4f}")
print(f"Average Final MAE: {np.mean(final_mae):.4f}")
print(f"Average Final MSE: {np.mean(final_mse):.4f}")

# Save the final model trained on the full dataset
model.save('typing_keystroke_predictor_lstm_model.h5')
print("Final model saved as 'typing_keystroke_predictor_lstm_model.h5'")