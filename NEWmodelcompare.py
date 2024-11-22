import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Load the data
data_path = 'combined_data_2Hz.csv'
data = pd.read_csv(data_path)

# Check for NaN values and drop rows containing any NaN values
print("Dropping rows with NaN values...")
data.dropna(inplace=True)

# Check for infinite values and replace them if necessary
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)  # Drop rows that became NaN after replacing infinities

# Separate the target column ('label')
y = data['label']

# Drop non-numeric columns except 'label'
X = data.drop(columns=data.select_dtypes(include=['object', 'datetime']).columns)

# Encode target labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
print("Training and cross-validating Logistic Regression model...")
log_reg = LogisticRegression(max_iter=1000)
log_reg_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression cross-validation accuracy: {log_reg_scores.mean():.4f}")
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

# Random Forest Model
print("Training and cross-validating Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest cross-validation accuracy: {rf_scores.mean():.4f}")
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# SVM Model
print("Training and cross-validating SVM model...")
svm = SVC(kernel='linear', random_state=42)
svm_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5, scoring='accuracy')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM cross-validation accuracy: {svm_scores.mean():.4f}")
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# K-Nearest Neighbors Model
print("Training and cross-validating K-Nearest Neighbors model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"K-Nearest Neighbors cross-validation accuracy: {knn_scores.mean():.4f}")
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Gradient Boosting Model
print("Training and cross-validating Gradient Boosting model...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_scores = cross_val_score(gb, X_train_scaled, y_train, cv=5, scoring='accuracy')
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting cross-validation accuracy: {gb_scores.mean():.4f}")
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb, f)

# XGBoost Model
print("Training and cross-validating XGBoost model...")
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgboost_scores = cross_val_score(xgboost_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
xgboost_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgboost_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost cross-validation accuracy: {xgboost_scores.mean():.4f}")
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgboost_model, f)

# Print Accuracy Results
print(f'Logistic Regression Accuracy: {lr_accuracy:.4f}')
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')
print(f'SVM Accuracy: {svm_accuracy:.4f}')
print(f'K-Nearest Neighbors Accuracy: {knn_accuracy:.4f}')
print(f'Gradient Boosting Accuracy: {gb_accuracy:.4f}')
print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')

# Define class names
class_names = label_encoder.classes_

# Print Classification Reports
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=class_names))

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

print("\nClassification Report for SVM:")
print(classification_report(y_test, y_pred_svm, target_names=class_names))

print("\nClassification Report for K-Nearest Neighbors:")
print(classification_report(y_test, y_pred_knn, target_names=class_names))

print("\nClassification Report for Gradient Boosting:")
print(classification_report(y_test, y_pred_gb, target_names=class_names))

print("\nClassification Report for XGBoost:")
print(classification_report(y_test, y_pred_xgb, target_names=class_names))
