import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for datasets
datasets = [
    'combined_data_1Hz.csv',
    'combined_data_2Hz.csv',
    'combined_data_4Hz.csv',
    'combined_data_10Hz.csv',
    'combined_data_60Hz.csv'
]

# To store overall metrics for plotting later
overall_accuracy = []
median_roc_auc = []
iqr_roc_auc = []

for i, data_path in enumerate(datasets):
    print(f"\nProcessing dataset {i+1}/{len(datasets)}: {data_path}")
    
    # Load the data
    data = pd.read_csv(data_path)

    # Separate the target column ('label')
    y = data['label']
    
    # Drop non-numeric columns except 'label'
    X = data.drop(columns=data.select_dtypes(include=['object', 'datetime']).columns)
    
    # Encode target labels to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost Model
    print("Training and cross-validating XGBoost model...")
    xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    # Perform Stratified K-Fold Cross-Validation and get predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(xgboost_model, X_train_scaled, y_train, cv=skf, method="predict_proba")
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Fit the model on the entire training data
    xgboost_model.fit(X_train_scaled, y_train)
    
    # Save the model
    model_filename = f'xgboost_model_{i+1}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(xgboost_model, f)
    
    # Evaluate the model
    overall_accuracy.append(accuracy_score(y_test, xgboost_model.predict(X_test_scaled)))
    print(f"Overall Accuracy: {overall_accuracy[-1]:.4f}")
    
    report = classification_report(y_test, xgboost_model.predict(X_test_scaled), target_names=class_names)
    print("Overall Classification Report:\n", report)
    
    roc_auc_scores = roc_auc_score(y_train, y_pred_proba, multi_class="ovr", average='macro')
    median_roc_auc.append(np.median(roc_auc_scores))
    iqr_roc_auc.append(np.percentile(roc_auc_scores, 75) - np.percentile(roc_auc_scores, 25))
    print(f"Median ROC AUC: {median_roc_auc[-1]:.4f}")
    print(f"IQR of ROC AUC: {iqr_roc_auc[-1]:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, xgboost_model.predict(X_test_scaled))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for Dataset {i+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Plot ROC AUC Curves
    plt.figure(figsize=(10, 8))
    for j, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_train == j, y_pred_proba[:, j])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve for {class_name} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC AUC Curves for Dataset {i+1}')
    plt.legend(loc="lower right")
    plt.show()

# Overall Accuracy, Median ROC AUC, IQR of ROC AUC Plots
x_axis_labels = [1, 6, 15, 30, 60]

plt.figure(figsize=(12, 6))
plt.plot(x_axis_labels, overall_accuracy, marker='o', label='Overall Accuracy')
plt.xlabel('Number of Readings')
plt.ylabel('Accuracy')
plt.title('Overall Accuracy vs. Number of Readings')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(x_axis_labels, median_roc_auc, marker='o', label='Median ROC AUC')
plt.xlabel('Number of Readings')
plt.ylabel('Median ROC AUC')
plt.title('Median ROC AUC vs. Number of Readings')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(x_axis_labels, iqr_roc_auc, marker='o', label='IQR of ROC AUC')
plt.xlabel('Number of Readings')
plt.ylabel('IQR of ROC AUC')
plt.title('IQR of ROC AUC vs. Number of Readings')
plt.grid(True)
plt.show()
