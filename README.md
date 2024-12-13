### typingscrolling
---

#### **1. `newmodelcompare.py`**
##### **Description:**
This script compares the performance of multiple classification models for smartphone activity classification using sensor data. It evaluates models like Logistic Regression, Random Forest, SVM, K-Nearest Neighbors, Gradient Boosting, and XGBoost.

##### **How to Use:**
1. Ensure the dataset (`combined_data_2Hz.csv`) is available in the same directory.
2. Run the script to:
   - Preprocess the data (handle NaN and infinite values).
   - Split the data into training and testing sets.
   - Train and cross-validate each model.
   - Save the trained models as `.pkl` files for future use.
3. Check the console output for classification accuracy and reports.

##### **Output:**
- Model comparison metrics (accuracy, precision, recall, F1 score).
- Saved model files (e.g., `logistic_regression_model.pkl`, `random_forest_model.pkl`).

---

#### **2. `newxgboost.py`**
##### **Description:**
This script implements the XGBoost model for smartphone activity classification (e.g., typing, scrolling, other) using datasets at various sampling rates. It identifies XGBoost as the best performer among tested models.

##### **How to Use:**
1. Place the datasets (`combined_data_1Hz.csv`, `combined_data_2Hz.csv`, etc.) in the same directory.
2. Run the script to:
   - Train an XGBoost model for each dataset.
   - Perform Stratified K-Fold cross-validation.
   - Save trained models as `.pkl` files for each dataset.
   - Visualize results:
     - Confusion matrices.
     - ROC AUC curves.
     - Accuracy and ROC AUC comparisons.

##### **Output:**
- Saved XGBoost models for each dataset (e.g., `xgboost_model_1.pkl`).
- Plots for confusion matrices and ROC AUC curves.
- Printed classification metrics (accuracy, ROC AUC).

---

#### **3. `typingexperimental.py`**
##### **Description:**
This script trains a deep learning model (LSTM) to predict typing speed, accuracy, or keystrokes based on smartphone sensor data. It can be adapted to other variables by modifying the target column.

##### **How to Use:**
1. Ensure the dataset (`combined_typing_data.csv`) includes the required features and the target variable (default: `keystrokes`).
2. To predict other targets, replace `keystrokes` with the desired column name in the script.
3. Run the script to:
   - Preprocess the data and generate time-series sequences.
   - Train an LSTM model using 5-fold cross-validation.
   - Save the final trained model as `typing_keystroke_predictor_lstm_model.h5`.

##### **Output:**
- Saved LSTM model (`typing_keystroke_predictor_lstm_model.h5`).
- Training metrics (loss, MAE, MSE) across folds.

---

#### **4. `typingmodelcompare.py`**
##### **Description:**
This script compares multiple regression models for predicting typing-related metrics (e.g., keystrokes) using smartphone sensor data. Models include Linear Regression, Random Forest, Gradient Boosting, SVR, and K-Nearest Neighbors.

##### **How to Use:**
1. Ensure the dataset (`combined_typing_data.csv`) is available and includes the required features and target column (`keystrokes` by default).
2. Run the script to:
   - Train each model using 5-fold cross-validation.
   - Evaluate the models based on Mean Squared Error (MSE) and Mean Absolute Error (MAE).
3. View the printed average scores for each model.

##### **Output:**
- Comparison of regression models based on MSE and MAE.
- Metrics printed for each model in the console.

---

### **General Notes:**
- Ensure all required Python libraries are installed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `shap`, `seaborn`, and `matplotlib`.
- Data files must be properly formatted (e.g., numeric columns without missing or infinite values).
- For best results, tailor the scripts to your specific use case by adjusting file paths, features, or targets as needed.
