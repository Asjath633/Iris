# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species'] = data['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# 2. Data Preprocessing
X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=['Setosa', 'Versicolor', 'Virginica'])
class_report = classification_report(y_test, y_pred)

# 5. Print evaluation metrics
print("Model Evaluation Metrics (Random Forest):")
print(f"Accuracy: {accuracy:.2%}")
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix, index=['Actual Setosa', 'Actual Versicolor', 'Actual Virginica'],
                   columns=['Predicted Setosa', 'Predicted Versicolor', 'Predicted Virginica']))
print("\nClassification Report:")
print(class_report)

# 6. Example prediction
sample = X_test[0].reshape(1, -1)
prediction = rf_model.predict(sample)
print("\nExample Prediction:")
print(f"Input Features (scaled): {scaler.inverse_transform(sample)[0]}")
print(f"Predicted Species: {prediction[0]}")
