# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)

# 1. Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species'] = data['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# 2. Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


sns.pairplot(data, hue='species', diag_kind='kde')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.show()

# 3. Data Preprocessing
# Features and target
X = data.drop('species', axis=1)
y = data['species']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
results = {}

for name, model in models.items():


    

# 5. Compare Model Performance
accuracies = [results[model]['Accuracy'] for model in results]
plt.figure(figsize=(10, 6))
sns.barplot(x=list(models.keys()), y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# 6. Feature Importance (for Random Forest)
rf_model = models['Random Forest']
feature_importance = pd.Series(rf_model.feature_importances_, index=iris.feature_names)
plt.figure(figsize=(8, 6))
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.show()

# 7. Cross-Validation for Best Model
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = models[best_model_name]
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"\nCross-Validation Scores for {best_model_name}:")
print(f"Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 8. Example Prediction
sample = X_test[0].reshape(1, -1)  # Take first test sample
prediction = best_model.predict(sample)
print(f"\nExample Prediction:")
print(f"Input Features: {scaler.inverse_transform(sample)[0]}")
print(f"Predicted Species: {prediction[0]}")   Iris flower classificationg
