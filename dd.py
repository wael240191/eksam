# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, auc
)
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Student Depression Dataset.csv'
data = pd.read_csv(file_path)

# Select relevant columns
relevant_columns = [
    'Academic Pressure', 'Sleep Duration', 'Financial Stress', 
    'Family History of Mental Illness', 'Depression'
]
data_relevant = data[relevant_columns]

# Handle missing values by imputing the median
data_relevant['Financial Stress'].fillna(data_relevant['Financial Stress'].median(), inplace=True)

# Encode categorical variables: 'Sleep Duration' and 'Family History of Mental Illness'
categorical_features = ['Sleep Duration', 'Family History of Mental Illness']
encoder = OneHotEncoder(sparse=False)
encoded_categorical = encoder.fit_transform(data_relevant[categorical_features])

# Convert encoded features into a DataFrame
encoded_columns = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_columns, index=data_relevant.index)

# Combine numerical features with encoded categorical features
numerical_features = ['Academic Pressure', 'Financial Stress']
X = pd.concat([data_relevant[numerical_features], encoded_df], axis=1)

# Target variable: 'Depression'
y = data_relevant['Depression']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features for Logistic Regression
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)


# Train Logistic Regression Model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_probabilities = lr_model.predict_proba(X_test)[:, 1]
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)
lr_classification_report = classification_report(y_test, lr_predictions)
lr_roc_auc = roc_auc_score(y_test, lr_probabilities)

# Evaluate Random Forest
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)

# Evaluate Logistic Regression
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)
lr_classification_report = classification_report(y_test, lr_predictions)
lr_roc_auc = roc_auc_score(y_test, lr_probabilities)

# Print evaluation metrics
print("Random Forest Classification Report:")
print(rf_classification_report)
print(f"Random Forest ROC AUC: {rf_roc_auc:.2f}")

print("\nLogistic Regression Classification Report:")
print(lr_classification_report)
print(f"Logistic Regression ROC AUC: {lr_roc_auc:.2f}")

# Visualize Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, ax=axes[0], cmap='Blues')
axes[0].set_title("Random Forest Confusion Matrix")

# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_estimator(lr_model, X_test, y_test, ax=axes[1], cmap='Greens')
axes[1].set_title("Logistic Regression Confusion Matrix")

plt.tight_layout()
plt.show()

# Visualize ROC Curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)

rf_auc = auc(rf_fpr, rf_tpr)
lr_auc = auc(lr_fpr, lr_tpr)

plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})", color='blue')
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.2f})", color='green')
plt.plot([0, 1], [0, 1], 'k--', label="No Skill Model")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
