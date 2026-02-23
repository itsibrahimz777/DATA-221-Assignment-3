import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("kidney_disease.csv")

# Create feature matrix X and label vector y
X = df.drop(columns=["classification"])
y = df["classification"]

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# TRAIN KNN MODEL
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# METRICS
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label="ckd")
rec = recall_score(y_test, y_pred, pos_label="ckd")
f1 = f1_score(y_test, y_pred, pos_label="ckd")

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)

# EXPLANATORY COMMENTS
# A True Positive means the model correctly predicted that a patient has kidney disease.
# A True Negative means the model correctly predicted that a patient does not have kidney disease.
# A False Positive occurs when the model predicts kidney disease for someone who is actually healthy.
# A False Negative occurs when the model predicts a patient is healthy even though they actually have kidney disease.
# Accuracy alone may not be enough because it can be misleading when the classes are imbalanced.
# If missing a kidney disease case is very serious, recall becomes the most important metric because it measures how many actual disease cases the model successfully identifies.
# Prioritizing recall helps reduce False Negatives, which is crucial in medical diagnosis.
