import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("kidney disease.csv")

# Create X and y
X = df.drop(columns=["classification"])
y = df["classification"]

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Values of k to test
k_values = [1, 3, 5, 7, 9]

results = []

# Train and evaluate KNN for each k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((k, acc))

# Create a small table of results
results_df = pd.DataFrame(results, columns=["k", "Accuracy"])
print(results_df)

# Identify best k
best_row = results_df.loc[results_df["Accuracy"].idxmax()]
print("\nBest k:", best_row["k"])
print("Highest Accuracy:", best_row["Accuracy"])

# EXPLANATORY COMMENTS
# Changing k affects how sensitive the KNN model is to local patterns in the data.
# Very small values of k, such as k=1, may cause overfitting because the model becomes too influenced by noise or outliers.
# Very large values of k may cause underfitting because the model averages over too many neighbors, smoothing away important distinctions.
# Choosing an appropriate k helps balance the trade‑off between capturing meaningful patterns and avoiding noise.