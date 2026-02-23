import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("kidney_disease.csv")

# Create feature matrix X (all columns except 'classification')
X = df.drop(columns=["classification"])

# Create label vector y (the classification column)
y = df["classification"]

# Split into training (70%) and testing (30%) with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# COMMENTS
# We should not train and test a model on the same data because the model would simply
# memorize the training examples, giving an unrealistically high performance score.
# This would prevent us from knowing whether the model can generalize to new, unseen data.
# The purpose of the testing set is to evaluate how well the trained model performs
# on data it has never encountered before.
# This helps us estimate the model’s real‑world predictive ability and detect overfitting.