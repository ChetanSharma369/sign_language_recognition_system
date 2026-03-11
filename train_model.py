import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading dataset...")

data = pd.read_csv("dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

print("Saving model...")

pickle.dump(model, open("sign_model.pkl", "wb"))

print("Model saved successfully")