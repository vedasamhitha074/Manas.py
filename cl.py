import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logr(X, y, lr=0.1, epochs=2000):
    samples, features = X.shape
    weights = np.zeros(features)
    bias = 0.0

    for i in range(epochs):
        model = np.dot(X, weights) + bias
        #z=predicted value of y 
        z = sigmoid(model)
        dw = (1 / samples) * np.dot(X.T, (z- y))
        db = (1 / samples) * np.sum(z - y)
        weights -= lr* dw
        bias -= lr * db

    return weights, bias


def main():
    file_path = "candydata.csv"
    df = pd.read_csv(file_path)

    featuress = ["chocolate", "fruity", "sugarpercent"]
    X = df[featuress].values
    y = (df["winpercent"] >= 50).astype(int).values
    weights, bias = logr(X, y, lr=0.1, epochs=2000)

    print("Logistic Regression Results")
    print(f"Bias (c) : {bias:.4f}\n")
    print("Learned Feature Weights (m):")
    for column, w in zip(featuress, weights):
        print(f"  - {column:15s} weight: {w:.4f}")

    
if __name__ == "__main__":
    main()