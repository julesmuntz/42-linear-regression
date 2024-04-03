import pandas as pd
import matplotlib.pyplot as plt
from tools import setTheta, estimatePrice, meanSquaredError
import numpy as np
import os


def main():
    data = pd.read_csv("data.csv")
    x = data["km"].to_numpy()
    y = data["price"].to_numpy()
    t0, t1 = 0, 0
    if os.path.exists("theta"):
        os.remove("theta")
    x = (x - min(x)) / (max(x) - min(x))
    y = (y - min(y)) / (max(y) - min(y))
    learning_rate = 0.0005
    m = len(x)
    loss: list = np.array([])
    y_hat: list = np.array([])
    for _ in range(1500):
        y_hat = np.array([])
        for i in x:
            y_hat = np.append(y_hat, [t0 + (t1 * i)])
        loss = np.append(loss, [meanSquaredError(y, y_hat)])
        formula0 = 1 / m * sum((estimatePrice(x[i]) - y[i]) for i in range(m))
        formula1 = (1 / m * sum((estimatePrice(x[i]) - y[i]) * x[i] for i in range(m)))
        t0 -= learning_rate * formula0
        t1 -= learning_rate * formula1
    print(loss)
    plt.plot(loss, c="r")
    plt.show()
    setTheta(t0, t1)
    plt.plot(x, t0 + t1 * x, c="r")
    plt.scatter(x, y, c="b")
    plt.xlabel("km")
    plt.ylabel("price")
    plt.show()


if __name__ == "__main__":
    main()
