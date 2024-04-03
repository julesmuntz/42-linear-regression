import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from tools import getTheta, setTheta, estimatePrice


def main():
    data = pd.read_csv("data.csv")
    x = data["km"]
    y = data["price"]
    plt.scatter(x, y, c="b")
    t0, t1 = getTheta()

    learning_rate = 0.1
    m = len(x)
    for _ in range(1000):
        formula0 = 1 / m * sum((estimatePrice(x[i]) - y[i]) for i in range(m))
        formula1 = 1 / m * sum((estimatePrice(x[i]) - y[i]) * x[i] for i in range(m))
        t0 -= learning_rate * formula0
        t1 -= learning_rate * formula1

    setTheta(t0, t1)
    plt.plot(x, [t0 + t1 * i for i in x], c="r")
    plt.show()


if __name__ == "__main__":
    main()
