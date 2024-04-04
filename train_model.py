import pandas as pd
import matplotlib.pyplot as plt
from tools import setTheta
import os
from LinearRegression import LinearRegression


def main():
    data = pd.read_csv("data.csv")
    x = data["km"].to_numpy().reshape(-1, 1)
    y = data["price"].to_numpy().reshape(-1, 1)
    if os.path.exists("theta"):
        os.remove("theta")
    model = LinearRegression()
    model.fit(x, y)
    setTheta(model.t0, model.t1)
    plt.scatter(x, y, c="b")
    plt.plot(x, model.predict(x), c="r")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()


if __name__ == "__main__":
    main()
