import pandas as pd
import matplotlib.pyplot as plt
from tools import setTheta, createAnimation
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

    ani = createAnimation(x, model)
    plt.title("Price of a car for a given mileage (prediction)")
    plt.scatter(x, y, c="b")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    ani.save("animation.gif")
    plt.show()

    plt.title("Algorithm precision")
    plt.plot(model.loss, c="r")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
