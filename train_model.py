import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from tools import setTheta
import os
from LinearRegression import LinearRegression


def main():
    try:
        data = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("File not found")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    x = data["km"].to_numpy().reshape(-1, 1)
    y = data["price"].to_numpy().reshape(-1, 1)
    if os.path.exists("theta"):
        os.remove("theta")

    model = LinearRegression()
    model.fit(x, y)
    setTheta(model.t0, model.t1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.set_title("Price of a car for a given mileage (prediction)")
    cursor = mplcursors.cursor(ax1.scatter(x, y, c="b"))
    cursor.connect(
        "add",
        lambda sel: sel.annotation.set_text(
            "Mileage: {}\nPrice: {}\nPredicted Price: {:.2f}".format(
                sel.target[0], sel.target[1], float(model.predict(sel.target[0].reshape(1, -1))[0, 0])
            )
        ),
    )
    ax1.plot(x, model.predict(x), c="r")
    ax1.set_xlabel("Mileage")
    ax1.set_ylabel("Price")

    ax2.set_title("Algorithm precision")
    ax2.plot(model.loss, c="r")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
