import pandas as pd
import matplotlib.pyplot as plt
from tools import setTheta, createAnimation
import os
from LinearRegression import LinearRegression
import multiprocessing


def save_animation(ani):
    print("Writing to animation.gif...")
    ani.save("animation.gif")
    print("Done!")


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
    ani = createAnimation(x, model)

    plt.title("Price of a car for a given mileage (prediction)")
    plt.scatter(x, y, c="b")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    save_process = multiprocessing.Process(target=save_animation, args=(ani,))
    save_process.start()
    plt.show()
    save_process.join()

    plt.title("Algorithm precision")
    plt.plot(model.loss, c="r")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
