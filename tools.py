import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def estimatePrice(mileage):
    t0, t1 = getTheta()
    return t0 + (t1 * mileage)


def getTheta():
    t0 = 0
    t1 = 0
    if os.path.exists("theta"):
        with open("theta", "rb") as f:
            t0, t1 = pickle.load(f)
    return t0, t1


def setTheta(t0, t1):
    with open("theta", "wb") as f:
        pickle.dump([t0, t1], f)


def meanSquaredError(y, y_hat):
    res = np.mean((y - y_hat) ** 2)
    return res


def createAnimation(x, model):
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, model.predict(x), c="r")

    def updateLine(num, line, x):
        line.set_ydata(model.y_min + model.y_range * model.all_y_hat[num])
        return (line)

    ani = animation.FuncAnimation(
        fig,
        updateLine,
        fargs=(line, x),
        frames=len(model.all_y_hat),
        interval=20,
        blit=True,
        repeat=False,
    )
    return ani


def calculatePrecision(model, num):

    if num < len(model.loss):
        initial_loss = model.loss[0]
        current_loss = model.loss[num]
        if initial_loss == 0:
            return 100.0  # If initial loss is zero, precision is perfect
        return 100 * (1 - (current_loss / initial_loss))
    return 0.0  # Return 0% precision if num is out of range
