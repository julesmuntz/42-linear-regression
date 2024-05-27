import os
import pickle
import numpy as np


def estimatePrice(mileage):
    t0, t1 = getTheta()
    t0 = float(t0)
    t1 = float(t1)
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


def calculatePrecision(model, num):

    if num < len(model.loss):
        initial_loss = model.loss[0]
        current_loss = model.loss[num]
        if initial_loss == 0:
            return 100.0
        return 100 * (1 - (current_loss / initial_loss))
    return 0.0
