import os
import pickle


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
