class LinearRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.t0 = 0
        self.t1 = 0
        self.x_min = None
        self.x_range = None
        self.y_min = None
        self.y_range = None

    def fit(self, x, y):
        self.x_min, self.x_range = min(x), max(x) - min(x)
        self.y_min, self.y_range = min(y), max(y) - min(y)
        self.x = (x - self.x_min) / self.x_range
        self.y = (y - self.y_min) / self.y_range
        m = len(self.x)
        y_hat: list = []
        initial_learning_rate = 0.1
        decay_rate = 0.1
        step_size = 1000
        for i in range(100000):
            learning_rate = initial_learning_rate * (1 / (1 + decay_rate * (i // step_size)))
            y_hat = self.t0 + self.t1 * self.x
            formula0 = 1 / m * sum(y_hat - self.y)
            formula1 = 1 / m * sum((y_hat - self.y) * self.x)
            self.t0 -= learning_rate * formula0
            self.t1 -= learning_rate * formula1
        return self

    def predict(self, x):
        return self.y_min + self.y_range * (self.t0 + self.t1 * ((x - self.x_min) / self.x_range))
