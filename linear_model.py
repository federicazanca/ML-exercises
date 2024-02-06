import numpy as np

class LinearModel:
    def __init__(self, features: int):
        self.b = 0.0
        self.w = np.array([0.0]*features)
        self.x_min = 0.0
        self.delta = 1.0

    def estimate(self, x: np.ndarray):
        assert x.shape == self.w.shape 
        if self.delta == 0.0:
            x_norm = x - self.x_min
        else:
            x_norm = (x-self.x_min)/self.delta
        return np.dot(self.w, x_norm) + self.b

    def train(self, x_train: np.ndarray, y_train: np.ndarray, rate: float):
        #check inputs and get shape
        self._validate_training_input(x_train, y_train)
        #normalise inputs
        x_train_norm = self._normalise_inputs(x_train)
        #gradient descent
        self._gradient_descent(x_train_norm, y_train, rate)
        #profit 

    def _validate_training_input(self, x_train: np.ndarray, y_train: np.ndarray):
        samples, features = x_train.shape
        assert y_train.shape == (samples,)
        assert self.w.shape == (features,)

    def _normalise_inputs(self, x_train: np.ndarray) -> np.ndarray:
        self.x_min = np.min(x_train)
        self.delta = np.max(x_train) - self.x_min
        if self.delta == 0.0:
            return x_train * 0.0
        x_train_norm = (x_train-self.x_min)/self.delta
        return x_train_norm
        
    def _gradient_descent(self, x_train: np.ndarray, y_train: np.ndarray, rate: float):
        samples, features = x_train.shape

        # Calculate initial gradient of the cost function
        db, *dw = self._gradient(x_train, y_train)

        # Calculate new params
        for i in range(features):
            self.w[i] -= dw[i]*rate
        self.b -= db*rate

    def _gradient(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple[float, np.ndarray]:
        samples, features = x_train.shape

        # partial derivatives w.r.t. w
        dw = [0.0]*features
        for j in range(features):
            for i in range(samples):
                dw[j] += x_train[i][j]*(np.dot(x_train[i], self.w)+self.b-y_train[i])
            dw[j] /= samples

        # partial derivate w.r.t. b
        db = 0
        for i in range(samples):
            db += np.dot(x_train[i], self.w)+self.b-y_train[i]
        db /= samples

        # gradient
        return db, np.array([*dw])



if __name__ == "__main__":
    x_train = np.array([[0.0], [500.0], [1000.0]])
    y_train = np.array([2.0, 2.5, 3.0])

    model = LinearModel(1)

    x_hat = np.array([2000.0])
    y_hat = model.estimate(x_hat)
    print(y_hat)

    for _ in range(4000):
        model.train(x_train, y_train, 0.1)
    # params = model.params

    y_hat_trained = model.estimate(x_hat)
    print(y_hat_trained)
