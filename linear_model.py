import numpy as np

class LinearModel:
    def __init__(self):
        self.params = [[0.0],[0.0]]

    def estimate(self, x_hat):
        b, *w = self.params
        y_hat = np.dot(w, x_hat) + b
        return y_hat

    def train(self, x_train: np.ndarray, y_train: np.ndarray, rate: float):
        #check inputs and get shape
        self._validate_input(x_train, y_train)
        #normalise inputs
        #gradient descent
        self._gradient_descent(x_train, y_train, rate)
        #profit 
    
    def _validate_input(self, x_train: np.ndarray, y_train: np.ndarray):
        samples, features = x_train.shape
        assert y_train.shape == (samples,)
        self.params = np.ndarray([0.0]*(features+1))
    
    def _gradient_descent(self, x_train: np.ndarray, y_train: np.ndarray, rate: float):
        samples, features = x_train.shape

        # Calculate initial gradient of the cost function
        gradient = self._gradient(x_train, y_train)

        # Calculate new params
        for i in range(features+1):
            self.params[i] -= gradient[i]*rate

    def _gradient(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        samples, features = x_train.shape
        b, *w = self.params

        # partial derivatives w.r.t. w
        dw = [0.0]*features
        for j in range(features):
            for i in range(samples):
                dw[j] += x_train[i][j]*(np.dot(x_train[i], w)+b-y_train[i])
            dw[j] /= samples

        # partial derivate w.r.t. b
        db = 0
        for i in range(samples):
            db += np.dot(x_train[i], w)+b-y_train[i]
        db /= samples

        # gradient
        return np.array([db, *dw])



if __name__ == "__main__":
    x_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([2.0, 2.5, 3.0])

    model = LinearModel()

    x_hat = np.array([2.0])
    y_hat = model.estimate(x_hat)
    print(y_hat)

    for _ in range(4000):
        model.train(x_train,y_train, 0.01)
    # params = model.params

    y_hat_trained = model.estimate(x_hat)
    print(y_hat)
