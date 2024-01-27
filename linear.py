import numpy as np

def linear_regression_gradient(x_train: np.ndarray, y_train: np.ndarray, params: np.ndarray) -> np.ndarray:
    '''
    Return cost function gradient for linear regression in params
    '''
    #cost = 1/2m * sum((fx-y)^2) = 1/2m * sum((wx+b-y)^2)
    #dw = 1/m * sum((wx+b-y)x)
    #db = 1/m * sum((wx+b-y))
    n = x_train.shape[0]
    dw = 0
    for i in range(n):
        dw += x_train[i]*(x_train[i]*params[0]+params[1]-y_train[i])
    dw /= n
    db = 0
    for i in range(n):
        db += (x_train[i]*params[0]+params[1]-y_train[i])
    db /= n
    return np.array([dw,db])


def train(x_train: np.ndarray, y_train: np.ndarray, params: np.ndarray, rate: float, calc_gradient) -> np.ndarray:
    '''
    Return new params after one iteration of steepest descent
    '''
    m = params.shape[0]
    gradient = calc_gradient(x_train, y_train, params)
    updated_params = np.array(params)
    for i in range(m):
        updated_params[i] -= gradient[i]*rate
    return updated_params

if __name__ == '__main__':
    x_train = np.array([0.0, 0.5, 1.0])
    y_train = np.array([1.0, 3.0, 5.0])
    params = np.array([0.0, 0.0])

    before_train = params[0] * 2.0 + params[1]
    print(str(before_train))

    for i in range(10000):
        params = train(x_train, y_train, params, 5, linear_regression_gradient)

    after_train = params[0] * 2.0 + params[1]
    print(str(after_train))