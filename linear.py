import numpy as np

def linear_regression_gradient(x_train: np.ndarray, y_train: np.ndarray, params: np.ndarray) -> np.ndarray:
    '''
    x_train: n*m matrix of features, n samples and m features 
    y_train: array of targets, n samples
    params: array of linear regression params, m + 1 entries [b, w1, w2, ..., wm]
    Return cost function gradient for linear regression in params
    '''
    # for linear regression, the cost function is:
    # cost = 1/2n * sum((fx-y)^2) = 1/2n * sum((wx+b-y)^2)
    # here we compute algebrically the gradient of the cost
    # dw = 1/n * sum((wx+b-y)x)
    # db = 1/n * sum((wx+b-y))

    n, m = x_train.shape
    assert y_train.shape == (n,)
    assert params.shape == (m+1,)
    b, *w = params

    # partial derivatives w.r.t. w
    dw = [0.0]*m
    for j in range(m):
        for i in range(n):
            dw[j] += x_train[i][j]*(np.dot(x_train[i],w)+b-y_train[i])
        dw[j] /= n

    # partial derivate w.r.t. b
    db = 0
    for i in range(n):
        db += np.dot(x_train[i], w)+b-y_train[i]
    db /= n

    # gradient
    return np.array([db, *dw])


def train(x_train: np.ndarray, y_train: np.ndarray, params: np.ndarray, rate: float, calc_gradient) -> np.ndarray:
    '''
    Return new params after one iteration of steepest descent
    x_train: n*m matrix of features, n samples and m features 
    y_train: array of targets, n samples
    params: array of linear regression params, m + 1 entries [b, w1, w2, ..., wm]
    rate: number, indicates the learning rate
    calc_gradient: a function to calculate the gradient (different for each model)
    '''

    m = params.shape[0]
    gradient = calc_gradient(x_train, y_train, params)
    updated_params = np.array(params)
    for i in range(m):
        updated_params[i] -= gradient[i]*rate
    return updated_params

if __name__ == '__main__':
    x_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([1.0, 3.0, 5.0])
    params = np.array([0.0, 0.0])

    before_train = params[1] * 2.0 + params[0]
    print(str(before_train))

    for i in range(10000):
        params = train(x_train, y_train, params, 0.1, linear_regression_gradient)

    after_train = params[1] * 2.0 + params[0]
    print(str(after_train))
