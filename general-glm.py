import numpy as np
import math

data = np.genfromtxt(r'AI-ML\Housing.csv', delimiter=',')[1:]
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
y = (data.T)[0]
x = (data.T)[1:5].T


def h(type, input, theta):

    if type == 'g':
        return np.dot(theta.T, input)
    elif type == 'b':
        return 1/(1+ math.exp(-np.dot(theta.T, input)))

def fit(type, x, y, learning_rate, epochs, batch_size):

    theta = np.random.rand(len(x[0]))

    for i in range(epochs):
        for j in range(0, len(x), batch_size):
            x_batch = x[j:j+batch_size]
            y_batch = y[j:j+batch_size]

            for k in range(len(x_batch)):
                theta += learning_rate * (y_batch[k] - h(type,x_batch[k],theta)) * x_batch[k]
    return theta

def test(model_type, x, y, theta):
    if model_type == 'b':
        correct = sum(1 for xi, yi in zip(x, y) if round(h('b', xi, theta)) == yi)
        return f"Accuracy: {correct / len(y):.2%}"
    else:
        mse = np.mean([(yi - h('g', xi, theta))**2 for xi, yi in zip(x, y)])
        return f"Mean Squared Error: {mse:.4f}"

x_test = x[300:]
y_test = y[300:]
x_train = x[:300]
y_train = y[:300]

theta = fit('g', x_train, y_train, 0.1, 1000, 10)
print("Test Accuracy: ", test('g', x_test, y_test, theta))
    
