import numpy as np
def predict(w, X):
    ''' predict label of each row of X, given w
    X: a 2-d numpy array of shape (N, d), each row is a datapoint
    w: a 1-d numpy array of shape (d)
    '''
    return np.sign(X.dot(w))
def perceptron(X, y, w_init):
    ''' perform perceptron learning algorithm
    X: a 2-d numpy array of shape (N, d), each row is a datapoint
    y: a 1-d numpy array of shape (N), label of each row of X. y[i] = 1/-1
    w_init: a 1-d numpy array of shape (d)
    '''
    w = w_init
    while True:
        pred = predict(w, X)
        # find indexes of misclassified points
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        # number of misclassified points
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:  # no more misclassified points
            return w
        # random pick one misclassified point
        random_id = np.random.choice(mis_idxs, 1)[0]
        # update w
        w = w + y[random_id] * X[random_id]

means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))

Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)
w_init = np.random.randn(Xbar.shape[1])

w = perceptron(Xbar, y, w_init)

# Predict labels for the training data
y_pred = predict(w, Xbar)

# Print the predictions
print("Predictions:")
for i, pred in enumerate(y_pred):
    print(f"Data point {i + 1}: {'Positive' if pred > 0 else 'Negative'}")

# Calculate and print accuracy
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

def visualize_perceptron(X, y, w, iterations):
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Positive')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', label='Negative')
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = predict(w, np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='g', levels=[0], alpha=0.5, linestyles=['--'])
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Perceptron Decision Boundary (Iterations: {iterations})')
    plt.legend()
    plt.show()

# Modify the perceptron function to return both w and iterations
def perceptron(X, y, w_init):
    w = w_init
    iterations = 0
    while True:
        iterations += 1
        pred = predict(w, X)
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:
            return w, iterations
        random_id = np.random.choice(mis_idxs, 1)[0]
        w = w + y[random_id] * X[random_id]

# Run the modified perceptron algorithm
w, iterations = perceptron(Xbar, y, w_init)

# Visualize the results
visualize_perceptron(X, y, w, iterations)


print(f"\nNumber of iterations: {iterations}")



