# SatvikkK, DeekshitaS

import matplotlib.pyplot as plt
import numpy as np

from gen_data import gen_simple, gen_xor


def calculate_accuracy(y_hat_class, Y):
    """Calculate accuracy."""
    return np.sum(Y.reshape(-1, 1) == y_hat_class) / len(Y)


('0.bias', Parameter containing:
tensor(, requires_grad=True))
('2.weight', Parameter containing:
tensor(, requires_grad=True))
('2.bias', Parameter containing:
tensor(, requires_grad=True))

# TODO
def predict(X):
    weight_layer1 = np.array([[-0.0335, -0.1838],
        [ 0.9810, -3.5658],
        [-4.8229,  1.6182],
        [-4.5253, -3.7302],
        [-0.0601, -0.5431]])
    bias_layer1 = np.array([-0.1844,  1.4693,  1.5449,  2.8962, -0.3810])
    weight_layer2 = np.array([[ 0.1817,  3.8994,  3.8091, -4.8408,  0.1390]])
    bias_layer2 = np.array([-3.5576])
    output = X @ weight_layer1.T + bias_layer1.reshape(1,5)
    output[output<0] = 0
    output = output @ weight_layer2.T + bias_layer2.reshape(1,1)
    output = 1 / (1 + np.exp(-output))
    return output

def plot_results(X, Y):
    """Plot testing results."""

    y_hat = predict(X)
    y_hat_class = np.where(y_hat < 0.5, 0, 1)

    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(
        np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing)
    )

    # Concatenate data to match input
    data = np.hstack(
        (
            XX.ravel().reshape(-1, 1),
            YY.ravel().reshape(-1, 1),
        )
    )

    # Pass data to predict method
    db_prob = predict(data)

    clf = np.where(db_prob < 0.5, 0, 1)

    Z = clf.reshape(XX.shape)

    print("Accuracy {:.2f}%".format(calculate_accuracy(y_hat_class, Y) * 100))

    plt.figure(figsize=(12, 8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=Y,
        cmap=plt.cm.RdYlBu,
    )
    plt.show()


def main():
    """Run experiment."""
    n_dims = 2
    # X, Y = gen_simple(400)

    np.random.seed(8776)
    X, Y = gen_xor(500)
    Y = Y.reshape(-1, 1)

    for y in np.unique(Y):
        plt.plot(X[Y[:, 0] == y, 0], X[Y[:, 0] == y, 1], "o")
    plt.show()

    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
    )

    net = BinaryLinear(n_dims)
    net.train(X_train, Y_train)
    print("The training loss and accuracy progress")
    net.plot_training_progress()

    print("The training accuracy results:")
    net.plot_training_results(X_train, Y_train)

    print("The testing accuracy results:")
    net.plot_testing_results(X_test, Y_test)

    for i in net.net.named_parameters():
        print(i)
        pass

    predictions = net.predict(torch.FloatTensor(X_test))
    print("The prediction vector is:")
    print(predictions)


if __name__ == "__main__":
    main()
