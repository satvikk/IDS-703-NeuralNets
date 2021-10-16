# SatvikkK, DeekshitaS

import numpy as np
from gen_data import gen_simple, gen_xor
from sklearn.model_selection import train_test_split

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


def main():
    """Run experiment."""
    np.random.seed(8776)
    X, Y = gen_xor(500)
    Y = Y.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state = 11221
    )

    torch_results = np.genfromtxt("predictions_torch.txt")
    np_results = predict(X_test)

    cosine_similarity = (torch_results/np.linalg.norm(torch_results)) * (np_results/np.linalg.norm(np_results)).reshape(-1)
    cosine_similarity = np.sum(cosine_similarity)
    print("The cosine similarity between Numpy implementation and PyTorch implementation is: ",cosine_similarity)
    

if __name__ == "__main__":
    main()
