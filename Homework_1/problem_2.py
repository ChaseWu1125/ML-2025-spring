import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Files path setting (don't change this)
TRAIN_PATH = "inputs/training_dataset.csv"
TEST_PATH = "inputs/testing_dataset.csv"
SAVE_PATH = "outputs/result_2.csv"

def plot_3d(x, y, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x[:, 0], x[:, 1], y, c=y, cmap='viridis', marker='o')
    ax.set_xlabel('X1 Label')
    ax.set_ylabel('X2 Label')
    ax.set_zlabel('Y Label')
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(filename)
    plt.close()

def get_Phi(X, Mean, Sigma):
    Phi = np.exp(-np.sum(((X[:, None, :] - Mean[None, :, :]) ** 2) / (2 * Sigma[None, :, :] ** 2), axis=2))
    Phi = np.column_stack([np.ones(Phi.shape[0]), Phi])
    return Phi

def train(O1, O2, l):
    # Param.
    N = 5
    mse = np.inf
    w_folded = None

    # Read data
    train_datas = pd.read_csv(TRAIN_PATH, header = None)
    # Split data into N pieces
    train_datas = np.array_split(train_datas,N)

    # N-fold
    for i in range(N):
        # Training data
        train_data = pd.concat([train_datas[j] for j in range(N) if j != i])
        x_train = train_data.iloc[:, :-1].values
        t_train = train_data.iloc[:, -1].values

        # Validation data
        valid_data = train_datas[i]
        x_valid = valid_data.iloc[:, :-1].values
        t_valid = valid_data.iloc[:, -1].values

        # Compute mean array
        mean_x1 = np.linspace(0, 1, O1)
        mean_x2 = np.linspace(0, 1, O2)
        Mean = np.array(np.meshgrid(mean_x1, mean_x2)).T.reshape(-1, 2)

        # Compute sigma array
        sig_x1 = 1.5/O1
        sig_x2 = 1.5/O2
        Sigma = np.tile(np.array([[sig_x1, sig_x2]]), (O1 * O2, 1))

        # Compute basis func. array
        Phi = get_Phi(x_train, Mean, Sigma)

        # Find weights (MAP)
        w = np.linalg.inv(Phi.T @ Phi + l * np.eye(Phi.shape[1])) @ Phi.T @ t_train

        # Validation
        Phi_valid = get_Phi(x_valid, Mean, Sigma)
        t_infer = Phi_valid @ w
        t_infer = np.maximum(t_infer, 0)

        mse_tmp = np.mean((t_valid - t_infer) ** 2)
        if mse_tmp < mse:
            w_folded = w
            mse = mse_tmp
    return w_folded,Mean,Sigma

def test(w, mean, sigma):
    # Testing
    testing_data = pd.read_csv(TEST_PATH, header=None)
    t_test = testing_data.iloc[:,-1].values
    x_test = testing_data.iloc[:,:-1].values
    phi = get_Phi(x_test, mean, sigma)
    y_test = phi @ w
    y_test = np.maximum(y_test, 0)
    mse = np.mean((y_test - t_test) ** 2)
    print(mse)

    return x_test, y_test

def save_result(preds: np.ndarray, weights: np.ndarray):
    """
    Save prediction and weights to a CSV file
    - `preds`: predicted values with shape (n_samples,)
    - `weights`: model weights with shape (n_basis,)
    """

    max_length = max(len(preds), len(weights))

    result = np.full((max_length, 2), "", dtype=object)
    result[:len(preds), 0] = preds.astype(str)
    result[:len(weights), 1] = weights.astype(str)

    np.savetxt(SAVE_PATH, result, delimiter=",", fmt="%s")

def main(O1,O2,l):
    w, mean, sigma = train(O1,O2,l)
    x_test, y_test = test(w, mean, sigma)
    plot_3d(x_test, y_test, "./p2.png")
    save_result(y_test, w)

main(25,25,0.00000001)