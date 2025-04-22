import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

global plotting_sample_period
plotting_sample_period = 2

def read_train_data():
    data_X = np.zeros((0, 28*28))
    data_T = np.zeros((0, 1), dtype=int)
    for i in range(10):
        for j in range(1000):
            img = np.array(Image.open(f"train/{i}/{j}.jpg"))
            data_X = np.vstack([data_X, img.reshape((1, 28*28))])
            data_T = np.vstack([data_T, i])
    return data_X, data_T

def read_test_data():
    data_X = np.zeros((0, 28*28))
    data_T = np.zeros((0, 1), dtype=int)
    for i in range(2000):
        img = np.array(Image.open(f"test/{i}.jpg"))
        data_X = np.vstack([data_X, img.reshape((1, 28*28))])
        data_T = np.vstack([data_T, i%2])
    return data_X

def save_result(y_pred):
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    with open('./outputs/result_2.csv', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f"{y_pred[i]}\n")

def get_phi(data_x: np.ndarray) -> np.ndarray:
    # Add bias term (column of ones) to input features
    ones = np.ones((data_x.shape[0], 1))
    return np.concatenate([ones, data_x], axis=1)

def to_one_hot(data_t: np.ndarray) -> np.ndarray:
    # Convert integer class labels (0~9) to one-hot vectors (flattened to (N*10, 1))
    data_t = data_t.flatten()
    one_hot = np.zeros((data_t.shape[0], 10), dtype=np.int64)
    one_hot[np.arange(data_t.shape[0]), data_t] = 1
    return one_hot.reshape(-1, 1)

def from_one_hot(one_hot_flat: np.ndarray) -> np.ndarray:
    # Convert flattened one-hot vectors (N*10, 1) back to class indices (N, 1)
    N = one_hot_flat.shape[0] // 10
    one_hot = one_hot_flat.reshape(N, 10)
    class_idx = np.argmax(one_hot, axis=1).reshape(-1, 1)
    return class_idx

def softmax(a: np.ndarray) -> np.ndarray:
    # Numerically stable softmax for flattened input of shape (N*10, 1)
    a = np.asarray(a, dtype=np.float64)
    N = a.shape[0] // 10
    a_reshaped = a.reshape(N, 10)

    a_max = np.max(a_reshaped, axis=1, keepdims=True)
    exp_shift = np.exp(a_reshaped - a_max)
    sum_exp = np.sum(exp_shift, axis=1, keepdims=True)
    result = exp_shift / sum_exp
    return result.reshape(-1, 1)

def get_gradient(w: np.ndarray, data_t: np.ndarray, phi_X: np.ndarray) -> np.ndarray:
    # Compute gradient of softmax cross-entropy loss
    w = np.asarray(w, dtype=np.float64)
    data_t = np.asarray(data_t, dtype=np.float64)

    k = phi_X.shape[1]  # feature dim with bias
    logits = phi_X @ w.reshape(k, 10)      # (N, 10)
    probs = softmax(logits.reshape(-1, 1)) # (N*10, 1)
    error = probs - data_t                 # (N*10, 1)

    grad = phi_X.T @ error.reshape(-1, 10) # (k, 10)
    return grad.reshape(-1, 1)             # flatten to (k*10, 1)

def get_MCE(w: np.ndarray, data_t: np.ndarray, phi_X: np.ndarray) -> float:
    k = phi_X.shape[1]
    logits = phi_X @ w.reshape(k, 10)
    probs = softmax(logits.reshape(-1, 1))

    eps = 1e-12
    y = np.clip(probs, eps, 1 - eps)

    return -np.sum(data_t * np.log(y)) / data_t.shape[0]

def train(
    data_x: np.ndarray, data_t: np.ndarray, 
    val_x: np.ndarray, val_t: np.ndarray, 
    lr: float, epochs: int
    ):
    # Train softmax logistic regression using gradient descent
    phi_X = get_phi(data_x)                     # (N, k)
    data_t = to_one_hot(data_t)                 # (N*10, 1)
    phi_val = get_phi(val_x)
    val_t = to_one_hot(val_t)
    w = np.zeros((phi_X.shape[1] * 10, 1), dtype=np.float64)

    train_accs = []
    val_accs = []
    train_MCE = []
    val_MCE = []

    for epoch in range(epochs):
        grad = get_gradient(w, data_t, phi_X)
        preds = inference(w, data_x)
        acc = (preds.reshape(-1, 1) == from_one_hot(data_t)).mean()
        if acc < 0.85:
            learning_rate = 0.01
        else:
            learning_rate = lr 
        w -= learning_rate * grad

        if (epoch + 1) % plotting_sample_period == 0:
            # If you want to speedup the training process, you can modify this.
            train_accs.append(acc)

            val_preds = inference(w, val_x)
            val_acc = (val_preds.reshape(-1, 1) == from_one_hot(val_t)).mean()
            val_accs.append(val_acc)

            loss = get_MCE(w, data_t, phi_X)
            train_MCE.append(loss)

            loss_val = get_MCE(w, val_t, phi_val)
            val_MCE.append(loss_val)

        # Optional: monitor training progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")

    return w, train_accs, val_accs, train_MCE, val_MCE

def inference(w: np.ndarray, data_x: np.ndarray) -> np.ndarray:
    # Predict class indices using trained weights
    phi_X = get_phi(data_x)
    k = phi_X.shape[1]
    logits = phi_X @ w.reshape(k, 10)
    probs = softmax(logits.reshape(-1, 1))  # (N*10, 1)

    y = np.empty_like(probs, dtype=np.int64)
    y[probs >= 0.5] = 1
    y[probs <  0.5] = 0

    return from_one_hot(y)  # shape: (N, 1)

def k_fold(
    data_x: np.ndarray, data_t: np.ndarray, 
    k: int, lr: float, epochs: int, 
    shuffle = True, random_state: int | None = None,
):
    # Perform k-fold cross-validation and return best model (by accuracy)
    N = data_x.shape[0]
    idx = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    fold_sizes = [N // k] * k
    for i in range(N % k):
        fold_sizes[i] += 1

    best_acc = -1
    best_w = None

    start = 0
    for fold, size in enumerate(fold_sizes, 1):
        val_idx = idx[start : start + size]
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=True)
        start += size

        x_train, y_train = data_x[train_idx], data_t[train_idx]
        x_val,   y_val   = data_x[val_idx],   data_t[val_idx]

        w, train_accs, val_accs, train_MCE, val_MCE = train(x_train, y_train, x_val, y_val, lr=lr, epochs=epochs)
        y_pred = inference(w, x_val)

        acc = (y_pred == y_val.reshape(-1, 1)).mean()
        print(f"Fold {fold}/{k}  |  val accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_w = w.copy()

    print(f"\nBest validation accuracy = {best_acc:.4f}")
    return best_w, train_accs, val_accs, train_MCE, val_MCE

# Load and train using k-fold validation
data_x, data_y = read_train_data()
w, train_accs, val_accs, train_MCE, val_MCE = k_fold(data_x, data_y, k=4, lr=0.0005, epochs=500, random_state=114514810)

# Run inference on test set and save predictions
test_x = read_test_data()
y_pred = inference(w, test_x)
save_result(y_pred.flatten())

plt.plot(np.arange(1, len(train_accs) + 1), train_accs, label='Train Acc')
plt.plot(np.arange(1, len(val_accs) + 1), val_accs, label = 'Val Acc')
plt.xlabel(f"per {plotting_sample_period} Epochs")
plt.ylabel("Training Accuracy")
plt.title("Learning Curve (Training Accuracy)")
plt.legend()
plt.grid(True)
plt.savefig("image2-1.png")
plt.close()

plt.plot(np.arange(1, len(train_MCE) + 1), train_MCE, label = 'Train Loss')
plt.plot(np.arange(1, len(val_MCE) + 1), val_MCE, label = 'Val Loss')
plt.xlabel(f"per {plotting_sample_period} Epochs")
plt.ylabel("Training Loss")
plt.title("Learning Curve (Training Loss)")
plt.legend()
plt.grid(True)
plt.savefig("image2-2.png")
plt.close()