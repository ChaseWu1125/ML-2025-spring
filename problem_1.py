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
            data_T = np.vstack([data_T, i%2])
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
    with open('./outputs/result_1.csv', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f"{y_pred[i, 0]}\n")

def get_phi(data_x: np.ndarray) -> np.ndarray:
    # Add bias term (column of ones) to the input features
    ones = np.ones((data_x.shape[0], 1))
    return np.concatenate([ones, data_x], axis=1)

def sigmoid(a: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid function
    a = np.asarray(a)
    out = np.empty_like(a, dtype=np.float64)

    mask = a >= 0
    out[mask] = 1 / (1 + np.exp(-a[mask]))
    exp_a = np.exp(a[~mask])  # Avoid overflow when a is very negative
    out[~mask] = exp_a / (1 + exp_a)
    return out

def get_gradient(w: np.ndarray, data_t: np.ndarray, phi_X: np.ndarray) -> np.ndarray:
    # Compute gradient of binary cross-entropy loss
    logits = phi_X @ w
    probs = sigmoid(logits)
    error = probs - data_t
    return phi_X.T @ error  # Shape: (D, 1)

def get_BCE(w: np.ndarray, data_t: np.ndarray, phi_X: np.ndarray) -> float:
    logits = phi_X @ w
    probs = sigmoid(logits)
    probs = probs.flatten()
    t = data_t.flatten()

    eps = 1e-12
    y = np.clip(probs, eps, 1 - eps)

    loss = - (t * np.log(y) + (1 - t) * np.log(1 - y))
    return np.sum(loss) / loss.shape[0]

def train(
    data_x: np.ndarray, data_t: np.ndarray,
    val_x: np.ndarray, val_t: np.ndarray,
    lr: float, epochs: int
    ):
    # Train logistic regression using batch gradient descent
    phi_X = get_phi(data_x)
    phi_val = get_phi(val_x)
    w = np.zeros((phi_X.shape[1], 1), dtype=np.float64)  # Initialize weights

    train_accs = []
    val_accs = []
    train_BCE = []
    val_BCE = []

    for epoch in range(epochs):
        grad = get_gradient(w, data_t, phi_X)
        preds = inference(w, data_x)
        acc = (preds == data_t).mean()
        if acc < 0.85:
            learning_rate = 0.01
        else:
            learning_rate = lr 
        w -= learning_rate * grad  # Gradient descent step

        if (epoch + 1) % plotting_sample_period == 0:
            # If you want to speedup the training process, you can modify this.
            train_accs.append(acc)

            preds_val = inference(w, val_x)
            val_acc = (preds_val == val_t).mean()
            val_accs.append(val_acc)

            loss = get_BCE(w, data_t, phi_X)
            train_BCE.append(loss)

            loss_val = get_BCE(w, val_t, phi_val)
            val_BCE.append(loss_val)

        # Optional: Uncomment to print accuracy during training
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")

    return w, train_accs, val_accs, train_BCE, val_BCE # Return trained weights

def inference(w: np.ndarray, data_x: np.ndarray) -> np.ndarray:
    # Run inference and threshold predictions at 0.5
    phi_X = get_phi(data_x)
    logits = phi_X @ w
    probs = sigmoid(logits)

    y = np.empty_like(probs, dtype=np.int64)
    mask = probs >= 0.5
    y[mask] = 1
    y[~mask] = 0
    return y  # Binary predictions (0 or 1)

def k_fold(
    data_x: np.ndarray, data_t: np.ndarray, 
    k: int, lr: float, epochs: int, 
    shuffle=True, random_state: int | None = None,
    ):
    # Perform k-fold cross-validation and return best-performing weights
    N = data_x.shape[0]
    idx = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    # Determine sizes for each fold
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
        x_val, y_val     = data_x[val_idx],   data_t[val_idx]

        w, train_accs, val_accs, train_BCE, val_BCE = train(x_train, y_train, x_val, y_val, lr=lr, epochs=epochs)
        y_pred = inference(w, x_val)
        acc = (y_pred == y_val).mean()
        print(f"Fold {fold}/{k}  |  val accuracy = {acc:.4f}")

        # Track the best performing model
        if acc > best_acc:
            best_acc = acc
            best_w = w.copy()

    print(f"\nBest validation accuracy = {best_acc:.4f}")
    return best_w, train_accs, val_accs, train_BCE, val_BCE  # Return best model weights

# Load training data
data_x, data_y = read_train_data()

# Train using 4-fold cross-validation
w, train_accs, val_accs, train_BCE, val_BCE = k_fold(data_x, data_y, k=4, lr=0.0005, epochs=1000, random_state=114514810)

# Load test data and run inference
test_x = read_test_data()
y_pred = inference(w, test_x)

# Save predictions to file
save_result(y_pred)

plt.plot(np.arange(1, len(train_accs) + 1), train_accs, label='Train Acc')
plt.plot(np.arange(1, len(val_accs) + 1), val_accs, label = 'Val Acc')
plt.xlabel(f"per {plotting_sample_period} Epochs")
plt.ylabel("Training Accuracy")
plt.title("Learning Curve (Training Accuracy)")
plt.legend()
plt.grid(True)
plt.savefig("image1-1.png")
plt.close()

plt.plot(np.arange(1, len(train_BCE) + 1), train_BCE, label = 'Train Loss')
plt.plot(np.arange(1, len(val_BCE) + 1), val_BCE, label = 'Val Loss')
plt.xlabel(f"per {plotting_sample_period} Epochs")
plt.ylabel("Training Loss")
plt.title("Learning Curve (Training Loss)")
plt.legend()
plt.grid(True)
plt.savefig("image1-2.png")
plt.close()