"""Simple 2-layer neural network to fit a sine curve using nanojax."""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nanojax import grad


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


def forward(x, w1, b1, w2, b2):
    """2-layer neural network forward pass."""
    h = relu(x @ w1 + b1)
    y = h @ w2 + b2
    return y


def mse_loss(pred, target):
    """Mean squared error loss."""
    return np.mean((pred - target) ** 2)


def main():
    np.random.seed(42)
    x_train = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y_train = np.sin(x_train)

    # Param init
    hidden_size = 32
    w1 = np.random.randn(1, hidden_size) * 1.0
    b1 = np.random.randn(hidden_size) * 0.5
    w2 = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
    b2 = np.zeros(1)

    # Define loss function and its gradient
    def loss_fn(w1, b1, w2, b2):
        pred = forward(x_train, w1, b1, w2, b2)
        return mse_loss(pred, y_train)

    grad_fn = grad(loss_fn, argnums=(0, 1, 2, 3), grad_direction=None)

    learning_rate = 0.05
    num_steps = 2000
    losses = []
    with tqdm(range(num_steps), desc="Training") as pbar:
        for step in pbar:
            # Compute gradients
            dw1, db1, dw2, db2 = grad_fn(w1, b1, w2, b2)

            # Update parameters
            w1 = w1 - learning_rate * dw1
            b1 = b1 - learning_rate * db1
            w2 = w2 - learning_rate * dw2
            b2 = b2 - learning_rate * db2
            loss = loss_fn(w1, b1, w2, b2)
            losses.append(loss)

            pbar.set_postfix({"loss": f"{loss:.6f}"})

    # Generate predictions
    x_test = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
    y_test = np.sin(x_test)
    y_pred = forward(x_test, w1, b1, w2, b2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use("seaborn-v0_8-bright")

    # Losses plot
    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set_xlabel("Step", fontsize=18)
    ax1.set_ylabel("Loss", fontsize=18)
    ax1.set_title("Training Loss over Time", fontsize=20)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.grid(True)

    # Fit plot
    ax2.plot(x_test, y_test, label="True sin(x)", linewidth=2)
    ax2.plot(x_test, y_pred, label="NN prediction", linewidth=2, linestyle="--")
    ax2.scatter(x_train, y_train, alpha=0.3, s=10, label="Training data")
    ax2.set_xlabel("x", fontsize=18)
    ax2.set_ylabel("y", fontsize=18)
    ax2.set_title("Sine Curve Fitting", fontsize=20)
    ax2.tick_params(axis="both", which="major", labelsize=14)
    ax2.legend(fontsize=16)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("examples/images/sine_fit_results.png", dpi=150)
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print("Plot saved to sine_fit_results.png")
    plt.show()


if __name__ == "__main__":
    main()
