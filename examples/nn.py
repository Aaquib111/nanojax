"""Simple 2-layer neural network to fit a sine curve using nanojax."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nanojax import grad


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


def forward(x, w1, b1, w2, b2):
    """2-layer neural network forward pass."""
    # Layer 1
    h = relu(x @ w1 + b1)
    # Layer 2 (output)
    y = h @ w2 + b2
    return y


def mse_loss(pred, target):
    """Mean squared error loss."""
    return np.mean((pred - target) ** 2)


def main():
    # Generate training data
    np.random.seed(42)
    x_train = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y_train = np.sin(x_train)

    # Initialize parameters
    hidden_size = 32
    w1 = np.random.randn(1, hidden_size) * 0.1
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, 1) * 0.1
    b2 = np.zeros(1)

    # Define loss function
    def loss_fn(w1, b1, w2, b2):
        pred = forward(x_train, w1, b1, w2, b2)
        return mse_loss(pred, y_train)

    # Create gradient function
    grad_fn = grad(loss_fn, argnums=(0, 1, 2, 3), grad_direction=None)

    # Training loop
    learning_rate = 0.01
    num_steps = 1000

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

            # Compute loss for logging
            loss = loss_fn(w1, b1, w2, b2)
            losses.append(loss)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.6f}"})

    # Generate predictions for plotting
    x_test = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
    y_test = np.sin(x_test)
    y_pred = forward(x_test, w1, b1, w2, b2)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss
    ax1.plot(losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss over Time")
    ax1.grid(True)

    # Plot 2: Sine curve fit
    ax2.plot(x_test, y_test, label="True sin(x)", linewidth=2)
    ax2.plot(x_test, y_pred, label="NN prediction", linewidth=2, linestyle="--")
    ax2.scatter(x_train, y_train, alpha=0.3, s=10, label="Training data")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Sine Curve Fitting")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("sine_fit_results.png", dpi=150)
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print("Plot saved to sine_fit_results.png")
    plt.show()


if __name__ == "__main__":
    main()
