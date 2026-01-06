"""Plot a Gaussian and its first two derivatives using nanojax."""

import matplotlib.pyplot as plt
import numpy as np

from nanojax import grad


def gaussian(x):
    """Standard Gaussian: exp(-x^2/2) / sqrt(2π)"""
    return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


def main():
    x = np.linspace(-4, 4, 200)
    y = gaussian(x)

    # Gradients: Compute with grad()
    grad_gaussian = grad(gaussian)
    dy = np.array([grad_gaussian(np.array(xi)) for xi in x])
    grad2_gaussian = grad(grad_gaussian)
    d2y = np.array([grad2_gaussian(np.array(xi)) for xi in x])

    plt.figure(figsize=(7, 3))
    plt.style.use("seaborn-v0_8-bright")
    plt.plot(x, y, label="Gaussian", linewidth=2)
    plt.plot(x, dy, label="First derivative", linewidth=2, linestyle="--")
    plt.plot(x, d2y, label="Second derivative", linewidth=2, linestyle=":")
    plt.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel("x", fontsize=19)
    plt.ylabel("y", fontsize=18)
    plt.title("Gaussian Derivatives", fontsize=20)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()

    plt.savefig(
        "examples/images/gaussian_derivatives.png", dpi=150, bbox_inches="tight"
    )
    print("Plot saved to gaussian_derivatives.png")


if __name__ == "__main__":
    main()
