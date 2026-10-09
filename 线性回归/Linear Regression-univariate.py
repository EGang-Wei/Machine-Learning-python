import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).with_name("ex1data1.txt")
OUTPUT_DIR = Path(__file__).with_name("output")
ALPHA = 0.01
ITERATIONS = 1500


def load_data(path):
    """Read population-profit data for univariate linear regression."""
    return pd.read_csv(path, header=None, names=["Population", "Profit"])


def prepare_features(data):
    """Build X with an intercept column and y as a column vector."""
    population = data["Population"].to_numpy()
    profit = data["Profit"].to_numpy()
    x = np.c_[np.ones(population.shape[0]), population]
    y = profit.reshape(-1, 1)
    return x, y


def compute_cost(x, y, theta):
    """Compute mean squared error cost J(theta)."""
    m = len(y)
    errors = x @ theta - y
    return ((errors.T @ errors) / (2 * m)).item()


def gradient_descent(x, y, theta, alpha, iterations):
    """Train theta with batch gradient descent."""
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        errors = x @ theta - y
        gradient = (x.T @ errors) / m
        theta = theta - alpha * gradient
        cost_history[i] = compute_cost(x, y, theta)

    return theta, cost_history


def predict(population, theta):
    """Predict profit for a city population measured in 10,000s."""
    return theta[0, 0] + theta[1, 0] * population


def print_analysis(data, x, y, theta, cost_history):
    """Print a compact text analysis of the fitted linear model."""
    predictions = x @ theta
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print("数据概览：")
    print(data.describe())
    print()
    print("训练结果：")
    print(f"样本数量: {len(data)}")
    print(f"学习率 alpha: {ALPHA}")
    print(f"迭代次数: {ITERATIONS}")
    print(f"theta0 截距: {theta[0, 0]:.6f}")
    print(f"theta1 斜率: {theta[1, 0]:.6f}")
    print(f"最终模型: Profit = {theta[0, 0]:.6f} + {theta[1, 0]:.6f} * Population")
    print(f"第一次迭代后代价: {cost_history[0]:.6f}")
    print(f"最终代价: {cost_history[-1]:.6f}")
    print(f"R^2: {r_squared:.6f}")
    print()
    print("预测示例：")
    print(f"人口 35,000 的城市，预测利润: {predict(3.5, theta) * 10000:.2f}")
    print(f"人口 70,000 的城市，预测利润: {predict(7.0, theta) * 10000:.2f}")


def plot_regression(data, theta):
    """Plot the training data and the fitted regression line."""
    x_line = np.linspace(data["Population"].min(), data["Population"].max(), 100)
    y_line = predict(x_line, theta)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data["Population"], data["Profit"], label="Training data", color="#2563eb")
    ax.plot(x_line, y_line, label="Linear regression", color="#dc2626", linewidth=2)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")
    ax.set_title("Profit Prediction by City Population")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_cost(cost_history):
    """Plot how the cost decreases during training."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(cost_history) + 1), cost_history, color="#16a34a")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J(theta)")
    ax.set_title("Cost Reduction During Gradient Descent")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_data(DATA_FILE)
    x, y = prepare_features(data)
    initial_theta = np.zeros((2, 1))

    theta, cost_history = gradient_descent(x, y, initial_theta, ALPHA, ITERATIONS)

    print_analysis(data, x, y, theta, cost_history)

    OUTPUT_DIR.mkdir(exist_ok=True)
    regression_fig = plot_regression(data, theta)
    cost_fig = plot_cost(cost_history)
    regression_fig.savefig(OUTPUT_DIR / "linear_regression_fit.png", dpi=150)
    cost_fig.savefig(OUTPUT_DIR / "cost_history.png", dpi=150)
    print()
    print(f"图像已保存到: {OUTPUT_DIR}")

    if os.environ.get("NO_SHOW") != "1":
        plt.show()
