import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# ========= 1.读取数据 =========
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print("数据总行数：", len(data))
print("数据前5行：")
print(data.head())

# 区分正负样本（仅用于画图）
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

# ========= 2.构造特征矩阵和标签 =========
# 加偏置项（全1列，不涉及数据统计，提前加无问题）
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1].values   # 特征矩阵 shape: (100, 3)
y = data.iloc[:, cols-1:cols].values # 标签列向量 shape: (100, 1)

# ========= 【核心新增】拆分训练集和测试集 =========
# 按 8:2 拆分，分层抽样保证正负样本比例，固定随机种子可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\n训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")
print(f"训练集正样本比例：{y_train.mean():.2%}")
print(f"测试集正样本比例：{y_test.mean():.2%}")

# ========= 3.定义 sigmoid、代价函数、梯度 =========
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = sigmoid(X * theta.T)
    first = np.multiply(-y, np.log(h))
    second = np.multiply((1 - y), np.log(1 - h))
    return np.sum(first - second) / len(X)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad

# ========= 4.【只用训练集】求解最优 theta =========
theta_init = np.zeros(3)
# 注意：这里只传入训练集 X_train 和 y_train！
result = opt.fmin_tnc(func=cost, x0=theta_init, fprime=gradient, args=(X_train, y_train))
best_theta = result[0]

print("\n最优theta参数：", best_theta)
print(f"训练集代价：{cost(best_theta, X_train, y_train):.4f}")
print(f"测试集代价：{cost(best_theta, X_test, y_test):.4f}")

# ========= 5.准确率评估 =========
def predict(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    prob = sigmoid(X * theta.T)
    return [1 if p >= 0.5 else 0 for p in prob]

train_pred = predict(best_theta, X_train)
test_pred = predict(best_theta, X_test)

train_acc = np.mean(np.array(train_pred) == y_train.flatten())
test_acc = np.mean(np.array(test_pred) == y_test.flatten())
print(f"\n训练集准确率：{train_acc:.2%}")
print(f"测试集准确率：{test_acc:.2%}")

# ========= 6.绘制决策边界 =========
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (-best_theta[0] - best_theta[1] * plotting_x1) / best_theta[2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', linewidth=2, label='Decision Boundary')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.set_title('Logistic Regression Decision Boundary')
plt.show()

# ========= 7.单样本预测 =========
prob = sigmoid(best_theta.dot([1, 45, 55]))
print(f"\n考试1=45, 考试2=55 的录取概率：{prob:.4f}")