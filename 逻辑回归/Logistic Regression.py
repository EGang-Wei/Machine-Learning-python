import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ========= 1.读取数据 =========
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print("数据前5行：")
print(data.head())

# 区分正负样本：录取=1，未录取=0
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

# 可视化原始数据分布
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


# ========= 2.定义sigmoid函数 =========
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ========= 3.逻辑回归代价函数 =========
def cost(theta, X, y):
    theta = np.matrix(theta)   # 参数θ转为矩阵
    X = np.matrix(X)           # 特征矩阵X转为矩阵，每行是一个样本
    y = np.matrix(y)           # 真实标签y转为矩阵，取值0或1
    # sigmoid(X * theta.T)：模型预测概率 h_θ(x)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# ========= 4.构造X,y矩阵，增加偏置项Ones =========
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

theta = np.zeros(3)   # theta0,theta1,theta2

X = np.array(X.values)
y = np.array(y.values)

# 初始代价（theta全为0）
print("\n初始代价函数值(theta全0):", cost(theta, X, y))


# ========= 5.梯度计算函数（不是梯度下降，只算梯度） =========
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


# ========= 6.使用fmin_tnc优化器求解theta =========
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print("\n优化结果result：", result)
best_theta = result[0]
print("最优theta参数：", best_theta)
print("优化后代价函数值：", cost(best_theta, X, y))


# ========= 7.绘制决策边界 =========
plotting_x1 = np.linspace(30, 100, 100)
# 决策边界： theta0 + theta1*x1 + theta2*x2 = 0  => x2 = -(theta0+theta1*x1)/theta2
plotting_h1 = (- best_theta[0] - best_theta[1] * plotting_x1) / best_theta[2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Decision Boundary')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


# ========= 8.预测：考试1=45分，考试2=55分，录取概率 =========
prob = sigmoid(best_theta.dot([1, 45, 55]))
print(f"\n考试45,55的录取概率：{prob:.4f}")