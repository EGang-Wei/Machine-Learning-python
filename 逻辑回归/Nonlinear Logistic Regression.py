import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as opt

# 读取数据
path = 'ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
print("数据前5行：")
print(data.head())

# 区分正负样本：接受=1，不接受=0
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

# 可视化原始数据分布
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Not Accepted')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

# sigmiod函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 逻辑回归代价函数
def cost(theta, X, y):
    X = np.array(X)
    y = np.array(y)
    theta = np.array(theta)

# sigmoid(X @ theta.T)：模型预测概率 h_θ(x)

    h = sigmoid(X @ theta.T)
    first = np.multiply(y , np.log(h))
    second = np.multiply((1 - y) , np.log(1 - h))
    return -np.sum(first + second) / (len(X))

# 构造X，y矩阵，添加矩阵映射

data.insert(0, 'ones', 1)
data.insert(3, 'Text 1 square', data.iloc[:, 1] * data.iloc[:, 1])
data.insert(4, 'Text 2 square', data.iloc[:, 2] * data.iloc[:, 2])
data.insert(5, 'Text 1 * Text 2', data.iloc[:, 1] * data.iloc[:, 2])
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1]
theta = np.zeros(cols-1)

X = X.to_numpy()
y = y.to_numpy()

print("\n初始代价函数值（theta为0）：", cost(theta, X, y))

# 梯度下降计算函数
def gradient(theta, X, y):
    X = np.array(X)
    y = np.array(y)
    theta = np.array(theta)
    parameters = len(theta)
    grad = np.zeros(parameters)
    error = -y + sigmoid(X @ theta.T)
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print("\n优化结果result：", result)
best_theta = result[0]
print("最优theta参数：", best_theta)
print("优化后代价函数值：", cost(best_theta, X, y))

# 绘制分类边界

# 提取原始数据
positive = data[data['Accepted'] == 1]
negative = data[data['Accepted'] == 0]

# 创建网格
x1_min = data['Test 1'].min() - 0.1
x1_max = data['Test 1'].max() + 0.1
x2_min = data['Test 2'].min() - 0.1
x2_max = data['Test 2'].max() + 0.1

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

# 构造网格对应的二次多项式特征
# 特征顺序必须与训练 X 时完全一致
grid_features = np.column_stack([
    np.ones(xx1.size),
    xx1.ravel(),
    xx2.ravel(),
    xx1.ravel() ** 2,
    xx2.ravel() ** 2,
    xx1.ravel() * xx2.ravel()
])

# 计算网格上的模型预测概率
probability = sigmoid(grid_features @ best_theta)

# 恢复网格形状
Z = probability.reshape(xx1.shape)

# 绘图
plt.figure(figsize=(10, 7))

plt.scatter(
    positive['Test 1'],
    positive['Test 2'],
    c='b',
    marker='+',
    s=60,
    label='Accepted'
)

plt.scatter(
    negative['Test 1'],
    negative['Test 2'],
    c='r',
    marker='o',
    s=40,
    label='Not Accepted'
)

# 分类边界：预测概率为 0.5
plt.contour(
    xx1,
    xx2,
    Z,
    levels=[0.5],
    colors='black',
    linewidths=2
)

plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(alpha=0.2)
plt.show()









