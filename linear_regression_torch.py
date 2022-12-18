# coding=UTF-8
# 用PyTorch实现线性回归模型

'''
@File: linear_regression_torch
@Author: WeiWei
@Time: 2022/12/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import torch
from matplotlib import pyplot as plt


# 定义模型函数
def linear_regression_model(x, w, b):
    return w * x + b


# 定义损失函数
def linear_regression_loss(y_, y):
    squared_diffs = (y_ - y) ** 2
    return squared_diffs.mean()


# gradient_loss = (d_loss/d_w, d_loss/d_b) = (d_loss/d_model * d_model/d_w, d_loss/d_model * d_model/d_b)

# 计算d_loss/d_model
def dloss_dmodel(y_, y):
    d_squ_diffs = 2 * (y_ - y) / y_.size(0)
    return d_squ_diffs


# 计算d_model/d_w
def dmodel_dw(x, w, b):
    return x


# 计算d_model/d_b
def dmodel_db(x, w, b):
    return 1.0


# 计算gradient_loss
def gradient_loss(x, w, b, y_, y):
    # d_loss/d_w
    dloss_dw = dloss_dmodel(y_, y) * dmodel_dw(x, w, b)
    # d_loss/d_b
    dloss_db = dloss_dmodel(y_, y) * dmodel_db(x, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


# model fit，模型训练函数
def fit(epochs, learning_rate, params, x, y):
    for epoch in range(1, epochs + 1):
        w, b = params
        y_ = linear_regression_model(x, w, b)
        loss = linear_regression_loss(y_, y)
        grad_loss = gradient_loss(x, w, b, y_, y)
        params = params - learning_rate * grad_loss
        if epoch % 10 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            print(f'Params(w, b):  {params}')
            print(f'Grad:          {params.grad}')
    return params


delta = 0.1
learning_rate = 1e-2
epochs = 5000
y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
x = torch.tensor(x) * 0.1
y = torch.tensor(y)
w = torch.ones(())
b = torch.zeros(())
y_ = linear_regression_model(x, w, b)
loss = linear_regression_loss(y_, y)
print(f"y_pred = {y_}")
print(f"loss = {loss}")
w, b = fit(epochs, learning_rate, torch.tensor([1.0, 0.0]), x, y)
print(f'Params(w, b):  {(w, b)}')

# 训练结果可视化
y_ = linear_regression_model(x, w, b)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (Fahrenheit)")
plt.ylabel("Temperature (Celsius)")
plt.plot(x.numpy(), y_.detach().numpy())
plt.plot(x.numpy(), y.numpy(), 'x')
plt.show()
