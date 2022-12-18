# coding=UTF-8
# 用PyTorch实现线性回归模型，使用自动求导的特性

'''
@File: linear_reg_torch_autograd
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


# model fit，模型训练函数
def fit(epochs, learning_rate, params, x, y):
    for epoch in range(1, epochs + 1):
        if params.grad is not None:
            params.grad.zero_()  # 经过一轮迭代训练，需要将grad置为0，并重新计算导数
        y_ = linear_regression_model(x, *params)
        loss = linear_regression_loss(y_, y)
        loss.backward()
        # 使用梯度，更新参数值
        with torch.no_grad():
            params -= learning_rate * params.grad
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
# 初始化参数，开启自动求导特性
params = torch.tensor([1.0, 0.0], requires_grad=True)
params = fit(epochs, learning_rate, params, x, y)
print(f'Params(w, b):  {params}')

# 训练结果可视化
y_ = linear_regression_model(x, *params)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (Fahrenheit)")
plt.ylabel("Temperature (Celsius)")
plt.plot(x.numpy(), y_.detach().numpy())
plt.plot(x.numpy(), y.numpy(), 'x')
plt.show()
