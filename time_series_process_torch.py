# coding=UTF-8
# 基于PyTorch的时间序列数据集的预处理

'''
@File: time_series_process_torch
@Author: WeiWei
@Time: 2022/12/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import numpy as np
import torch

print(torch.__version__)
print(np.__version__)

bikes_numpy = np.loadtxt("./Datasets/CSV/hour-fixed.csv", dtype=np.float32, delimiter=",", skiprows=1,
                         converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
print(bikes.shape, bikes.stride())  # stride是tensor存储的步长统计函数，返回一个元组，值分别是同一维度相邻数字之间的步长。

# view函数相当于np.resize()，用来重构tensor的维度，参数值-1意味着，以其它已确认的维度，进行自动补齐。
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape, daily_bikes.stride())

# transpose转置函数
daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())

# 分类数据one-hot编码
# scatter，one-hot编码函数，dim是对哪个维度编码，index是对那一列进行编码，index的张量须于输出张量同纬度，value是编码的具体数值。
one_day = bikes[:24].long()
oneday_weather_onehot = torch.zeros(one_day.shape[0], 4).scatter(dim=1,
                                                                 index=one_day[:, 9].unsqueeze(1).long() - 1,
                                                                 value=1.0)
print(oneday_weather_onehot)

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
# scatter_()函数与scatter()的区别是，带下划线的函数，在原tensor对象上做更新，不带下划线的是输出新的tensor对象
daily_weather_onehot.scatter_(dim=1,
                              index=daily_bikes[:, 9, :].unsqueeze(1).long() - 1,
                              value=1.0)
print(daily_weather_onehot.shape)

# 将onehot编码拼接到原始tensor上
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)
print(daily_bikes.shape)

# 除了用one-hot编码之外，还可以用归一化来进行处理，针对temporary
temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min))  # 均差
print(daily_bikes[0:24, :, :])
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp)) / torch.std(temp))  # 均标准差
print(daily_bikes[0:24, :, :])
