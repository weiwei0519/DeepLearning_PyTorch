# coding=UTF-8
# PyTorch中的预训练模型

'''
@File: pretrained_models
@Author: WeiWei
@Time: 2022/12/15
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from torchvision import models

# 查看PyTorch支持的预训练模型
for model in dir(models):
    print(model)
