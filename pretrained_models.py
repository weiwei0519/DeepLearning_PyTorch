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
from torchvision import transforms
from PIL import Image
import torch

print(torch.__version__)

# 查看PyTorch支持的预训练模型
for model in dir(models):
    print(model)

resnet = models.resnet101(pretrained=True)
print(resnet)

# 创建一个图片预处理函数
preprocess = transforms.Compose([
    transforms.Resize(256),  # 图片缩放到256 x 256
    transforms.CenterCrop(224),  # 围绕中心将图像裁剪为224 x 224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(  # 对RGB进行归一化处理，使其具有定义的均值和标准差。
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("./Datasets/Image/elephant.jpg")
img.show()
# 通过预处理管道，对图像进行预处理
img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)
print(batch_t)
# 在新数据上运行训练过的模型成为推理或预测，为了进行推理，需要将网络置为eval模式
resnet.eval()
out = resnet(batch_t)
print(out)
# 获得输出张量中得分最高所对应的索引。
_, index = torch.max(out, 1)
print(index)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(percentage[index[0]].item())
