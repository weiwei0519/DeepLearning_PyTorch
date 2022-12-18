# coding=UTF-8
# 文本数据集的预处理，文本向量化

'''
@File: text_process_torch
@Author: WeiWei
@Time: 2022/12/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import numpy as np
import torch

print(torch.__version__)
print(np.__version__)

with open('./Datasets/NLP/1342-0.txt', encoding='utf8') as f:
    text = f.read()
lines = text.split('\n')
line = lines[200]
print(line)
letter_t = torch.zeros(len(line), 128)  # ASCII为128个字符限制
print(letter_t.shape)

# one-hot编码字符
letter_index = torch.zeros(len(line)).long()
for i, letter in enumerate(line.lower().strip()):
    letter_index[i] = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index[i].long()] = 1
print(letter_t)
print(letter_index)

# 用scatter函数进行one-hot编码字符
letter_t.scatter_(dim=1, index=letter_index.unsqueeze(1).long(), value=1.0)
print(letter_t)


# one-hot编码单词
def words_split(text):
    punctuation = '.,;:"!?“_-,'
    word_list = text.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


words_a_line = words_split(line)
print(words_a_line)

words = sorted(set(words_split(text)))
word2index = {word: i for (i, word) in enumerate(words)}
index2words = {i: word for (i, word) in enumerate(words)}
print("words number: " + str(len(word2index)))
print("words '{0}' index is: {1}".format("impossible", word2index['impossible']))

for i, word in enumerate(words_a_line):
    print("{0}: {1}".format(word, word2index[word]))
