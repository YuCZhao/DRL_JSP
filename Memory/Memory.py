# 定义了一个经验回放缓冲区（Memory），
# 它可以存储训练样本，并提供随机采样功能，
# 用于从缓冲区中获取一批训练样本以供训练神经网络模型使用。

import random
from collections import deque

class Memory(object):
    # 初始化经验回放缓冲区
    def __init__(self, capacity=8000):
        # 设置缓冲区的容量，默认为8000
        self.capacity = capacity
        # 创建一个双端队列，用于存储训练样本，限制队列长度不超过容量
        self.memory = deque(maxlen=self.capacity)

    # 存储训练样本
    def remember(self, sample):
        # 将传入的样本添加到缓冲区
        self.memory.append(sample)

    # 从缓冲区中随机采样n个样本
    def sample(self, n):
        # 确保采样数量不超过缓冲区中的样本数量
        n = min(n, len(self.memory))
        # 使用随机采样函数从缓冲区中获取n个随机样本
        sample_batch = random.sample(self.memory, n)
        return sample_batch
