# 这段代码定义了一个具有优先级的经验回放缓冲区（preMemory），
# 它与普通经验回放缓冲区不同，可以根据样本的TD误差来调整样本的优先级。
# 这允许在训练时更关注那些对网络预测误差大的样本，以提高训练效果。

import random
from Memory.sum_tree import SumTree as ST

class preMemory(object):
    e = 0.05  # 用于计算优先级的小常数，避免优先级为0的情况

    def __init__(self, capacity=8000, pr_scale=0.5):
        self.capacity = capacity  # 经验回放缓冲区的容量
        self.memory = ST(self.capacity)  # 使用SumTree作为经验回放缓冲区的数据结构
        self.pr_scale = pr_scale  # 优先级的缩放因子，用于调整优先级的分布
        self.max_pr = 0  # 记录最大的优先级，初始值为0

    # 计算优先级
    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale  # 根据TD误差计算优先级

    # 存储训练样本和对应的优先级
    def remember(self, sample, error):
        p = self.get_priority(error)  # 计算优先级
        self_max = max(self.max_pr, p)  # 更新最大优先级
        self.memory.add(self_max, sample)  # 将样本和优先级添加到经验回放缓冲区

    # 从缓冲区中随机采样n个样本，返回样本、对应的索引和优先级
    def sample(self, n):
        sample_batch = []  # 存储采样的样本
        sample_batch_indices = []  # 存储样本的索引
        sample_batch_priorities = []  # 存储样本的优先级
        num_segments = self.memory.total() / n  # 计算分段的数量

        for i in range(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)  # 在分段范围内随机选择一个值
            idx, pr, data = self.memory.get(s)  # 获取优先级对应的样本
            sample_batch.append((idx, data))
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    # 更新缓冲区中样本的优先级
    def update(self, batch_indices, errors):
        for i in range(len(batch_indices)):
            p = self.get_priority(errors[i])  # 根据新的TD误差计算优先级
            self.memory.update(batch_indices[i], p)  # 更新样本的优先级
