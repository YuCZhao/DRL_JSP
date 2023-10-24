# 这段代码定义了一个优先级树（SumTree），
# 用于管理经验回放缓冲区中样本的优先级和数据。
# 它提供了添加样本、更新样本的优先级、获取样本的功能，以及一些内部方法来维护树结构和计算优先级。

import numpy

class SumTree(object):

    def __init__(self, capacity):
        self.write = 0  # 写入位置的索引
        self.capacity = capacity  # 缓冲区的容量
        self.tree = numpy.zeros(2 * capacity - 1)  # 存储优先级的树结构
        self.data = numpy.zeros(capacity, dtype=object)  # 存储数据的数组

    # 更新节点及其父节点的优先级
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # 计算父节点的索引

        self.tree[parent] += change  # 更新父节点的优先级

        if parent != 0:
            self._propagate(parent, change)  # 递归更新父节点的父节点

    # 根据优先级获取对应的样本索引
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # 返回树中所有优先级的总和
    def total(self):
        return self.tree[0]

    # 向树中添加样本的优先级和数据
    def add(self, p, data):
        idx = self.write + self.capacity - 1  # 计算样本在树中的索引

        self.data[self.write] = data  # 存储样本数据
        self.update(idx, p)  # 更新树中对应节点的优先级

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    # 更新树中某个节点的优先级
    def update(self, idx, p):
        change = p - self.tree[idx]  # 计算优先级变化

        self.tree[idx] = p  # 更新节点的优先级
        self._propagate(idx, change)  # 更新父节点的优先级

    # 获取样本的索引、优先级和数据
    def get(self, s):
        idx = self._retrieve(0, s)  # 根据优先级获取样本的索引
        dataIdx = idx - self.capacity + 1  # 计算样本在data数组中的索引

        return idx, self.tree[idx], self.data[dataIdx]  # 返回样本的索引、优先级和数据
