import torch
import torch.nn as nn #神经网络
from Agent.RL_network import CNN_FNN,CNN_dueling
from Memory.Memory import Memory
from Memory.PreMemory import preMemory
import numpy as np

class Agent():
    def __init__(self,n,O_max_len,dueling=False,double=False,PER=False):
        #dueling、double 和 PER 都是布尔型的可选参数，用于配置代理的不同设置
        self.double=double
        self.PER=PER
        self.GAMMA=1#代理在计算未来奖励的折现时使用的折现因子。
        self.n=n
        self.O_max_len=O_max_len
        super(Agent,self).__init__()
        if dueling:
            self.eval_net,self.target_net=CNN_dueling(self.n,self.O_max_len),CNN_dueling(self.n,self.O_max_len)
        else:
            self.eval_net,self.target_net=CNN_FNN(self.n,self.O_max_len),CNN_FNN(self.n,self.O_max_len)
        self.Q_NETWORK_ITERATION=100#表示在多少步之后更新目标网络
        self.BATCH_SIZE=256#每次训练时使用的批量大小
        self.learn_step_counter=0#跟踪学习步数
        self.memory_counter=0#跟踪记忆库中的样本数量
        if PER:
            self.memory=preMemory()
        else:
            self.memory=Memory()
        self.EPISILO=0.8#表示某种策略参数的初始值。
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.00001)#使用 Adam 优化算法来优化代理的神经网络参数
        self.loss_func = nn.MSELoss()#代码创建了一个均方误差（MSE）损失函数对象 self.loss_func，用于计算损失

    def choose_action(self, state):
        #这行代码将输入的 state 重新形状为一个四维张量，维度为 (-1, 3, self.n, self.O_max_len)。
        #这通常是为了适应神经网络的输入要求，其中 -1 表示该维度将根据其他维度的大小自动计算。
        state=np.reshape(state,(-1,3,self.n,self.O_max_len))
        state=torch.FloatTensor(state)
        if np.random.randn() <= self.EPISILO:
            action_value=self.eval_net.forward(state)#通过调用神经网络模型 self.eval_net 对输入状态 state 进行前向传播，以获取每个可能动作的估值。
            action = torch.max(action_value, 1)[1].data.numpy()[0]#选择具有最高估值的动作作为最优动作。它通过 torch.max 函数找到最大估值的索引，然后使用 data.numpy() 将结果转换为 NumPy 数组，并取第一个元素作为动作。这实现了贪婪策略，即选择估值最高的动作。
        else:
            action=np.random.randint(0,17)

        self.EPISILO = min(0.001, self.EPISILO - 0.00001)
        return action

    def PER_error(self,state,action,reward,next_state):
        #用于计算经验优先级（Priority）误差的方法，通常用于实现优先级经验回放（PER）的深度强化学习算法中。
        #该方法的目的是为了为每个经验样本分配一个优先级，以便在经验回放中更有选择性地采样和训练那些对智能体学习更有帮助的样本

        state=torch.FloatTensor(np.reshape(state,(-1,3,self.n,self.O_max_len)))
        #这行代码将输入的 next_state 也重新形状为一个四维张量并转换为 PyTorch 的浮点数张量。
        next_state=torch.FloatTensor(np.reshape(next_state, (-1, 3, self.n, self.O_max_len)))
        #获取当前状态下每个可能动作的估值
        p=self.eval_net.forward(state)
        #这行代码获取下一个状态 next_state 下每个可能动作的估值
        p_ = self.eval_net.forward(next_state)
        p_target=self.target_net(state)

        if self.double:#检查是否启用了双重 Q 网络，如果启用，则执行以下代码块，否则执行下一个分支
            q_a=p_.argmax(dim=1)#具有最大估值的动作的索引
            q_a = torch.reshape(q_a, (-1, len(q_a)))#将 q_a 重新形状为一个二维张量
            qt = reward + self.GAMMA * p_target.gather(1, q_a)
        else:
            #这行代码计算目标 Q 值 qt，其中 reward 是当前时刻的奖励，p_target.max(1)[0].view(self.BATCH_SIZE, 1) 表示从目标网络中获取最大估值的动作对应的估值。
            qt = reward + self.GAMMA * p_target.max(1)[0].view(self.BATCH_SIZE, 1)
        qt = qt.detach().numpy()
        p = p.detach().numpy()
        errors = np.abs(p[0][action] - qt[0][0])
        return errors

    def store_transition(self, state, action, reward, next_state):#store_transition 方法用于将智能体与环境的交互结果（状态、动作、奖励、下一个状态）存储到经验记忆中。
        if self.PER:#检查是否启用了优先经验回放（PER）
            #这行代码调用 PER_error 方法来计算 Q 学习中的误差（PER 方式计算），这个误差将用于指导经验记忆的存储。这个误差表示了当前动作估值与目标估值之间的差距。
            errors = self.PER_error(state, action, reward, next_state)
            self.memory.remember((state, action, reward, next_state), errors)
            self.memory_counter += 1
        else:
            self.memory.remember((state, action, reward, next_state))
            self.memory_counter += 1

    def learn(self):
        # 更新神经网络参数
        if self.learn_step_counter % self.Q_NETWORK_ITERATION == 0:
            # 将目标网络的参数更新为当前评估网络的参数
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从经验回放中随机采样一个批次
        batch = self.memory.sample(self.BATCH_SIZE)

        # 从批次中提取状态、下一个状态、动作、奖励等信息
        batch_state = np.array([o[0] for o in batch])
        batch_next_state = np.array([o[3] for o in batch])
        batch_action = np.array([o[1] for o in batch])
        batch_reward = np.array([o[1] for o in batch])

        # 转换动作和奖励为 PyTorch 张量
        batch_action = torch.LongTensor(np.reshape(batch_action, (-1, len(batch_action))))
        batch_reward = torch.LongTensor(np.reshape(batch_reward, (-1, len(batch_reward))))

        # 转换状态和下一个状态为 PyTorch 张量
        batch_state = torch.FloatTensor(np.reshape(batch_state, (-1, 3, self.n, self.O_max_len)))
        batch_next_state = torch.FloatTensor(np.reshape(batch_next_state, (-1, 3, self.n, self.O_max_len)))

        if self.double:
            # 对于双重 DQN，计算当前状态的 Q 值
            q_eval = self.eval_net(batch_state).gather(1, batch_action)

            # 计算下一个状态的 Q 值（在不更新评估网络的情况下）
            q_next_eval = self.eval_net(batch_next_state).detach()

            # 获取目标网络下一个状态的 Q 值，并选择最优动作
            q_next = self.target_net(batch_next_state).detach()
            q_a = q_next_eval.argmax(dim=1)
            q_a = torch.reshape(q_a, (-1, len(q_a)))

            # 计算目标 Q 值
            q_target = batch_reward + self.GAMMA * q_next.gather(1, q_a)
        else:
            # 对于单一 DQN，计算当前状态的 Q 值
            q_eval = self.eval_net(batch_state).gather(1, batch_action)

            # 计算目标 Q 值（在不更新目标网络的情况下）
            q_next = self.target_net(batch_next_state).detach()

            # 计算目标 Q 值
            q_target = batch_reward + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)

        # 计算损失函数
        loss = self.loss_func(q_eval, q_target)

        # 清零优化器的梯度
        self.optimizer.zero_grad()

        # 反向传播并更新神经网络参数
        loss.backward()
        self.optimizer.step()






















