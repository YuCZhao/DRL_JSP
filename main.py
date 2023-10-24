import matplotlib.pyplot as plt
from JSP_env import JSP_Env
from action_space import Dispatch_rule
from Dataset.data_extract import change
from Agent.Agent import Agent

def main(Agent, env, batch_size):
    Reward_total = []  # 存储每个周期的总回报
    C_total = []  # 存储每个周期的完成时间
    rewards_list = []  # 存储每个周期内每步的回报
    C = []  # 存储每个周期的完成时间

    episodes = 8000  # 训练周期的总数
    print("Collecting Experience....")
    for i in range(episodes):  # 循环训练周期
        print(i)
        state, done = env.reset()  # 重置环境并获取初始状态
        ep_reward = 0  # 初始化当前周期内的总回报为0
        while True:  # 开始一个周期的训练
            action = Agent.choose_action(state)  # 代理根据当前状态选择动作

            a = Dispatch_rule(action, env)  # 根据选择的动作执行作业调度
            try:
                next_state, reward, done = env.step(a)  # 执行动作，获取下一个状态、奖励和是否完成的信息
            except:
                print(action, a)

            Agent.store_transition(state, action, reward, next_state)  # 存储经验到代理的经验池中
            ep_reward += reward  # 累积当前周期内的总回报
            if Agent.memory_counter >= batch_size:  # 如果经验池已满
                Agent.learn()  # 调用代理的学习方法进行学习
                if done and i % 1 == 0:  # 如果一个周期完成，并且周期数符合条件
                    ret, f, C1, R1 = evaluate(i, Agent, env)  # 评估代理的性能
                    Reward_total.append(R1)  # 存储总回报
                    C_total.append(C1)  # 存储完成时间
                    rewards_list.append(ep_reward)  # 存储当前周期内的总回报
                    C.append(env.C_max())  # 存储完成时间
            if done:  # 如果一个周期完成
                break  # 结束当前周期的训练
            state = next_state  # 更新当前状态为下一个状态
    x = [_ for _ in range(len(C))]  # 创建 x 值，用于绘制图形
    plt.plot(x, rewards_list)  # 绘制总回报图
    plt.plot(x, C)  # 绘制完成时间图
    return Reward_total, C_total  # 返回总回报和完成时间的列表


def evaluate(i, Agent, env):
    returns = []  # 存储多次评估的总回报
    C = []        # 存储多次评估的完成时间

    for total_step in range(10):  # 执行多次评估，每次评估执行10步
        state, done = env.reset()  # 重置环境并获取初始状态
        ep_reward = 0             # 初始化当前评估周期内的总回报为0

        while True:  # 开始一个评估周期
            action = Agent.choose_action(state)  # 代理根据当前状态选择动作

            a = Dispatch_rule(action, env)  # 根据选择的动作执行作业调度
            try:
                next_state, reward, done = env.step(a)  # 执行动作，获取下一个状态、奖励和是否完成的信息
            except:
                print(action, a)

            ep_reward += reward  # 累积当前评估周期内的总回报
            if done == True:     # 如果评估周期完成
                fitness = env.C_max()  # 获取完成时间（C_max）
                C.append(fitness)      # 存储完成时间
                break  # 结束当前评估周期

        returns.append(ep_reward)  # 存储当前评估周期内的总回报

    # 打印评估结果
    print('time step:', i, 'Reward:', sum(returns) / 10, 'C_max:', sum(C) / 10)

    # 返回当前评估周期内的平均回报、平均完成时间、完成时间列表和回报列表
    return sum(returns) / 10, sum(C) / 10, C, returns



if __name__ == '__main__':
    import pickle
    import os

    n, m, PT, MT = change('la', 16)

    f=r'.\result\la'
    if not os.path.exists(f):
        os.mkdir(f)
    f1=os.path.join(f,'la'+'16')
    if not os.path.exists(f1):
        os.mkdir(f1)
    print(n, m, PT, MT)
    env = JSP_Env(n, m, PT, MT)
    # (0,0)CNN+FNN+DQN (1,0):CNN+Dueling network+DQN (0,1):CNN+FNN+DDQN (1,1):CNN+Dueling network+DDQN
    agent=Agent(env.n,env.O_max_len,1,1)
    Reward_total,C_total=main(agent,env,100)
    print(os.path.join(f1, 'C_max' + ".pkl"))
    with open(os.path.join(f1, 'C_max' + ".pkl"), "wb") as f2:
        pickle.dump(C_total, f2, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(f1, 'Reward' + ".pkl"), "wb") as f3:
        pickle.dump(Reward_total, f3, pickle.HIGHEST_PROTOCOL)

