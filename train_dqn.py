import copy
import math
import random
import sys

import gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils.utils
from dqn.agent import DQN, DQNAgent

# 超参数
n_state = 2  # 状态
n_action = 3  # 动作
n_hidden = 256  # dqn隐藏层dim
gamma = 0.99  # q值更新的折扣因子, gamma后继状态的效力就越大。
lr = 0.001  # 学习率
batch_size = 64  # 批
buf_size = 1000000  # 经验缓冲
sync_freq = 100  # 同步间隔
epsilon_min = 0.01  # 采样的最小epsilon
epsilon_decay = 0.999  # epsilon衰减率
DDQN = False  # 是否使用DDQN
exp_name = 'exp2'  # 实验名称
if __name__ == '__main__':
    env = gym.envs.make('MountainCar-v0')
    env = env.unwrapped  # 取消epoch、step限制(200, 200)
    # 模型及训练超参数设置
    model = DQN(n_state=n_state, n_action=n_action, n_hidden=n_hidden)
    model = model.cuda()
    agent = DQNAgent(env, gamma=gamma, lr=lr, batch_size=batch_size, buf_size=buf_size, sync_freq=sync_freq,
                     epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, DDQN=DDQN, exp_name=exp_name)
    print(model)
    epochs = 1000
    step = 0
    scores, episodes, avg_scores, goals, qs, qs_fix = [], [], [], [], [], []  # 记录每个episode的分数
    goal = -110  # >= -110才表示模型学习成功
    for epoch in tqdm(range(epochs), file=sys.stdout):
        score = 0
        state = env.reset()[0]  # 重置环境
        while True:
            action = agent.sample_act(state)  # 采样动作
            nxt_state, reward, done, _, info = env.step(action)  # 执行动作，获得env的反馈
            score += reward
            agent.save_transition(state, action, reward, nxt_state, done)  # save buf
            step += 1
            agent.learn(epoch, step)  # learn
            state = nxt_state
            if done or _:  # terminated
                print('epoch: {} end'.format(epoch))
                break
        scores.append(score)
        goals.append(goal)
        episodes.append(epoch)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(epoch, epochs, score, agent.epsilon,
                                                                         avg_score))
        # 保存q值
        qs.append(agent.get_q(state, action))
        qs_fix.append(agent.get_q(np.array([0.2, 0]), 0))
        # 保存模型
        if avg_score > -110 and score > -100:
            torch.save(agent.model.state_dict(),
                       './result/dqn/{0}/dqn_epoch{1}_avgScore{2}.pth'.format(exp_name, epoch + 1, avg_score))
    utils.utils.plt_graph(episodes, scores, avg_scores, goals, 'MountainCar-v0', 'DQN', 'train',
                          './result/dqn/' + exp_name)

    # 保存q值
    qs = np.array(qs)
    qs_fix = np.array(qs_fix)
    np.save('qs1.npy', qs)
    np.save('qs_fix1.npy', qs_fix)
