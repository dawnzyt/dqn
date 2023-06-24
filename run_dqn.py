import sys

import numpy as np
import torch
import gym
import tqdm

import utils.utils
from dqn.agent import DQN, DQNAgent
if_render = True
env = gym.envs.make('MountainCar-v0', render_mode='human' if if_render else 'rgb_array')
env = env.unwrapped  # 取消epoch、step限制
DQN = DQN(n_state=2, n_action=3, n_hidden=256)
DQN.load_state_dict(torch.load('./result/dqn/exp1/dqn_epoch519_avgScore-99.87.pth'))
DQN.eval()
num_episodes = 100  # 跑多少次episode
epsilon = 0.01
scores, episodes, avg_scores, goals = [], [], [], []  # 记录每个episode的分数
goal = -110
for i in tqdm.tqdm(range(num_episodes), file=sys.stdout):
    state = env.reset()[0]  # 重置环境
    score = 0
    while True:  # 开始一个episode (每一个循环代表一步)
        env.render()  # 显示图形界面
        # ε-greedy策略采样动作。
        if np.random.random() < epsilon:
            action = np.random.randint(0, 3)
        else:
            action = DQN.predict(torch.unsqueeze(torch.Tensor(state), 0))[0]

        nxt_state, reward, terminated, truncated, info = env.step(action)  # 执行动作，获得env的反馈
        state = nxt_state
        score += reward
        if terminated:
            print('round: {} successfully terminated'.format(i))
            break
    print("Episode {0}/{1}, Score: {2}".format(i, num_episodes, score))
    scores.append(score)
    goals.append(goal)
    episodes.append(i + 1)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
env.close()
print('avg_score: {}'.format(np.mean(scores)))
utils.utils.plt_graph(episodes, scores, avg_scores, goals, 'MountainCar-v0', 'DQN', 'test',save_path='./')
