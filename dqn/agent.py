import copy
import math
import torch
import torch.nn as nn
import numpy as np
import utils.dataloger
from dqn.net import DQN


# DQN agent, 由DQN网络和经验回放等组成, 用于训练DQN网络,
class DQNAgent:
    def __init__(self, env, gamma=0.9, lr=0.001, batch_size=128, buf_size=1000000, sync_freq=100,
                 epsilon_min=0.01, epsilon_decay=0.999, DDQN=False, exp_name='exp1', device='cpu'):
        # 初始化agent
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.n
        self.model = DQN(self.n_state, self.n_action).to(device)
        self.device = device
        self.target_model = copy.deepcopy(self.model)
        self.DDQN = DDQN
        # 初始化训练超参
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buf_size = buf_size
        self.loger = utils.dataloger.DataLoger(dir='./result/dqn/{}'.format(exp_name))  # 记录loss

        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # buffer size: 1000000, state action reward next_state
        self.buffer = np.zeros(shape=(self.buf_size, self.n_state * 2 + 3))
        self.bf_counter = 0  # buffer中已存储的经验
        self.learn_counter = 0  # 记录学习次数
        self.sync_freq = sync_freq

        # loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def sample_act(self, state):
        """
        前面buf_size个经验随机采样，后面基于greedy策略采样

        :param state: (2,)
        :return: act: (1,)
        """
        if np.random.rand() < self.epsilon:
            act = np.random.randint(self.n_action)
        else:
            act = self.model.predict(torch.unsqueeze(torch.Tensor(state).to(self.device), 0))[0]
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return act

    def sample_batch(self):
        """
        从buffer中随机抽取batch_size个经验
        :return:
        """
        max_buffer = min(self.bf_counter, self.buf_size)  # 当前buffer内已存储的经验
        # 随机抽取batch_size个经验
        idx = np.random.choice(max_buffer, self.batch_size, replace=False)
        batch = self.buffer[idx, :]
        state = batch[:, :self.n_state]
        action = batch[:, self.n_state:self.n_state + 1]
        reward = batch[:, self.n_state + 1:self.n_state + 2]
        nxt_state = batch[:, self.n_state + 2:self.n_state * 2 + 2]
        done = batch[:, self.n_state * 2 + 2:]
        return state, action, reward, nxt_state, done

    def learn(self, epoch, step):
        """
        训练网络, 批量训练
        训练方法: 用r + γ * max(Q(s', a'))逼近Q(s, a)。
        其中Q(s, a)是网络的输出，Q(s', a')是目标网络的输出。
        state: batch_size * n_state
        action: batch_size * 1
        reward: batch_size * 1
        nxt_state: batch_size * n_state

        :return:
        """

        max_buffer = min(self.bf_counter, self.buf_size)  # 当前buffer内已存储的历史经验回放数据
        if max_buffer < self.batch_size:
            return
        self.learn_counter += 1  # 记录学习次数
        if self.learn_counter % self.sync_freq == 0:
            self.sync_target_model()  # 每隔sync_freq次同步一次target_model

        state, action, reward, nxt_state, done = self.sample_batch()
        # 将numpy转换为tensor
        state = torch.Tensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        nxt_state = torch.Tensor(nxt_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        # 计算loss, 即Q(s, a)与r + γ * max(Q(s', a'))的均方差
        q = self.model(state).gather(dim=1, index=action)
        q_next = self.target_model(nxt_state).detach().max(dim=1)[0].unsqueeze(dim=1)
        # 对于target, 已经done的不用+gamma * max(Q(s', a'))
        done_idx = done == 1
        # double DQN: 用当前网络选择动作, 用目标网络计算Q(s', a')
        q_target = reward + self.gamma * (
            q_next if not self.DDQN else self.target_model(nxt_state).detach().gather(dim=1, index=
            self.model(nxt_state).detach().max(dim=1)[1].unsqueeze(dim=1)))
        q_target[done_idx] = reward[done_idx]
        loss = self.loss_fn(q, q_target)
        if step % 100 == 0:
            self.loger.log('mse_loss', loss.item(), step)
            if step % 1000 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, step, loss.item()))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def save_transition(self, state, action, reward, nxt_state, done):
        """
        经验重放, 保存经验
        """
        idx = self.bf_counter % self.buf_size
        self.buffer[idx, :] = np.hstack((state, action, reward, nxt_state, done))
        self.bf_counter += 1

    def sync_target_model(self):
        """
        同步target_model, 目的是减少target目标值的抖动。
        :return:
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_q(self, state, action):
        """
        获取Q(s,a)
        :param state: (2,)
        :param action: (1,)
        :return:
        """
        return self.model(torch.unsqueeze(torch.Tensor(state).to(self.device), 0))[0].detach().cpu().numpy()[action]
