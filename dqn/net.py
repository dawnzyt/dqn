import numpy as np
import torch
from torch import nn


# deep Q network
# 由简单的全连接层组成
# 建模由state到q值的映射
class DQN(torch.nn.Module):
    def __init__(self, n_state, n_action, n_hidden=24):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action)
        )
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        """

        :param inputs: shape: [batch_size, n_state]
        :return:
        """
        return self.fnn(inputs)

    def predict(self, inputs):
        """
        基于贝尔曼最优方程的greedy策略。根据当前状态，选择最优动作
        :param inputs: shape: [batch_size, n_state]
        :return: action: shape: [batch_size, 1]
        """
        q = self.forward(inputs).detach().cpu().numpy()
        return np.argmax(q, axis=1)