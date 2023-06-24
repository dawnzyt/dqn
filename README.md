# Introduction

A simple implementing of deep q-network (DQN) with pytorch.

DDQN is also implemented. You can see the control flag `DDQN` in the code. The dueling dqn is not implemented yet.

the original paper link: [《Playing Atari with Deep Reinforcement Learning》](https://arxiv.org/abs/1312.5602)
and [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236).

And the environment is [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/), with the continuous state space of
2 and discrete action space of 3.

# Structure

```
├── README.md
├── train_dqn.py: train the dqn model
├── run_dqn.py: run the dqn model
├── dqn
│   ├── net.py: the dqn network
│   ├── agent.py:  the dqn agent, including the replay buffer, the epsilon greedy policy and the training process...
├── utils: some useful functions
├── result: the result of the experiment
...
```

# run

```
python train_dqn.py
python run_dqn.py
```

You can directly change relative parameters in the code.

Also, `cuda` is supported, just change the `device` in the code.

For more details, please see the code.

# result

Available parameters:

| Parameter     | Description                                             |
|---------------|---------------------------------------------------------|
| n_state       | Number of states in the environment                     |
| n_action      | Number of possible actions in the environment           |
| n_hidden      | Dimensionality of the hidden layer in the DQN           |
| gamma         | Discount factor for updating Q-values                   |
| lr            | Learning rate for the optimizer                         |
| batch_size    | Number of samples to include in each training batch     |
| buf_size      | Maximum size of the experience buffer                   |
| sync_freq     | Number of steps between updates of the target network   |
| epsilon_min   | Minimum value of epsilon for epsilon-greedy exploration |
| epsilon_decay | Decay rate for epsilon in epsilon-greedy exploration    |
| DDQN          | Flag indicating whether to use Double DQN               |
| exp_name      | Name of the experiment for tracking results             |

the baseline is set as:

|     parm      |  value  | 
|:-------------:|:-------:| 
|    n_state    |    2    |
|   n_action    |    3    |
|   n_hidden    |   256   |
|     gamma     |  0.99   |
|      lr       |  0.001  |
|  batch_size   |   64    |
|   buf_size    | 1000000 |
|   sync_freq   |   100   |
|  epsilon_min  |  0.01   |
| epsilon_decay |  0.999  |
|     DDQN      |  False  |

Training result:

![MountainCar](./result/dqn/exp1/MountainCar-v0_DQN_train.png 'Train')

Testing result:

![MountainCar](./result/dqn/exp1/MountainCar-v0_DQN_test.png 'Test')