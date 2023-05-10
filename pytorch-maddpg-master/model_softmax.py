import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024+act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        # print("输出result和acts的shape：", result.shape, acts.shape)
        combined = th.cat([result, acts], 1)
        # print("combined的shape:", combined.shape)
        result = F.relu(self.FC2(combined))
        result = self.FC3(result)
        result = F.relu(result)
        # print("输出result1", result)
        result = self.FC4(result)
        # print("result2:", result)
        return result
        # return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = th.tanh(self.FC3(result))
        result = F.softmax(result, dim=1)
        # print("本次actor网络的计算结果为：", result)
        return result
