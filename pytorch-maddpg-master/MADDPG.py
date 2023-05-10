from model_softmax import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward
import torch.nn.functional as F

# th.cuda.set_device(1)


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0
        th.save(self.critics[0], 'zero_critic_1.pkl')
        th.save(self.critics[1], 'one_critic_1.pkl')
        th.save(self.critics[2], 'two_critic_1.pkl')
        th.save(self.actors_target[0], 'zero_actor_1.pkl')
        th.save(self.actors_target[1], 'one_actor_1.pkl')
        th.save(self.actors_target[2], 'two_actor_1.pkl')

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            # print("当前更新网络参数的智能体ID为：", agent)
            transitions = self.memory.sample(self.batch_size)
            # print("取出的转换对为：", transitions)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            # print("输出action_batch", action_batch)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            # whole_state = state_batch.view(self.batch_size, -1)
            # print("测试一下state_batch的view前和view后尺寸是不是一样的")
            # print("前：", state_batch[:, agent, :].shape)
            whole_state = state_batch[:, agent, :].view(self.batch_size, -1)
            # print("后：", whole_state.shape)
            # print("测试一下action_batch的view前和view后尺寸是不是一样的")
            # print("前：", action_batch[:, agent, :].shape)
            whole_action = action_batch.view(self.batch_size, -1)
            # print("后：", whole_action.shape)
            self.critic_optimizer[agent].zero_grad()
            # print("计算Q值的两个项目为：", whole_state, whole_action)
            current_Q = self.critics[agent](whole_state, whole_action)
            # print("输出Q值：", current_Q)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)
            # print("计算target-Q第一步的两个部分为：",non_final_next_states[:, agent, :].view(-1, self.n_states))
            # print(non_final_next_actions.view(-1,
            #                                 self.n_agents * self.n_actions))
            # for it in range(self.batch_size):
            #     if non_final_next_states[:, agent, :].view(-1, self.n_states)[it] == N
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states[:, agent, :].view(-1, self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions
            # print("通过target网络计算t_Q的一部分为：", target_Q)
            # print("计算当前智能体t-q的reward为：", reward_batch[:, agent])
            # print("计算targetQ的两个部分为：", target_Q)
            # print("reward:", reward_batch[:, agent])
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent] * scale_reward)
            # print("计算得到的target_Q的值为：", target_Q)
            # print("当前的智能体id为：", agent, end='')
            print("计算loss_Q的两部分值为：", current_Q, target_Q)
            # print("计算loss_Q的两部分值的shape为：", current_Q.shape, target_Q.shape)
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            print("得到的loss_q的shape为：", loss_Q.shape)
            # print("loss_Q的值为：", loss_Q)
            # print("计算得到loss——Q的值为：", loss_Q)
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            # print("生成的action_i为：", action_i)
            ac = action_batch.clone()
            # print("原来的action为：", ac)
            action_i = F.gumbel_softmax(action_i, 0.1, True)
            # print("action_i为：", action_i)
            ac[:, agent, :] = action_i
            # print("原来的action2为：", ac)
            # print("加入action_i后为：", ac)
            whole_action = ac.view(self.batch_size, -1)
            # print("计算当前智能体的actor的loss的两个量为：1.whole_state", whole_state)
            # print("2.whole_action:", whole_action)
            # print("当前的智能体编号为:", agent)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            # print("actor_1的loss为：", actor_loss)
            # for it in range(self.batch_size):
            #     if actor_loss[it].cpu() == th.tensor([float('nan')]):
            #         print("1111111111111111111111111111111111111111")
            # print("计算得到actor_loss为：", actor_loss)
            actor_loss = actor_loss.mean()
            # print("actor的loss为：", actor_loss)
            # print("backward开始")
            actor_loss.backward()
            # print("backward结束")
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 49 == 0 and self.steps_done > 0:
            # print("进行参数的软更新")
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        with th.no_grad():
            actions = th.zeros(
                self.n_agents,
                self.n_actions)
            FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
            for i in range(self.n_agents):
                sb = state_batch[i, :].detach()
                act1 = self.actors[i](sb.unsqueeze(0)).squeeze()
                act_i = th.max(act1, 0)[1]
                act = th.zeros(1, 9)
                act[0][act_i] = 1

                a_r = np.random.rand(1)
                # print("用于判断是否使用噪声动作的随机数为：", a_r)
                # print("当前的var值为：", self.var[i])
                if a_r <= self.var[i]:
                    ra = th.from_numpy(
                        np.random.randn(1, 9)).type(FloatTensor)
                    # print("生成的矩阵为：", ra)
                    act = F.gumbel_softmax(ra, 1, True)
                    # print("经过gumbel_softmax的act:", act)
                    # print("经过softmax处理后为：", act)


                if self.episode_done > self.episodes_before_train and \
                        self.var[i] > 0.05:
                    # self.var[i] *= 0.999998
                    self.var[i] *= 0.9993
                # act = th.clamp(act, -1.0, 1.0)

                actions[i, :] = act

        self.steps_done += 1

        return actions
