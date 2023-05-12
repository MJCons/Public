from model_softmax import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from memory import BufferU, BufferU_experience
from memory import BufferB, BufferB_experience
from memory import BufferC, BufferC_experience
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
    def __init__(self, n_agents, dim_states, dim_obs, dim_act, batch_size,
                 capacity, episode_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = Critic(n_agents, dim_states, dim_act)  # 中心化的方式，只有一个critic
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.bufferU = BufferU(capacity)
        self.bufferB = [BufferB(capacity) for i in range(n_agents)]  # 每一个agent有一个buffer 用来存放训练需要的经验
        self.bufferC = BufferC(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episode_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = Adam(self.critics.parameters(), lr=0.001)
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            self.critics.cuda()
            self.critics_target.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.actors:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0
        th.save(self.critics, 'critic_1.pkl')
        th.save(self.actors[0], 'zero_actor_1.pkl')
        th.save(self.actors[1], 'one_actor_1.pkl')
        th.save(self.actors[2], 'two_actor_1.pkl')
        th.save(self.actors[3], 'three_actor_1.pkl')
        th.save(self.actors[4], 'four_actor_1.pkl')
        th.save(self.actors[5], 'five_actor_1.pkl')
        th.save(self.actors[6], 'six_actor_1.pkl')
        th.save(self.actors[7], 'seven_actor_1.pkl')
        th.save(self.actors[8], 'eight_actor_1.pkl')
        th.save(self.actors[9], 'nine_actor_1.pkl')

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
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            # whole_state = state_batch.view(self.batch_size, -1)
            whole_state = state_batch[:, agent, :].view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
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
            # print("计算loss_Q的两部分值为：", current_Q, target_Q)
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
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
            ac[:, agent, :] = action_i
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

                    # print("经过softmax处理后为：", act)

                if self.episode_done > self.episodes_before_train and \
                        self.var[i] > 0.05:
                    # self.var[i] *= 0.999998
                    self.var[i] *= 0.9993
                    # act = th.clamp(act, -1.0, 1.0)

                actions[i, :] = act

        self.steps_done += 1

        return actions

    def initialize(self, capacity):
        # 更新所有的buffer 每一轮episode之后 buffer 清空
        # 清空相当于重新进行一次初始化
        # print("缓冲区初始化")
        self.memory.__init__(capacity)
        self.bufferU.__init__(capacity)
        self.bufferC.__init__(capacity)
        for i_agents in range(self.n_agents):
            self.bufferB[i_agents].__init__(capacity)

    def learn(self):
        # print("------------------------------------------------进入learn函数")
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        a_loss = []
        # 首先对每一个agent的actor进行更新，选择K个experiences进行学习
        for agent in range(self.n_agents):
            # print("当前更新网络参数的智能体ID为：", agent)
            transitions = self.bufferB[agent].sample(self.batch_size)
            # print("取出的转换对为:", transitions)
            batch = BufferB_experience(*zip(*transitions))
            # print("batch的内容为;", batch)
            advantage_batch = th.stack(batch.A_a).type(FloatTensor)
            # print("advantage_batch的内容为：", advantage_batch)
            self.actor_optimizer[agent].zero_grad()
            # for name, parms in self.actors[agent].named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            actor_loss = -advantage_batch
            # print("输出actor_loss的第一步值：", actor_loss)
            actor_loss = actor_loss.mean()
            # print("输出actor_loss的平均值：", actor_loss)
            # print("当前的agent的id为：", agent)
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            a_loss.append(actor_loss)
            print("开始检查神经网络的梯度是否更新")
            # for name, parms in self.actors[agent].named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
        # 更新central_critic网络 需要将全部的experiences都取出来 利用均方误差损失函数求loss
        # 在计算MSEloss的时候，没戏都需要下一个时间片的Q值，所以需要保证选出来的转换对是按照顺序的
        sum_experience = self.bufferU.__len__()
        central_transitions_1 = self.bufferU.sample(sum_experience)
        # print("测试一下这样sample出来的转换对是不是按照顺序来的：", central_transitions_1)
        central_transitions_2 = self.bufferU.memory
        # print("测试一下直接把memory全部拿出来的形式：", central_transitions_2)
        central_transitions = central_transitions_2
        central_batch = BufferU_experience(*zip(*central_transitions))
        # print("得到的解压后的batch为：", central_batch)
        q_value_batch = th.stack(central_batch.q_value).type(FloatTensor)
        # print("q_value_batch", q_value_batch)
        q_value_batch = q_value_batch.requires_grad_()
        # print("q_value_batch", q_value_batch)
        sum_reward_batch = th.stack(central_batch.sum_reward).type(FloatTensor)
        for t in range(sum_experience - 1):
            self.critic_optimizer.zero_grad()
            for name, parms in self.critics.named_parameters():
                print('-->name:', name)
                print('-->para:', parms)

                print('-->grad_requirs:', parms.requires_grad)
                print('-->grad_value:', parms.grad)
                print("===")
            target_Q = q_value_batch[t+1] + sum_reward_batch[t+1] * scale_reward
            # print("计算loss_Q的两个部分为：", target_Q.detach(), q_value_batch[t])
            loss_Q = nn.MSELoss()(q_value_batch[t], target_Q.detach())
            # print("loss_Q：", loss_Q)
            loss_Q.backward()
            self.critic_optimizer.step()
            for name, parms in self.critics.named_parameters():
                print('-->name:', name)
                print('-->para:', parms)

                print('-->grad_requirs:', parms.requires_grad)
                print('-->grad_value:', parms.grad)
                print("===")
        c_loss = loss_Q
        a_loss.append(0)
        a_loss.append(1)
        a_loss.append(2)
        a_loss.append(3)
        a_loss.append(4)
        a_loss.append(5)
        a_loss.append(6)
        a_loss.append(7)
        a_loss.append(8)
        a_loss.append(9)

        return c_loss, a_loss

    def learn1(self):
        # print("------------------------------------------------进入learn函数")
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        a_loss = []
        transitions = self.memory.memory
        batch = Experience(*zip(*transitions))
        # print("batch的类型", batch, batch.shape())
        state_batch = th.stack(batch.states).type(FloatTensor)
        print("输出state_batch的类型：", state_batch.shape)
        action_batch = th.stack(batch.actions).type(FloatTensor)
        print("输出action_batch的类型：", action_batch.shape)
        reward_batch = th.stack(batch.rewards).type(FloatTensor)
        next_state_batch = th.stack(batch.next_states).type(FloatTensor)
        # 首先对每一个agent的actor进行更新，选择K个experiences进行学习
        # 试一下统计的方法能不能进行backward
        for agent in range(self.n_agents):
            print("当前更新网络参数的智能体ID为：", agent)
            actions = th.zeros(self.n_agents)
            for i in range(self.n_agents):
                print("输出action_batch[:, :, :]", action_batch[:, :, :])
                actions[i] = th.max(action_batch[:, :, :], 0)[1]
            action_statistics = th.zeros(9)
            for i_d in range(self.n_agents):
                if i_d != agent:
                    actions_id_item = int(actions[i_d].item())
                    action_statistics[actions_id_item] += 1
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            print("新生成的动作为：", action_i)
            action_i_item = int(actions.item())
            action_statistics_total = action_statistics.clone()
            action_statistics_total[action_i_item] += 1
            c_1 = self.critics(state_batch, action_statistics_total)
            c_2 = self.critics(state_batch, action_statistics)
            advantage_batch = c_1-c_2.detach()
            self.actor_optimizer[agent].zero_grad()
            actor_loss = -advantage_batch
            # print("输出actor_loss的第一步值：", actor_loss)
            actor_loss = actor_loss.mean()
            # print("输出actor_loss的平均值：", actor_loss)
            # print("当前的agent的id为：", agent)
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            a_loss.append(actor_loss)
        # 更新central_critic网络 需要将全部的experiences都取出来 利用均方误差损失函数求loss
        # 在计算MSEloss的时候，没戏都需要下一个时间片的Q值，所以需要保证选出来的转换对是按照顺序的
        # 在求critic两个部分的时候，需要分别计算两个部分的动作统计量
        for step in range(39):
            self.critic_optimizer.zero_grad()
            actions_first_statis = th.zeros(9)
            for i_d in range(self.n_agents):
                actions_id_item_1 = int(action_batch[step, i_d].item())
                # print("输出actions_id_item：", actions_id_item)
                actions_first_statis[actions_id_item_1] += 1
            curr_Q = self.critics(state_batch[step], actions_first_statis)
            actions_next_statis = th.zeros(9)
            for i_d_2 in range(self.n_agents):
                actions_id_item_2 = int(action_batch[step, i_d_2].item())
                # print("输出actions_id_item：", actions_id_item)
                actions_next_statis[actions_id_item_2] += 1
            target_Q = self.critics(next_state_batch[step], actions_next_statis)
            target_Q = target_Q + reward_batch[step]
            loss_Q = nn.MSELoss(curr_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer.step()
        c_loss = loss_Q
        return c_loss, a_loss
# 思路3：每次保存状态、动作的统计数据、reward数据
# critic正常按照求出，只需要当前的step和下一步的step即可
# actor第一部分就用转换对的数据，第二部分用统计的动作减去现在求出的动作

    def learn3(self):
        # print("------------------------------------------------进入learn3函数")
        if self.episode_done <= self.episodes_before_train:
            return None, None

        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        c_loss = []
        a_loss = []
        # 每次都取最新的batch_size个经验
        transitions = self.bufferC.memory[-self.batch_size:]
        # print("测试一下取出的是不是最新的n个:")
        # print("首先打印取出的转换对：", transitions[0])
        # print("打印长度：", len(transitions))
        # print("首先打印取出的转换对：", transitions[self.batch_size-1])
        # len = self.bufferC.__len__()
        # print("当前buffer的len为：", self.bufferC.__len__())
        # print("打印倒数第batch_size附近的五个个：", self.bufferC.memory[len-self.batch_size])
        # print(self.bufferC.memory[len - 1])
        batch = BufferC_experience(*zip(*transitions))
        # print("batch的类型", batch, batch.shape())
        state_batch = th.stack(batch.states).type(FloatTensor)
        # print("输出state_batch的类型：", state_batch.shape)
        obs_batch = th.stack(batch.obs).type(FloatTensor)
        # print("输出obs的类型：", obs_batch.shape)
        action_batch = th.stack(batch.action_statistic).type(FloatTensor)
        # print("输出action_batch的类型：", action_batch.shape)
        reward_batch = th.stack(batch.rewards).type(FloatTensor)
        next_state_batch = th.stack(batch.next_states).type(FloatTensor)
        # 首先对每一个agent的actor进行更新，选择K个experiences进行学习
        # 试一下统计的方法能不能进行backward
        for agent in range(self.n_agents):
            # print("当前更新网络参数的智能体ID为：", agent)
            obs_i = obs_batch[:, agent, :]
            action_i = self.actors[agent](obs_i)
            # action_i = F.gumbel_softmax(action_i, 0.1, True)
            # print("新生成的动作为：", action_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            ac = ac.view(self.batch_size, -1)
            # print("state_batch, ac的size：", state_batch.shape, ac.shape)
            c_1 = self.critics(state_batch, ac)
            action_batch_i = action_batch.clone()
            action_batch_i[:, agent, :] = th.zeros(9)
            # print("action_batch_i[:, agent, :]", action_batch_i[:, agent, :])
            action_batch_i = action_batch_i.view(self.batch_size, -1)
            # print("打印action_batch_i.shape", action_batch_i.shape)
            c_2 = self.critics(state_batch, action_batch_i)
            advantage_batch = c_1 - c_2
            self.actor_optimizer[agent].zero_grad()
            # for name, parms in self.actors[agent].named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            actor_loss = (-advantage_batch) * 1000
            # print("输出actor_loss的第一步值：", actor_loss)
            actor_loss = actor_loss.mean()
            # print("输出actor_loss的平均值：", actor_loss)
            # print("当前的agent的id为：", agent)
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            # for name, parms in self.actors[agent].named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            a_loss.append(actor_loss)
        # 更新central_critic网络 需要将全部的experiences都取出来 利用均方误差损失函数求loss
        # 在计算MSEloss的时候，没戏都需要下一个时间片的Q值，所以需要保证选出来的转换对是按照顺序的
        # 在求critic两个部分的时候，需要分别计算两个部分的动作统计量
        self.critic_optimizer.zero_grad()
        # for name, parms in self.critics.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        # print("测试各种shape:")
        # print("state_batch.shape:", state_batch.shape)
        curr_Q_1 = state_batch[0:self.batch_size - 1, :]
        # print("state_batch[step, :].shape", state_batch[0:self.batch_size - 1].shape)
        # print("curr_Q_1.shape", curr_Q_1.shape)
        curr_Q_2 = action_batch[0:self.batch_size - 1, :]
        curr_Q_2 = curr_Q_2.view(self.batch_size-1, -1)
        # print("计算curr_Q的两个部分的形状：", curr_Q_1.shape, curr_Q_2.shape)
        curr_Q = self.critics(curr_Q_1, curr_Q_2)
        # print("curr_Q.shape:", curr_Q.shape)
        target_Q_1 = state_batch[1:self.batch_size, :]
        target_Q_2 = action_batch[1:self.batch_size, :]
        target_Q_2 = target_Q_2.view(self.batch_size-1, -1)
        # print("计算target_Q的两个部分的形状为：", target_Q_1.shape, target_Q_2.shape)
        target_Q = self.critics(target_Q_1, target_Q_2)
        # print("target_Q为：", target_Q)
        # print("reward_batch[step]为:", reward_batch[step])
        # print("target_Q.shape:", target_Q.shape)
        target_Q = target_Q * self.GAMMA + reward_batch[0:self.batch_size - 1] * 0.001
        # print("reward_batch[0:self.batch_size - 1].shape:", reward_batch[0:self.batch_size - 1].shape)
        # print("输出计算loss_Q的两部分的值：", curr_Q, target_Q)
        loss_Q = nn.MSELoss()(curr_Q, target_Q.detach())
        # loss_Q = loss_Q + nn.MSELoss()(curr_Q, target_Q.detach())
        # loss_Q = loss_Q.mean()
        loss_Q.backward()
        self.critic_optimizer.step()
        # for name, parms in self.critics.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        # print("输出返回的loss:", loss_Q * 10000)
        c_loss.append(loss_Q * 100)

        return c_loss, a_loss

