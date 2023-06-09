import os
import numpy as np
import torch as th
from copy import copy
env_size = 9  #地图的大小 9*9
num_poi = 16
num_fall = 5
num_agent = 10
p_obstacle = 10
up_energy = 5  #每次补充的能量
up_p = 100    #每次补充能量以后 收益的减少值
b_energy = 1
step_reward = 3
man_charge = 6
chong_this = 1 * man_charge

# 将能量的数量级降了一下，如果需要原来的代码，见experiment：env8_l
# 需要构建一个有10个UAV的环境，要把访问的频率和空间中POI的位置都处理


class env:
    def __init__(self):
        self.state_fall = np.array([(2, 2), (2, 7), (4, 4), (6, 2), (6, 6)], dtype=int)
        self.state_poi = np.array([(1, 1, 1), (1, 6, 1), (2, 4, 1), (2, 6, 1),
                                   (3, 3, 1), (3, 5, 1), (4, 1, 1), (4, 7, 1),
                                   (5, 2, 1), (5, 4, 1), (5, 7, 1), (6, 5, 1),
                                   (7, 1, 1), (7, 3, 1), (7, 7, 1), (8, 5, 1)], dtype=int)
        self.state_uav = np.array([(4, 4, 6), (4, 4, 6), (4, 4, 6), (4, 4, 6), (4, 4, 6),
                                   (4, 4, 6), (4, 4, 6), (4, 4, 6), (4, 4, 6), (4, 4, 6)], dtype=float)
        # self.state_re = np.array([(0, 0, 3.0), (0, 1, 4.0), (0, 2, 4.9), (0, 3, 4.2), (0, 4, 4.0), (0, 5, 3.5),
        #                           (1, 0, 4.0), (1, 1, 5.0), (1, 2, 6.2), (1, 3, 8.0), (1, 4, 6.0), (1, 5, 4.0),
        #                           (2, 0, 5.0), (2, 1, 6.0), (2, 2, 3.2), (2, 3, 5.0), (2, 4, 8.2), (2, 5, 4.5),
        #                           (3, 0, 5.0), (3, 1, 8.0), (3, 2, 15.2), (3, 3, 3.9), (3, 4, 7.5), (3, 5, 4.8),
        #                           (4, 0, 4.0), (4, 1, 7.0), (4, 2, 5.6), (4, 3, 3.0), (4, 4, 4.2), (4, 5, 3.8),
        #                           (5, 0, 3.0), (5, 1, 3.8), (5, 2, 4.0), (5, 3, 4.3), (5, 4, 4.2), (5, 5, 3.0)], dtype=float)
        # state_uav = [[1, 6, 100], [4, 4, 100], [8, 2, 100]]
        # self.state_poi_times = np.array([5, 5, 5, 20, 20, 8, 8, 5, 5, 8], dtype=float)
        self.state_poi_need_times = np.array([5, 5, 10, 10, 8, 8, 4, 5, 5, 20, 4, 10, 8, 8, 8, 2], dtype=float)
        self.sum_poi_times = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    def jfairness(self):
        poi_fairness = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        sum_poi_fairness1 = 0.0
        sum_poi_fairness2 = 0.0
        sum_poi_fairness = 0.0
        for i in range(num_poi):
            poi_fairness[i] = self.sum_poi_times[i] / self.state_poi_need_times[i]
            sum_poi_fairness1 = sum_poi_fairness1 + poi_fairness[i]  # 求和
            sum_poi_fairness2 = sum_poi_fairness2 + (poi_fairness[i] * poi_fairness[i]) #平方和
        # print(poi_fairness)
        sum_poi_fairness = sum_poi_fairness1 * sum_poi_fairness1
        if sum_poi_fairness2 != 0:
            sum_poi_fairness = sum_poi_fairness / (num_poi * sum_poi_fairness2)
        else:
            sum_poi_fairness = 0.0
        # print(sum_poi_fairness)
        return sum_poi_fairness

    def get_obs(self, agent_id):
        # 只能观察到周围9个格子的情况
        obs_get = np.zeros(9)
        x = self.state_uav[agent_id][0]
        y = self.state_uav[agent_id][1]
        i = 0
        a, b = self.exit_poi(x-1, y-1)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x-1, y-1):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x-1, y-1):
            if self.exit_uav(x - 1, y - 1):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif x-1 < 0 or y - 1 < 0:
            obs_get[i] = -1
        elif self.exit_uav(x - 1, y - 1):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1 # 1
        a, b = self.exit_poi(x - 1, y)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x - 1, y):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x - 1, y):
            if self.exit_uav(x - 1, y):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif x - 1 < 0:
            obs_get[i] = -1
        elif self.exit_uav(x - 1, y):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1 # 2
        a, b = self.exit_poi(x - 1, y + 1)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x - 1, y + 1):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x - 1, y + 1):
            if self.exit_uav(x - 1, y + 1):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif x - 1 < 0 or y + 1 >= env_size:
            obs_get[i] = -1
        elif self.exit_uav(x - 1, y + 1):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1 #3
        a, b = self.exit_poi(x, y - 1)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x, y - 1):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x, y - 1):
            if self.exit_uav(x, y - 1):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif y - 1 < 0:
            obs_get[i] = -1
        elif self.exit_uav(x, y - 1):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1  # 4
        a, b = self.exit_poi(x, y)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x, y):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x, y):
            if self.exit_uav(x, y):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif self.exit_uav(x, y):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1#5
        a, b = self.exit_poi(x, y + 1)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x, y + 1):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x, y + 1):
            if self.exit_uav(x, y + 1):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif y + 1 >= env_size:
            obs_get[i] = -1
        elif self.exit_uav(x, y + 1):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1#6
        a, b = self.exit_poi(x + 1, y - 1)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x + 1, y - 1):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x + 1, y - 1):
            if self.exit_uav(x + 1, y - 1):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif y + 1 >= env_size or x + 1 >= env_size:
            obs_get[i] = -1
        elif self.exit_uav(x + 1, y - 1):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1 # 7
        a, b = self.exit_poi(x + 1, y)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x + 1, y):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x + 1, y):
            if self.exit_uav(x + 1, y):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif x + 1 >= env_size:
            obs_get[i] = -1
        elif self.exit_uav(x + 1, y):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        i = i + 1 # 8
        # print(x+1,end='')
        # print(y+1)
        a, b = self.exit_poi(x + 1, y + 1)
        if a and self.state_poi[b][2] == 1:
            if self.exit_uav(x + 1, y + 1):
                obs_get[i] = 5
            else:
                obs_get[i] = 1
        elif exit_fall(self, x + 1, y + 1):
            if self.exit_uav(x + 1, y + 1):
                obs_get[i] = 6
            else:
                obs_get[i] = 2
        elif x + 1 >= env_size or y + 1 >= env_size:
            obs_get[i] = -1
        elif self.exit_uav(x + 1, y + 1):
            obs_get[i] = 4
        else:
            obs_get[i] = 3
        obs_self = self.state_uav[agent_id]
        obs_get = np.hstack((obs_self, obs_get))
        # print("当前的智能体编号为：", agent_id, end='')
        # print("观测值为：", obs_get)
        return obs_get

    def get_states(self):
        for agent_i in range(num_agent):
            if agent_i == 0:
                states_uav = self.state_uav[agent_i]
            else:
                states_uav_i = self.state_uav[agent_i]
                states_uav = np.hstack((states_uav, states_uav_i))
        # print("得到UAV自身的信息tensor为：", states_uav)
        # print("对应的形状为：", states_uav.shape)
        states_grid = np.zeros(81)
        grid_id = 0
        # 用双层循环来探索各个格子的状态
        for grid_x in range(env_size):
            for grid_y in range(env_size):
                a, b = self.exit_poi(grid_x, grid_y)
                if a and self.state_poi[b][2] == 1:
                    if self.exit_uav(grid_x, grid_y):
                        states_grid[grid_id] = 5
                    else:
                        states_grid[grid_id] = 1
                elif exit_fall(self, grid_x, grid_y):
                    if self.exit_uav(grid_x, grid_y):
                        states_grid[grid_id] = 6
                    else:
                        states_grid[grid_id] = 2
                elif grid_x == 0 or grid_y == 0 or grid_x == env_size - 1 \
                        or grid_y == env_size - 1:
                    states_grid[grid_id] = -1
                elif self.exit_uav(grid_x, grid_y):
                    states_grid[grid_id] = 4
                else:
                    states_grid[grid_id] = 3
                grid_id = grid_id + 1
        states = np.hstack((states_uav, states_grid))
        # print("输出得到的最终状态：", states)
        return states

    def reward_ac(self, i):
        # i 是agent的id
        times = 0
        for fall_i in range(num_fall):
            if self.state_fall[fall_i][0] == self.state_uav[i][0] and self.state_fall[fall_i][1] == \
                    self.state_uav[i][1]:
                # print("充电一次，补充的能量为：", 100 - self.state_uav[i][2])
                # reward_ac = reward_ac + (120 - self.state_uav[i][2]) * 0.5 * ((120 - self.state_uav[i][2]) / 120)
                # self.state_uav[i][2] = man_charge
                self.state_uav[i][2] = min(man_charge, self.state_uav[i][2] + chong_this)
        for poi_i in range(num_poi):
            if self.state_poi[poi_i][0] == self.state_uav[i][0] and \
                    self.state_poi[poi_i][1] == self.state_uav[i][1] and \
                    self.state_poi[poi_i][2] == 1:
                # print("收集数据一次")
                times = times + 1
                self.state_poi[poi_i][2] = 0
                self.sum_poi_times[poi_i] += 1
        return times

    def update_state(self, action, n_agents):
        actions1 = action.clone()
        actions = th.zeros(n_agents, 1)
        for i in range(n_agents):
            actions[i][0] = th.max(actions1[i], 0)[1]
        # print("与环境交互的actions为：", actions)
        reward = np.zeros((n_agents, 1))
        sum_times = 0
        # print("定义时收益矩阵的类型为：", reward.shape)
        for i in range(n_agents):
            if actions[i][0] == 0:  # 左上
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] >= 1 and self.state_uav[i][1] >= 1:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] -= 1
                    self.state_uav[i][1] -= 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle  # 做出的action碰壁会受到的惩罚
                    self.state_uav[i][2] -= 0.5
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 1: #正上方
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] >= 1:
                    # 有充足的电量且可以做向上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] -= 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5# 做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 2: # 右上
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] >= 1 and self.state_uav[i][1] <= env_size-2:
                    # 有充足的电量且可以做右上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] -= 1
                    self.state_uav[i][1] += 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5# 做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 3: # 左
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][1] >= 1:
                    # 有充足的电量且可以做左的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][1] -= 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5# 做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 4: # 不移动位置
                reward[i] = reward[i] - step_reward
                self.state_uav[i][2] -= 0.5
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 5: # 向右
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][1] <= env_size-2:
                    # 有充足的电量且可以做右的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][1] += 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5# 做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 6: #向左下
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] <= env_size-2 and self.state_uav[i][1] >= 1:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] += 1
                    self.state_uav[i][1] -= 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5#做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 7: #向下
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] <= env_size - 2:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] += 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5#做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2


            if actions[i][0] == 8: #向右下
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] <= env_size - 2 and self.state_uav[i][1] <= env_size - 2:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] += 1
                    self.state_uav[i][1] += 1
                    reward[i] = reward[i] - step_reward
                else:
                    reward[i] -= p_obstacle
                    self.state_uav[i][2] -= 0.5#做出的action碰壁会受到的惩罚
                a, b = self.reward_ac(i)
                reward[i] = reward[i] + a
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    self.state_uav[i][2] += up_energy
                    reward[i] = reward[i] - up_p
                num = self.loca_poi_number(self.state_uav[i][0], self.state_uav[i][1])
                # reward[i] = reward[i] + num * 2

            if self.state_uav[i][2] > man_charge:
                self.state_uav[i][2] = man_charge
            #避免充电充的超过了可以承受的范围
        curr_state = self
        # 返回的是所有agent执行完动作以后，整体环境的状态变化，各个智能体的观测值还要自己get到
        # reward 设置成为一个数组 对应每个agent的收益
        # print("计算本次的收益为：", reward)
        # print("收益reward矩阵定义的类型为：", reward.shape)
        return curr_state, reward, sum_times

    def central_update_state(self, action, n_agents):
        # reward应该是一个总值：总的收集次数*50*收集后的公平性-总的惩罚值（惩罚值包括碰壁、耗尽电量）
        total_reward = th.zeros(1)
        actions1 = action.clone()
        actions = th.zeros(n_agents, 1)
        for i in range(n_agents):
            actions[i][0] = th.max(actions1[i], 0)[1]
        # print("与环境交互的actions为：", actions)
        sum_times = 0
        # 为了统计计算的reward是否正确，检查惩罚和奖励的智能体的数量
        total_p_o = 0
        total_p_b = 0
        for i in range(n_agents):
            if actions[i][0] == 0:  # 左上
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] >= 1 and self.state_uav[i][1] >= 1:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] -= 1
                    self.state_uav[i][1] -= 1
                else:
                    total_reward -= p_obstacle  # 做出的action碰壁会受到的惩罚
                    self.state_uav[i][2] -= 0.5
                    total_p_o += 1
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 1:  # 正上方
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] >= 1:
                    # 有充足的电量且可以做向上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] -= 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 2:  # 右上
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] >= 1 and self.state_uav[i][1] <= env_size - 2:
                    # 有充足的电量且可以做右上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] -= 1
                    self.state_uav[i][1] += 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 3:  # 左
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][1] >= 1:
                    # 有充足的电量且可以做左的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][1] -= 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 4:  # 不移动位置
                self.state_uav[i][2] -= 0.5
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 5:  # 向右
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][1] <= env_size - 2:
                    # 有充足的电量且可以做右的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][1] += 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 6:  # 向左下
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] <= env_size - 2 and self.state_uav[i][1] >= 1:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] += 1
                    self.state_uav[i][1] -= 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 7:  # 向下
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] <= env_size - 2:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] += 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p

            if actions[i][0] == 8:  # 向右下
                if self.state_uav[i][2] >= 1 and \
                        self.state_uav[i][0] <= env_size - 2 and self.state_uav[i][1] <= env_size - 2:
                    # 有充足的电量且可以做左上的动作
                    self.state_uav[i][2] -= 1
                    self.state_uav[i][0] += 1
                    self.state_uav[i][1] += 1
                else:
                    total_reward -= p_obstacle
                    total_p_o += 1
                    self.state_uav[i][2] -= 0.5  # 做出的action碰壁会受到的惩罚
                b = self.reward_ac(i)
                sum_times = sum_times + b
                if self.state_uav[i][2] <= b_energy:
                    total_p_b += 1
                    self.state_uav[i][2] += up_energy
                    total_reward = total_reward - up_p
            if self.state_uav[i][2] > man_charge:
                self.state_uav[i][2] = man_charge
            # 避免充电充的超过了可以承受的范围
        # print("本step的总收集次数sum_times:", sum_times, end='')
        # print("总碰壁次数sum_o:", total_p_o, end='')
        # print("总补充能量的次数sum_b:", total_p_b, end='')
        total_reward[0] = total_reward[0] + 50 * sum_times * self.jfairness()
        # print("本次的reward为：", total_reward)
        curr_state = self
        # 返回的是所有agent执行完动作以后，整体环境的状态变化，各个智能体的观测值还要自己get到
        # reward 设置成为一个数组 对应每个agent的收益
        # print("计算本次的收益为：", reward)
        # print("收益reward矩阵定义的类型为：", reward.shape)
        return curr_state, total_reward, sum_times

    def step_update(self, t):

        if t % 2 == 0:
            self.state_poi[9][2] = 1
        if t % 4 == 0:
            self.state_poi[2][2] = 1
            self.state_poi[3][2] = 1
            self.state_poi[11][2] = 1
        if t % 5 == 0:
            self.state_poi[4][2] = 1
            self.state_poi[5][2] = 1
            self.state_poi[12][2] = 1
            self.state_poi[13][2] = 1
            self.state_poi[14][2] = 1
        if t % 8 == 0:
            self.state_poi[0][2] = 1
            self.state_poi[1][2] = 1
            self.state_poi[7][2] = 1
            self.state_poi[8][2] = 1
        if t % 10 == 0:
            self.state_poi[6][2] = 1
            self.state_poi[10][2] = 1
        if t % 20 == 0:
            self.state_poi[15][2] = 1

    def exit_poi(self, x, y):
        flag = False
        poi_id = -1
        for i in range(num_poi):
            if self.state_poi[i][0] == x and self.state_poi[i][1] == y:
                flag = True
                poi_id = i
        return flag, poi_id

    def loca_poi_number(self, x, y):
        obs_get = np.zeros(9)
        i = 0
        a, b = self.exit_poi(x - 1, y - 1)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x - 1, y)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x - 1, y + 1)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x, y - 1)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x, y)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x, y + 1)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x + 1, y - 1)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x + 1, y)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        i = i + 1
        a, b = self.exit_poi(x + 1, y + 1)
        if a:
            if self.state_poi[b][2] == 1:
                obs_get[i] = 1
        sum = 0
        for count in range(9):
            if obs_get[count] == 1:
                sum = sum + 1
        return sum

    def exit_uav(self, x, y):
        for i in range(num_agent):
            if self.state_uav[i][0] == x and self.state_uav[i][1] == y:
                return True
        return False


def exit_fall(self, x, y):
    flag = False
    for i in range(num_fall):
        if self.state_fall[i][0] == x and self.state_fall[i][1] == y:
            flag = True
    return flag




