import time
from MADDPG import MADDPG
import numpy as np
import torch as th
from env import env
import matplotlib.pyplot as plt
from params import scale_reward


# do not render the scene
# np.random.seed(1234)
# th.manual_seed(1234)

n_agents = 10
n_observations = 12
n_actions = 9
capacity = 520
batch_size = 520
n_states = n_agents * 3 + 81
# 假设状态信息包括每个UAV自身的信息（位置的x y坐标、剩余的电量、整个格子区间的81中状态）

n_episode = 30000
max_steps = 40
episodes_before_train = 15


maddpg = MADDPG(n_agents, n_states, n_observations, n_actions, batch_size, capacity, episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

'''
画图像需要包括：
各个agent的actor图像 
central critic loss图像
'''

def __main__():
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
    t1 = time.ctime()
    print("主程序运行开始：", t1)
    start = time.perf_counter()
    # 各个agent的actor的loss的图像
    list_c_loss = np.zeros(1000000)
    list_zero = np.zeros(1000000)
    list_one = np.zeros(1000000)
    list_two = np.zeros(1000000)
    list_three = np.zeros(1000000)
    list_four = np.zeros(1000000)
    list_five = np.zeros(1000000)
    list_six = np.zeros(1000000)
    list_seven = np.zeros(1000000)
    list_eight = np.zeros(1000000)
    list_nine = np.zeros(1000000)
    count = 0
    # 只需要一个总体的reward因为在中心化的critic下无法得到每一个agent的reward
    # lei_episode_one = np.zeros(30000)
    # lei_episode_two = np.zeros(30000)
    # lei_episode_three = np.zeros(30000)
    sum_lei = np.zeros(1000000)
    sum_times = np.zeros(1000000)
    count_lie_re = 0
    max_times = 0

    for i_episode in range(n_episode):
        sum_reward = 0
        sum_time = 0
        # print('当前的episode为:   ', maddpg.episode_done)
        curr_state = env()  # 生成一个新的环境
        '''
        有一个总的buffer U 用来帮助critic训练
        每一个actor都有一个自己的buffer用来存放自己训练的数据
        '''
        # 第一步初始化两个buffer
        for t in range(max_steps):
            # print("当前step为：", t)
            if t % 1 == 0 and t > 0:
                curr_state.step_update(t)

            '''
            第二步：获得每个智能体的动作
            '''

            for i in range(n_agents):
                if i == 0:
                    curr_obs = curr_state.get_obs(i)
                else:
                    curr_obs_i = curr_state.get_obs(i)
                    curr_obs = np.row_stack((curr_obs, curr_obs_i))
            curr_obs = th.from_numpy(curr_obs).float()
            curr_obs = curr_obs.type(FloatTensor)
            curr_actions = maddpg.select_action(curr_obs)
            # print("生成的动作为：", curr_actions)
            # print("对应的tensor.shape为：", curr_actions.shape)
            # 得到计算Q值所需要的s_t的状态信息，假设状态信息包括每个UAV自身的信息（位置的x y坐标、剩余的电量、整个格子区间的81中状态）
            s_t = curr_state.get_states()
            # print("s_t的类型", s_t)
            s_t = th.from_numpy(s_t)
            # print("s_t的类型", s_t)
            '''
            第三步： 根据联合动作 计算出这个时间片的联合reward
            '''
            first_state = curr_state
            for agent_f in range(n_agents):
                if agent_f == 0:
                    first_state_j = first_state.get_obs(agent_f)
                else:
                    first_state_jj = first_state.get_obs(agent_f)
                    first_state_j = np.row_stack((first_state_j, first_state_jj))
            first_state_j = th.from_numpy(first_state_j).float()
            next_state, curr_reward, times = curr_state.central_update_state(curr_actions, n_agents)
            # print("环境变化得到的结果为：", curr_reward, times)
            sum_reward = sum_reward + curr_reward
            # print("返回本时间片的total_reward为：", curr_reward)
            # print("返回本时间片的sum_times为：", times)
            sum_time = sum_time + times
            curr_state = next_state
            s_t_2 = curr_state.get_states()
            s_t_2 = th.from_numpy(s_t_2)
            # actions_statistic = statistics(curr_actions)
            # print("整理出的统计动作数据为：", actions_statistic)
            # print("对应的shape为:", actions_statistic.shape)
            maddpg.bufferC.push(s_t, first_state_j, s_t_2, curr_actions, curr_reward)
            # maddpg.memory.push(s_t, curr_actions, s_t_2, curr_reward)
            '''
            计算critic网络的Q值：Q(s_t, a_t)
            s_t 采用从状态中获得的方式，a_t采用统计的方式，这样可以解决在计算A的时候的输入纬度不同的问题
            '''

            # a_t = statistics(curr_actions)
            # # 计算得到Q值
            # s_t = th.from_numpy(s_t).float()
            # s_t = s_t.cuda()
            # a_t = a_t.cuda()
            # # print("打印输入到critic网络的两个部分的shape", s_t, a_t)
            # with th.no_grad():
            #     q_value = maddpg.critics(s_t, a_t)

            '''
            第四步：存（Q,R）在buffer U 中
            '''
            # print("存放到bufferU中的数据为：", q_value, curr_reward)
            # maddpg.bufferU.push(q_value, curr_reward)

            '''
            第五步： 对每一个agent根据信誉度机制，求在本时间片下的A, 并将action和A_a存放在bufferB中
            '''
            # for agent_i in range(n_agents):
            #     # 需要求有动作a和没有动作a时的q值差
            #     a_t_i = statistics_i(agent_i, curr_actions)
            #     a_t_i = a_t_i.cuda()
            #     # if a_t_i.equal(a_t):
            #     #     print("当智能体的编号是", agent_i, end='')
            #     #     print("的时候，a_t_i与a_t相等")
            #     # print("打印输入到critic网络的两个部分的shape", s_t, a_t_i)
            #     q_value_i = maddpg.critics(s_t, a_t_i)
            #     a_i = th.max(curr_actions[agent_i], 0)[1]
            #     # print("!!!!!!!!!!!!!!!!!!!!!!!!------------------------用来计算advantage的两部分的值为：", q_value, q_value_i)
            #     advantage_i = (q_value - q_value_i) * 10000
            #     maddpg.bufferB[agent_i].push(a_i, advantage_i)
            # print("本次episode的reward为：", sum_reward)
            '''
            第六步：跳出时间片循环，maddpg利用buffer中现有的数据，进行网络的更新
            '''
            c_loss, a_loss = maddpg.learn3()
            # print("输出所有的loss：", c_loss, a_loss)
            '''
                for agent_j in range(n_agents):
                    if agent_j == 0:
                        curr_state_j = curr_state.get_obs(agent_j)
                        curr_reward_j = np.array([curr_reward[agent_j]])
                    else:
                        curr_state_jj = curr_state.get_obs(agent_j)
                        curr_state_j = np.row_stack((curr_state_j, curr_state_jj))
                        curr_reward_jj = np.array([curr_reward[agent_j]])
                        curr_reward_j = np.row_stack((curr_reward_j, curr_reward_jj))
    
                first_state_j = th.from_numpy(first_state_j).float()
                curr_state_j = th.from_numpy(curr_state_j).float()
                curr_reward_j = th.from_numpy(curr_reward_j).float()
                maddpg.memory.push(first_state_j, curr_actions, curr_state_j, curr_reward_j)
                # print("存入buffer中的转换对内容为：", first_state_j, curr_actions, curr_state_j, curr_reward_j)
                c_loss, a_loss = maddpg.update_policy()
                # print("完成一次训练,输出当前网络的损失值：", c_loss, a_loss)
        '''
            # 画图部分
            # print("输出本episode的sum_reward:", sum_reward)
            # print("输出本episode的sum_times:", sum_times)
            # print("输出返回的c_loss：", c_loss)
            # print("n_agent个a_loss：", a_loss)
            # print("c_loss, a_loss:", c_loss, a_loss)
            if c_loss != None and a_loss != None:
                c_loss1 = c_loss
                a_loss1 = a_loss
                list_c_loss[count] = c_loss1[0]
                list_zero[count] = a_loss1[0]
                list_one[count] = a_loss1[1]
                list_two[count] = a_loss1[2]
                list_three[count] = a_loss1[3]
                list_four[count] = a_loss1[4]
                list_five[count] = a_loss1[5]
                list_six[count] = a_loss1[6]
                list_seven[count] = a_loss1[7]
                list_eight[count] = a_loss1[8]
                list_nine[count] = a_loss1[9]
                count = count + 1
        sum_lei[count_lie_re] = sum_reward
        sum_times[count_lie_re] = sum_time
        count_lie_re = count_lie_re + 1
        if i_episode > maddpg.episodes_before_train \
                and i_episode % 1 == 0:
            # print("保存第", end='')
            # print(i_episode, end='')
            # print("个episode的网络参数", end='')
            # print("当前的时间为：", time.ctime())
            if sum_time >= max_times:
                max_times = sum_time
                th.save(maddpg.critics, 'f_critic.pkl')
                th.save(maddpg.actors[0], 'f_zero_actor.pkl')
                th.save(maddpg.actors[1], 'f_one_actor.pkl')
                th.save(maddpg.actors[2], 'f_two_actor.pkl')
                th.save(maddpg.actors[3], 'f_three_actor.pkl')
                th.save(maddpg.actors[4], 'f_four_actor.pkl')
                th.save(maddpg.actors[5], 'f_five_actor.pkl')
                th.save(maddpg.actors[6], 'f_six_actor.pkl')
                th.save(maddpg.actors[7], 'f_seven_actor.pkl')
                th.save(maddpg.actors[8], 'f_eight_actor.pkl')
                th.save(maddpg.actors[9], 'f_nine_actor.pkl')
                print("当前最大收集次数为：", sum_time, end='')
                print("对应的公平性大小为：", curr_state.jfairness(), end='')
                print("对应的reward大小为：", sum_reward)
            if i_episode % 100 == 0 and i_episode > 0:
                print('    当前的episode为:', maddpg.episode_done, end='')
                print("    本次episode的总计收集次数为：", sum_time, end='')
                print("    收集的公平性大小为：", curr_state.jfairness(), end='')
                print("    reward的大小为：", sum_reward, end='')
                print("    对应的c_loss大小为：", c_loss[0])
                th.save(maddpg.critics, 'critic.pkl')
                th.save(maddpg.actors[0], 'zero_actor.pkl')
                th.save(maddpg.actors[1], 'one_actor.pkl')
                th.save(maddpg.actors[2], 'two_actor.pkl')
                th.save(maddpg.actors[3], 'three_actor.pkl')
                th.save(maddpg.actors[4], 'four_actor.pkl')
                th.save(maddpg.actors[5], 'five_actor.pkl')
                th.save(maddpg.actors[6], 'six_actor.pkl')
                th.save(maddpg.actors[7], 'seven_actor.pkl')
                th.save(maddpg.actors[8], 'eight_actor.pkl')
                th.save(maddpg.actors[9], 'nine_actor.pkl')
                # 画图
                x = np.arange(maddpg.episode_done)
                x1 = np.arange(count_lie_re)
                # 画loss的图
                y0 = list_c_loss[x]
                y1 = list_zero[x]
                y2 = list_one[x]
                y3 = list_two[x]
                y4 = list_three[x]
                y5 = list_four[x]
                y6 = list_five[x]
                y7 = list_six[x]
                y8 = list_seven[x]
                y9 = list_eight[x]
                y12 = list_nine[x]
                # 画累计收益和次数的图
                y10 = sum_lei[x1]
                y11 = sum_times[x1]
                plt.close('all')
                plt.figure(num=13, figsize=(10, 10))
                plt.plot(x, y0, color='r', label='central_critic')
                plt.legend(loc='best')
                plt.savefig('.//0.png')

                plt.figure(num=1, figsize=(10, 10))
                plt.plot(x, y1, color='r', label='zero_actor')
                plt.legend(loc='best')
                plt.savefig('.//1.png')

                plt.figure(num=2, figsize=(10, 10))
                plt.plot(x, y2, 'r--', label='one_actor')
                plt.legend(loc='best')
                plt.savefig('.//2.png')

                plt.figure(num=3, figsize=(10, 10))
                plt.plot(x, y3, color='g', label='two_actor')
                plt.legend(loc='best')
                plt.savefig('.//3.png')

                plt.figure(num=4, figsize=(10, 10))
                plt.plot(x, y4, 'g--', label='three_actor')
                plt.legend(loc='best')
                plt.savefig('.//4.png')

                plt.figure(num=5, figsize=(10, 10))
                plt.plot(x, y5, 'b', label='four_actor')
                plt.legend(loc='best')
                plt.savefig('.//5.png')

                plt.figure(num=6, figsize=(10, 10))
                plt.plot(x, y6, 'b--', label='five_actor')
                plt.legend(loc='best')
                plt.savefig('.//6.png')

                plt.figure(num=7, figsize=(10, 10))
                plt.plot(x, y7, 'b', label='six_actor')
                plt.legend(loc='best')
                plt.savefig('.//7.png')

                plt.figure(num=8, figsize=(10, 10))
                plt.plot(x, y8, 'b', label='seven_actor')
                plt.legend(loc='best')
                plt.savefig('.//8.png')

                plt.figure(num=9, figsize=(10, 10))
                plt.plot(x, y9, 'b', label='eight_actor')
                plt.legend(loc='best')
                plt.savefig('.//9.png')

                plt.figure(num=10, figsize=(10, 10))
                plt.plot(x, y12, 'r', label='nine_actor')
                plt.legend(loc='best')
                plt.savefig('.//12.png')

                plt.figure(num=11, figsize=(10, 10))
                plt.plot(x1, y11, 'r', label='sum_times')
                plt.legend(loc='best')
                plt.savefig('.//11.png')

                plt.figure(num=12, figsize=(10, 10))
                plt.plot(x1, y10, 'r', label='sum_reward')
                plt.legend(loc='best')
                plt.savefig('.//10.png')
                # plt.subplot(3, 2, 6)

        maddpg.episode_done += 1

        '''
        设置另外一个网络终止条件
        '''

    print('Ending time: ', time.ctime())
    end = time.perf_counter()
    running_time = end - start
    print("程序运行的时间为", running_time, 's')

    '''
    存储网络
    '''

    th.save(maddpg.critics, 'zero_critic.pkl')
    th.save(maddpg.actors[0], 'zero_actor.pkl')
    th.save(maddpg.actors[1], 'one_actor.pkl')
    th.save(maddpg.actors[2], 'two_actor.pkl')
    th.save(maddpg.actors[3], 'three_actor.pkl')
    th.save(maddpg.actors[4], 'four_actor.pkl')
    th.save(maddpg.actors[5], 'five_actor.pkl')
    th.save(maddpg.actors[6], 'six_actor.pkl')
    th.save(maddpg.actors[7], 'seven_actor.pkl')
    th.save(maddpg.actors[8], 'eight_actor.pkl')
    th.save(maddpg.actors[9], 'nine_actor.pkl')
    plt.show()


def statistics(curr_actions):
    # 统计所有智能体选择0-8各个动作的数量
    # 传过来的curr_actions得shape是n_agents * 9（0 1 矩阵）
    # 首先将其变成0-8的数字 再进行统计
    # print("处理前curr_actions矩阵为：", curr_actions)
    actions = th.zeros(n_agents)
    for i in range(n_agents):
        actions[i] = th.max(curr_actions[i], 0)[1]
    # print("处理完的actions为：", actions)
    action_statistics = th.zeros(9)
    for i_d in range(n_agents):
        # print("输出用作引用的id：", actions[i_d])
        actions_id_item = int(actions[i_d].item())
        # print("输出actions_id_item：", actions_id_item)
        action_statistics[actions_id_item] += 1
    # print("actions的统计为：", action_statistics)
    return action_statistics


def statistics_i(agent_i, curr_actions):
    actions = th.zeros(n_agents)
    for i in range(n_agents):
        actions[i] = th.max(curr_actions[i], 0)[1]
    action_statistics = th.zeros(9)
    for i_d in range(n_agents):
        if i_d != agent_i:
            actions_id_item = int(actions[i_d].item())
            action_statistics[actions_id_item] += 1
    # print("actions_i的统计为：", action_statistics)
    return action_statistics


if __name__ == "__main__":
    print("进入程序执行入口：")
    th.autograd.set_detect_anomaly(True)
    __main__()


