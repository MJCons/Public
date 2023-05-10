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

n_agents = 3
n_states = 12
n_actions = 9
capacity = 16384
batch_size = 1024

n_episode = 30000
max_steps = 40
episodes_before_train = 15


maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

def __main__():

    t1 = time.ctime()
    print("主程序运行开始：", t1)
    start = time.perf_counter()

    list_zero = np.zeros(1000000)
    list_one = np.zeros(1000000)
    list_two = np.zeros(1000000)
    list_three = np.zeros(1000000)
    list_four = np.zeros(1000000)
    list_five = np.zeros(1000000)
    count = 0

    lei_episode_one = np.zeros(30000)
    lei_episode_two = np.zeros(30000)
    lei_episode_three = np.zeros(30000)
    sum_lei = np.zeros(30000)
    sum_times = np.zeros(30000)
    count_lie_re = 0
    max_times = 0

    for i_episode in range(n_episode):
        lei_reward_episode_one = 0
        lei_reward_episode_two = 0
        lei_reward_episode_three = 0
        sum_reward = 0
        sum_time = 0
        # print('当前的episode为:   ', maddpg.episode_done)
        curr_state = env()  # 生成一个新的环境
        for t in range(max_steps):
            # print("当前step为：", t)
            if t % 1 == 0 and t > 0:
                curr_state.step_update(t)
            for i in range(n_agents):
                if i == 0:
                    curr_obs = curr_state.get_obs(i)
                    # print("obs1:", curr_obs)
                else:
                    curr_obs_i = curr_state.get_obs(i)
                    curr_obs = np.row_stack((curr_obs, curr_obs_i))
            curr_obs = th.from_numpy(curr_obs).float()
            curr_obs = curr_obs.type(FloatTensor)
            curr_actions = maddpg.select_action(curr_obs)
            # print("生成的动作为：", curr_actions)
            # print("对应的tensor.shape为：", curr_actions.shape)

            first_state = curr_state
            for agent_f in range(n_agents):
                if agent_f == 0:
                    first_state_j = first_state.get_obs(agent_f)
                else:
                    first_state_jj = first_state.get_obs(agent_f)
                    first_state_j = np.row_stack((first_state_j, first_state_jj))
            next_state, curr_reward, times = curr_state.update_state(curr_actions, n_agents)
            lei_reward_episode_one = lei_reward_episode_one + curr_reward[0]
            lei_reward_episode_two = lei_reward_episode_two + curr_reward[1]
            lei_reward_episode_three = lei_reward_episode_three + curr_reward[2]
            sum_reward = sum_reward + curr_reward[0] + curr_reward[1] + curr_reward[2]
            sum_time = sum_time + times
            curr_state = next_state
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

            if c_loss != None and a_loss != None:
                    c_loss1 = c_loss
                    a_loss1 = a_loss
                    list_zero[count] = c_loss1[0]
                    list_one[count] = c_loss1[1]
                    list_two[count] = c_loss1[2]
                    list_three[count] = a_loss1[0]
                    list_four[count] = a_loss1[1]
                    list_five[count] = a_loss1[2]
                    count = count + 1
        lei_episode_one[count_lie_re] = lei_reward_episode_one
        lei_episode_two[count_lie_re] = lei_reward_episode_two
        lei_episode_three[count_lie_re] = lei_reward_episode_three
        sum_lei[count_lie_re] = sum_reward
        sum_times[count_lie_re] = sum_time
        # print("本次episode的总计收集次数为：", sum_time)
        count_lie_re = count_lie_re + 1
        if i_episode > maddpg.episodes_before_train \
                and i_episode % 1 == 0:
            # print("保存第", end='')
            # print(i_episode, end='')
            # print("个episode的网络参数", end='')
            # print("当前的时间为：", time.ctime())
            if sum_time >= max_times:
                max_times = sum_time
                th.save(maddpg.critics[0], 'f_zero_critic.pkl')
                th.save(maddpg.critics[1], 'f_one_critic.pkl')
                th.save(maddpg.critics[2], 'f_two_critic.pkl')
                th.save(maddpg.actors[0], 'f_zero_actor.pkl')
                th.save(maddpg.actors[1], 'f_one_actor.pkl')
                th.save(maddpg.actors[2], 'f_two_actor.pkl')
                print("当前最大收集次数为：", sum_time, end='')
                print("对应的公平性大小为：", curr_state.jfairness())
            if i_episode % 100 == 0:
                print('当前的episode为:', maddpg.episode_done, end='')
                print("本次episode的总计收集次数为：", sum_time)
                print("本次收集的公平性大小为：", curr_state.jfairness())
                th.save(maddpg.critics[0], 'zero_critic.pkl')
                th.save(maddpg.critics[1], 'one_critic.pkl')
                th.save(maddpg.critics[2], 'two_critic.pkl')
                th.save(maddpg.actors[0], 'zero_actor.pkl')
                th.save(maddpg.actors[1], 'one_actor.pkl')
                th.save(maddpg.actors[2], 'two_actor.pkl')

                # 画图
                x = np.arange((maddpg.episode_done - episodes_before_train) * max_steps)
                x1 = np.arange(count_lie_re)
                y1 = list_zero[x]
                y2 = list_one[x]
                y3 = list_two[x]
                y4 = list_three[x]
                y5 = list_four[x]
                y6 = list_five[x]
                y7 = lei_episode_one[x1]
                y8 = lei_episode_two[x1]
                y9 = lei_episode_three[x1]
                y10 = sum_lei[x1]
                y11 = sum_times[x1]
                plt.close('all')
                plt.figure(num=1, figsize=(10, 10))
                plt.plot(x, y1, color='r', label='zero_critic')
                plt.legend(loc='best')
                plt.savefig('.//1.png')

                plt.figure(num=2, figsize=(10, 10))
                plt.plot(x, y4, 'r--', label='zero_actor')
                plt.legend(loc='best')
                plt.savefig('.//2.png')

                plt.figure(num=3, figsize=(10, 10))
                plt.plot(x, y2, color='g', label='one_critic')
                plt.legend(loc='best')
                plt.savefig('.//3.png')

                plt.figure(num=4, figsize=(10, 10))
                plt.plot(x, y5, 'g--', label='one_actor')
                plt.legend(loc='best')
                plt.savefig('.//4.png')

                plt.figure(num=5, figsize=(10, 10))
                plt.plot(x, y3, 'b', label='two_critic')
                plt.legend(loc='best')
                plt.savefig('.//5.png')

                plt.figure(num=6, figsize=(10, 10))
                plt.plot(x, y6, 'b--', label='two_actor')
                plt.legend(loc='best')
                plt.savefig('.//6.png')

                plt.figure(num=7, figsize=(10, 10))
                plt.plot(x1, y7, 'b', label='reward_zero')
                plt.legend(loc='best')
                plt.savefig('.//7.png')

                plt.figure(num=8, figsize=(10, 10))
                plt.plot(x1, y8, 'b', label='reward_one')
                plt.legend(loc='best')
                plt.savefig('.//8.png')

                plt.figure(num=9, figsize=(10, 10))
                plt.plot(x1, y9, 'b', label='reward_three')
                plt.legend(loc='best')
                plt.savefig('.//9.png')

                plt.figure(num=10, figsize=(10, 10))
                plt.plot(x1, y10, 'r', label='sum_reward')
                plt.legend(loc='best')
                plt.savefig('.//10.png')

                plt.figure(num=11, figsize=(10, 10))
                plt.plot(x1, y11, 'r', label='sum_times')
                plt.legend(loc='best')
                plt.savefig('.//11.png')
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

    th.save(maddpg.critics[0], 'zero_critic.pkl')
    th.save(maddpg.critics[1], 'one_critic.pkl')
    th.save(maddpg.critics[2], 'two_critic.pkl')
    th.save(maddpg.actors[0], 'zero_actor.pkl')
    th.save(maddpg.actors[1], 'one_actor.pkl')
    th.save(maddpg.actors[2], 'two_actor.pkl')
    plt.show()



if __name__ == "__main__":
    print("进入程序执行入口：")
    th.autograd.set_detect_anomaly(True)
    __main__()


