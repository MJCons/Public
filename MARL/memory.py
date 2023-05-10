from collections import namedtuple
import random
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))

# BufferU 用来存储critic网络训练需要的经验
BufferU_experience = namedtuple('BufferU_experience',
                                ('q_value', 'sum_reward'))

# BufferB 用来存储actor网络训练需要的经验
BufferB_experience = namedtuple('BufferB_experience',
                                ('action', 'A_a'))

BufferC_experience = namedtuple('BufferC_experience', ('states', 'obs', 'next_states', 'action_statistic', 'rewards'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BufferU:
    # 更新critic 时是要取出所有的转换对
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = BufferU_experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, K):
        return random.sample(self.memory, K)

    def __len__(self):
        return len(self.memory)


class BufferB:
    # 更新actor 时是要取出K个转换对
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = BufferB_experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, K):
        return random.sample(self.memory, K)

    def __len__(self):
        return len(self.memory)


class BufferC:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = BufferC_experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
