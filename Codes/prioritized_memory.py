import random
import numpy as np
from SumTree import SumTree


class Memory:
    '''
    The prioritized replay memory class.
    Ref: Schaul, T., et al. (2015). "Prioritized experience replay."
    Class methods:
        __init__: initialization function
        _get_priority: Calculate priority given a td error
        add: add a transition into the replay memory
        sample: sample a batch of transitions from the replay memory
        update: update the priority of the corresponding transition
    '''
    e = 0.01  # epsilon which is a small positive constant that prevents edge-case of transitions not being revisited
    # once their error is zero
    alpha = 0.6  # parameter in the paper
    beta = 0.4  # parameter in the paper

    def __init__(self, memory_size, num_state):
        '''
        Initialization function
        Input:
            memory_size: the memory size
            num_state: the dimension of observation space
        '''
        self.sumtree = SumTree(memory_size, num_state)  # instantiate a SumTree
        self.memory_size = memory_size
        self.num_state = num_state

    def _get_priority(self, td_error):
        '''
        Calculate priority given a td error.
        Input:
            td_error: td error
        Output:
            priority of the corresponding transition
        '''
        return (np.abs(td_error) + self.e) ** self.alpha  # p^alpha, formula 1 in the paper

    def add(self, td_error, transition):
        '''
        Add a transition into the replay memory.
            td_error: the td error of the corresponding transition
            transition: (s,a,r,s')
        '''
        priority = self._get_priority(td_error)
        self.sumtree.add(priority, transition)

    def sample(self, batch_size, i_episode, episode_max):
        '''
        Sample a batch of transitions from the replay memory.
        Input:
            batch_size: batch size
            i_episode: the index of current episode
            episode_max: the amount of episodes used to train policies
        Output:
            batch: the batch of transitions
            indexs: the corresponding indexes (index in self.sumtree.memory)
            omega: the corresponding importance-sampling weight
        '''
        batch = np.zeros((batch_size, self.sumtree.memory.shape[1]))  # the buffer of samples
        indexs = np.zeros((batch_size, 1), dtype=int)  # the corresponding indexes in the SumTree
        priorities = np.zeros((batch_size, 1))  # the corresponding priorities
        segment = self.sumtree.total() / batch_size  # sample a transition every 'segment' priorities
        beta = np.min([1., self.beta + (1 - self.beta) * i_episode / episode_max])  # the parameter in the paper

        for i in range(batch_size):
            a = segment * i  # the beginning index to sample
            b = segment * (i + 1)  # the ending index to sample

            s = random.uniform(a, b)  # sample a real number from [a,b)
            (index, priority, transition) = self.sumtree.get(s)  # sample a transition from the SumTree
            # add the transition into the batch
            indexs[i, 0] = index
            priorities[i, 0] = priority
            batch[i:i + 1, :] = transition
        # Calculate the importance-sampling weights
        P = priorities / self.sumtree.total()
        omega = np.power(self.sumtree.n_entries * P, -beta)
        omega = omega / np.max(omega)

        return batch, indexs, omega

    def update(self, index, td_error):
        '''
        Update the priority of the corresponding transition.
        Input:
            index: the index of the corresponding transition in the SumTree
            td_error: the td error of the corresponding transition
        '''
        priority = self._get_priority(td_error)
        self.sumtree.update(index, priority)
