import numpy


class SumTree:
    '''
    The SumTree class
    Class methods:
        __init__: initialization function
        _propagate: update the priorities of all parent nodes
        _retrieve: sample a transition index from self.tree
        total: calculate the amount of priorities of all transitions
        add: add a transition into the SumTree
        update: update the priorities of all nodes
        get: sample a transition from the SumTree
    '''

    def __init__(self, memory_size, num_state):
        '''
        Initialization function.
        Input:
            memory_size: the size of replay memory
            num_state: the dimension of observation space
        '''
        self.memory_size = memory_size
        self.tree = numpy.zeros((2 * memory_size - 1, 1))  # the priorities of all nodes
        self.memory = numpy.zeros((memory_size, num_state * 2 + 2 + 1))  # the transitions saved in the SumTree
        self.memory_counter = 0  # the counter for adding a new transition
        self.n_entries = 0  # the amount of transitions saved in the SumTree

    def _propagate(self, index, change):
        '''
        Update the priorities of all parent nodes
        Input:
            index: the transition index whose priority has been update (index in self.tree)
            change: the difference between old and new priorities
        '''
        parent = (index - 1) // 2  # the index of parent node
        self.tree[parent, 0] = self.tree[parent, 0] + change  # update the priority of the parent node
        if parent != 0:  # if the parent node is not root node, update its parent node
            self._propagate(parent, change)

    def _retrieve(self, index, s):
        '''
        Sample a transition index from self.tree
        Input:
            index: the index of parent node (index in self.tree)
            s : the sampled priority value
        Output:
            the index in self.tree
        '''
        left = 2 * index + 1  # the index of left child node
        right = left + 1  # the index of right child node
        if left >= self.tree.size:  # if 'index' corresponds to a transition rather than the parent node of transitions
            return index
        if s <= self.tree[left, 0]:  # if the sampled priority value is less than the left child node's priority
            return self._retrieve(left, s)  # search the left child node
        else:
            return self._retrieve(right, s - self.tree[left, 0])  # search the right child node

    def total(self):
        '''
        Calculate the amount of priorities of all transitions.
        Output:
            the amount of priorities of all transitions
        '''
        return self.tree[0, 0]

    def add(self, priority, transition):
        '''
        Add a transition into the SumTree.
        Input:
            priority: the priority of the transition
            transition: (s,a,r,s')
        '''
        index = self.memory_counter + self.memory_size - 1  # the node index in self.tree
        self.memory[self.memory_counter:self.memory_counter + 1, :] = transition  # save the transition
        self.update(index, priority)  # update the priorities of all nodes
        # update self.memory_counter and self.n_entries
        self.memory_counter += 1
        if self.memory_counter >= self.memory_size:
            self.memory_counter = 0

        if self.n_entries < self.memory_size:
            self.n_entries += 1

    def update(self, index, priority):
        '''
        Update the priorities of all nodes.
        Input:
            index: the transition index whose priority will be updated
            priority: the new priority value
        '''
        change = priority - self.tree[index, 0]  # the difference between old and new priorities
        self.tree[index, 0] = priority  # update the priority of the transition
        self._propagate(index, change)  # update the priority of its parent nodes

    # get priority and sample
    def get(self, s):  # s为采样值
        '''
        Sample a transition from the SumTree
        Input:
            s: the sampled priority value
        Output:
            index: the index in self.memory
            self.tree[index, 0]: the corresponding priority
            self.memory[dataIdx:dataIdx + 1, :]: the transition
        '''
        index = self._retrieve(0, s) # the index in self.tree
        dataIdx = index - self.memory_size + 1 # the index in self.memory

        return (index, self.tree[index, 0], self.memory[dataIdx:dataIdx + 1, :])
