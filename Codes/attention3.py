import torch
import torch.nn as nn
import numpy as np
import environment
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import random
from prioritized_memory import Memory
from validation import validation
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Hyper Parameters
episode_max = 4000  # the amount of episodes used to train
batch_size = 128  # batch size
lr = 3e-5  # learning rate
epsilon_origin = 1  # original epsilon
epsilon_decrement = 1 / 2000  # epsilon decay
gamma = 0.99  # discount factor
target_replace_iter = 1000  # update frequency of target network
memory_size = int(1e6)  # the size of replay memory
env = environment.environment(gamma, 'Train', 'attention2')  # instantiate the environment

kl_weight = 0.05  # weight of KL divergence

num_action = env.num_action  # the number of discretized actions
num_state = env.num_state  # the dimension of state space
num_agent = env.num_agent  # the number of pursuers

device = torch.device('cuda')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128 + 5, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, num_action)
        self.layer6 = nn.Linear(64, 1)
        self.key = nn.Linear(128, 64)
        self.attention = nn.Linear(64 + 2 + 2 + 1 + 128, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        state_loc = x[:, :5]
        state_ij = x[:, 5:]
        # embedding
        x = state_ij.reshape(-1, 2)
        x = torch.relu(self.layer1(x))
        # mean embedding
        mean_embedding = (x[0::2, :] + x[1::2, :]) / 2
        mean_embedding = torch.hstack((mean_embedding, mean_embedding)).reshape(-1, mean_embedding.shape[1])
        # attention
        key = torch.relu(self.key(x))
        attention_input = torch.hstack((key, mean_embedding,
                                        torch.reshape(torch.hstack((state_loc, state_loc)), (-1, state_loc.shape[1]))))
        attention_score = self.attention(attention_input)
        attention_score_normalized = self.softmax(torch.reshape(attention_score, (-1, 2))).reshape((-1, 1))
        # attention-based embedding
        temp = attention_score_normalized * x
        weighted_feature = temp[0::2, :] + temp[1::2, :]
        # concatenate
        network_input = torch.hstack((state_loc, weighted_feature))
        # calculate advantage
        x = torch.relu(self.layer2(network_input))
        advantage = torch.relu(self.layer3(x))
        advantage = self.layer5(advantage)
        # calculate state value
        state_value = torch.relu(self.layer4(x))
        state_value = self.layer6(state_value)
        # calculate Q value
        action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_value, attention_score_normalized.reshape(-1, 2)


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # evaluation network and target network
        self.learn_step_counter = 0  # the counter of update
        self.memory = Memory(memory_size, env.num_state)  # the prioritized replay memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # adam optimizer
        self.loss_func = nn.MSELoss(reduction='none')  # MSE loss
        self.max_td_error = 0.  # the maximal td error

    def choose_action(self, state, epsilon):
        # observations
        state = torch.tensor(np.ravel(state), dtype=torch.float32, device=device).view(1, -1)
        # epsilon-greedy method
        if np.random.uniform() > epsilon:
            # choose the action with maximal Q value
            actions_value, attention_score = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

        else:
            # choose the action randomly
            actions_value, attention_score = self.eval_net(state)
            action = np.random.randint(0, num_action)
        return action, attention_score.detach().to('cpu').numpy()

    def store_transition(self, state, action, reward, state_next, done):
        # add transitions into replay memory
        transition = np.hstack(
            (np.ravel(state), np.ravel(action), np.ravel(reward), np.ravel(state_next), np.ravel(done)))
        self.memory.add(self.max_td_error, transition)

    def learn(self, i_episode):
        if self.learn_step_counter % target_replace_iter == 0:
            # periodically update the target network
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample transition batch
        b_memory, indexs, omega = self.memory.sample(batch_size, i_episode, episode_max)
        b_state = torch.tensor(b_memory[:, :num_state], dtype=torch.float32, device=device)
        b_action = torch.tensor(b_memory[:, num_state:num_state + 1], dtype=torch.int64, device=device)
        b_reward = torch.tensor(b_memory[:, num_state + 1:num_state + 2], dtype=torch.float32, device=device)
        b_state_next = torch.tensor(b_memory[:, num_state + 2:num_state * 2 + 2], dtype=torch.float32, device=device)
        b_done = torch.tensor(b_memory[:, num_state * 2 + 2:num_state * 2 + 3], dtype=torch.float32, device=device)
        # calculate Q values
        temp1, attention_now = self.eval_net(b_state)
        q_eval = temp1.gather(1, b_action)
        q_next_targetnet, _ = self.target_net(b_state_next)
        q_next_evalnet, _ = self.eval_net(b_state_next)
        q_target = b_reward + gamma * torch.abs(1 - b_done) * q_next_targetnet.gather(1, torch.argmax(q_next_evalnet,
                                                                                                      axis=1,
                                                                                                      keepdim=True))
        # calculate td errors
        td_errors = (q_target - q_eval).to('cpu').detach().numpy().reshape((-1, 1))
        # update prioritized replay memory
        self.max_td_error = max(np.max(np.abs(td_errors)), self.max_td_error)
        for i in range(batch_size):
            index = indexs[i, 0]
            td_error = td_errors[i, 0]
            self.memory.update(index, td_error)
        ##########################
        # KL divergence
        _, attention_old = self.target_net(b_state)
        attention_old = attention_old.detach()
        loss_kl = attention_old[:, 0:1] * torch.log(attention_old[:, 0:1] / attention_now[:, 0:1]) + \
                  attention_old[:, 1:2] * torch.log(attention_old[:, 1:2] / attention_now[:, 1:2])
        ##########################
        # calculate loss
        loss = (self.loss_func(q_eval, q_target.detach()) * torch.FloatTensor(omega).to(device).detach() + \
                loss_kl * kl_weight).mean()
        # back propagate
        self.optimizer.zero_grad()
        loss.backward()
        # update parameters
        self.optimizer.step()


class RunningStat:
    def __init__(self):
        self.n = 0  # the number of reward signals collected
        self.mean = np.zeros((1,))  # the mean of all rewards
        self.s = np.zeros((1,))
        self.std = np.zeros((1,))  # the std of all rewards

    def push(self, x):
        self.n += 1  # update the number of reward signals collected
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n  # update mean
            self.s = self.s + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.s / (self.n - 1) if self.n > 1 else np.square(self.mean))  # update std


dqn = DQN()
running_stat = RunningStat()
dqn.eval_net.to(device)
dqn.target_net.to(device)

i_episode = 0

episode_return_total = np.zeros(0)  # episode return recorder

while True:
    state = env.reset()  # reset the environment
    last_done = np.array([[0., 0, 0]])  # whether pursuers are inactive at the last timestep
    episode_return = 0  # cumulative reward

    while True:
        # env.render()  # render the training process
        action = np.zeros((1, num_agent))  # action index buffer
        attention_score_array = np.zeros((num_agent, num_agent - 1))
        # choose actions
        for i in range(num_agent):
            temp = state[:, i:i + 1].reshape(1, -1)
            action_temp, attention_score = dqn.choose_action(temp,
                                                             max(epsilon_origin - epsilon_decrement * i_episode, 0.01))
            action[0, i] = action_temp
            attention_score_array[i:i + 1, :] = attention_score
        # execute actions
        state_next, reward, done = env.step(action, attention_score_array)

        for i in range(num_agent):
            if not np.ravel(last_done)[i]:  # if the pursuer is active at the last timestep
                episode_return += np.ravel(reward)[i]
                running_stat.push(reward[:, i])
                reward[0, i] = np.clip(reward[0, i] / (running_stat.std + 1e-8), -10, 10)  # reward normalization
                # store the transition
                dqn.store_transition(state[:, i:i + 1], action[:, i:i + 1], reward[:, i:i + 1], state_next[:, i:i + 1],
                                     done[:, i:i + 1])

        if np.all(done):  # if all pursuers are inactive (the episode finishes)
            if dqn.memory.sumtree.n_entries == memory_size:  # if the replay memory collects enough transitions
                for _ in range(1000):
                    dqn.learn(i_episode)  # train the network
                i_episode += 1
            break
        state = state_next
        last_done = done
    # if the replay memory doesn't collect enough transitions, print information
    if dqn.memory.sumtree.n_entries < memory_size:
        temp = "collecting experiences: " + str(dqn.memory.sumtree.n_entries) + ' / ' + str(memory_size)
        print(temp)
    # if the replay memory collects enough transitions, plot cumulative reward and print information
    if dqn.memory.sumtree.n_entries == memory_size:
        episode_return_total = np.hstack((episode_return_total, episode_return))
        print('i_episode: ', i_episode, 'episode_return: ', round(episode_return, 2))
    # periodically save networks
    if i_episode % 500 == 0:
        net = dqn.eval_net
        string = str(i_episode) + '.pt'
        torch.save(net.state_dict(), string)
        string = str(i_episode) + '.txt'
        np.savetxt(string, episode_return_total)
    # kill the training process
    if i_episode == episode_max:
        validation('attention2')
        break
