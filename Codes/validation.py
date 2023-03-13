import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import environment
import random
import os
import math
import copy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
gamma = 0.99  # discount factor
device = torch.device('cuda')

# random seed
# randomseed = 69963
# torch.manual_seed(randomseed)
# np.random.seed(randomseed)
# random.seed(randomseed)

def validation(algorithm):
    env = environment.environment(gamma, 'Valid', algorithm)  # instantiate the environment
    num_action = env.num_action  # the number of discretized actions
    num_agent = env.num_agent

    class Net_attention2(nn.Module):
        def __init__(self):
            super(Net_attention2, self).__init__()
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
            mean_embedding = (x[0::2, :] + x[1::2, :]) / 2
            mean_embedding = torch.hstack((mean_embedding, mean_embedding)).reshape(-1, mean_embedding.shape[1])
            # feature = torch.vstack((x[0::2, :].reshape(1, -1), x[1::2, :].reshape(1, -1)))
            # mean_feature = torch.mean(feature, dim=0, keepdim=True).reshape(-1, x.shape[1])
            # attention
            key = torch.relu(self.key(x))
            attention_input = torch.hstack((key, mean_embedding,
                                            torch.reshape(torch.hstack((state_loc, state_loc)),
                                                          (-1, state_loc.shape[1]))))
            attention_score = self.attention(attention_input)
            attention_score_normalized = self.softmax(torch.reshape(attention_score, (-1, 2))).reshape((-1, 1))
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

            return action_value, attention_score_normalized.detach().to('cpu').numpy().reshape(1, -1)

    for file in ['4000']:
        # file = '1000'
        t_per_episode = list()
        is_collision = list()
        if algorithm in ['attention2']:
            net = Net_attention2()
            net.load_state_dict(torch.load(file + '.pt', map_location='cuda'))
        # load parameters
        net.to(device)
        for num_episode in range(1000):
            state = env.reset()  # reset the environment
            while True:
                env.render()  # render the pursuit process
                # plt.savefig(str(j)) # save figures
                if algorithm in ['attention2']:
                    action = np.zeros((1, 0))  # action index buffer
                    attention_score_array = np.zeros((num_agent, num_agent - 1))
                    # choose actions
                    for i in range(num_agent):
                        temp = state[:, i:i + 1].reshape(1, -1)
                        temp, attention_score = net(
                            torch.tensor(np.ravel(temp), dtype=torch.float32, device=device).view(1, -1))
                        action_temp = torch.max(temp, 1)[1].data.to('cpu').numpy()
                        action = np.hstack((action, np.array(action_temp, ndmin=2)))
                        attention_score_array[i:i + 1, :] = attention_score
                    # execute actions
                    state_next, reward, done = env.step(action, attention_score_array)
                # trajectory = np.vstack((state, action, reward, done))
                # trajectory_record[:, :, env.t, num_episode] = trajectory
                temp1 = done == 1
                temp2 = done == 2
                temp = np.vstack((temp1, temp2))
                if np.any(done == 3):
                    # if collisions occur
                    t_per_episode.append(1000)
                    is_collision.append(1)
                    break
                if np.all(np.any(temp, axis=0, keepdims=True)):
                    # if all pursuers capture the evader or the episode reaches maximal length
                    t_per_episode.append(env.t)
                    is_collision.append(0)
                    break
                state = state_next
        np.savetxt(file + '_time.txt', t_per_episode)
        np.savetxt(file + '_collision.txt', is_collision)
        # np.save(file, trajectory_record)


if __name__ == '__main__':
    validation('attention2')
