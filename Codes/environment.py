import numpy as np
import APF_function_for_DQN
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import escaper
import copy


class environment():
    def __init__(self, gamma, mode, algorithm):
        ####environment features#######
        self.mode = mode  # training mode or validation mode
        self.algorithm = algorithm
        self.num_agent = 3  # number of pursuers
        if self.algorithm in ['attention2']:
            self.num_action = 24  # number of discretized actions
        self.num_state = 4 + 1 + (self.num_agent - 1) * 2  # dimension of state space
        self.t = 0  # timestep
        self.v = 300  # velocity of pursuers (mm/s)
        self.delta_t = 0.1  # time interval (s)
        self.gamma = gamma  # discount factor
        self.r_perception = 2000  # sense range of pursuers
        #######agent features#######
        self.wall_following = np.zeros((1, self.num_agent))  # whether pursuers move according to wall following rules
        self.agent_orientation = np.zeros((2, self.num_agent))
        self.agent_orientation_last = np.zeros((2, self.num_agent))
        self.agent_orientation_origin = np.zeros((2, self.num_agent))
        self.agent_position = np.zeros((2, self.num_agent))
        self.agent_position_last = np.zeros((2, self.num_agent))
        self.agent_position_origin = np.zeros((2, self.num_agent))
        self.obstacle_with_other_agent = []
        self.obstacle_closest_with_other_agent = []
        self.obstacle_closest = np.zeros((2, self.num_agent))
        self.distance_from_target = np.zeros((1, self.num_agent))
        self.distance_from_target_last = np.zeros((1, self.num_agent))
        self.done = np.zeros((1, self.num_agent))
        self.state = np.zeros((self.num_state, self.num_agent))
        #######evader features########
        self.target_position = np.zeros((2, 1))
        self.target_orientation = np.zeros((2, 1))
        self.escaper_slip_flag = 0
        self.escaper_wall_following = 0
        self.escaper_zigzag_flag = 0
        self.last_e = np.zeros((2, 1))
        self.zigzag_count = 0
        self.zigzag_last = np.zeros((2, 1))
        #######obstacle features#######
        self.boundary = APF_function_for_DQN.generate_boundary(np.array([[0.0], [0]]), np.array([[3600], [0]]),
                                                               np.array([[3600], [5000]]), np.array([[0], [5000]]), 51)

        self.obstacle1 = APF_function_for_DQN.generate_boundary(np.array([[900.0], [1000]]),
                                                                np.array([[1550], [1000]]),
                                                                np.array([[1550], [1100]]), np.array([[900], [1100]]),
                                                                11)
        self.obstacle2 = APF_function_for_DQN.generate_boundary(np.array([[2050.0], [1000]]),
                                                                np.array([[2700], [1000]]),
                                                                np.array([[2700], [1100]]), np.array([[2050], [1100]]),
                                                                11)
        self.obstacle3 = APF_function_for_DQN.generate_boundary(np.array([[1400.0], [2450]]),
                                                                np.array([[2200], [2450]]),
                                                                np.array([[2200], [2550]]), np.array([[1400], [2550]]),
                                                                11)
        self.obstacle4 = APF_function_for_DQN.generate_boundary(np.array([[900.0], [3900]]),
                                                                np.array([[1550], [3900]]),
                                                                np.array([[1550], [4000]]), np.array([[900], [4000]]),
                                                                11)
        self.obstacle5 = APF_function_for_DQN.generate_boundary(np.array([[2050.0], [3900]]),
                                                                np.array([[2700], [3900]]),
                                                                np.array([[2700], [4000]]), np.array([[2050], [4000]]),
                                                                11)
        self.obstacle_total = np.hstack(
            (self.boundary, self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle4, self.obstacle5))

    def reset(self):
        self.wall_following = np.zeros((1, self.num_agent))  # whether pursuers move according to wall following rules
        self.agent_orientation = np.zeros((2, self.num_agent))
        self.agent_orientation_last = np.zeros((2, self.num_agent))
        self.agent_orientation_origin = np.zeros((2, self.num_agent))
        self.agent_position = np.zeros((2, self.num_agent))
        self.agent_position_last = np.zeros((2, self.num_agent))
        self.agent_position_origin = np.zeros((2, self.num_agent))
        self.obstacle_with_other_agent = []
        self.obstacle_closest_with_other_agent = []
        self.obstacle_closest = np.zeros((2, self.num_agent))
        self.distance_from_target = np.zeros((1, self.num_agent))
        self.distance_from_target_last = np.zeros((1, self.num_agent))
        self.done = np.zeros((1, self.num_agent))
        self.state = np.zeros((self.num_state, self.num_agent))
        #######evader features########
        self.target_position = np.zeros((2, 1))
        self.target_orientation = np.zeros((2, 1))
        self.escaper_slip_flag = 0
        self.escaper_wall_following = 0
        self.escaper_zigzag_flag = 0
        self.last_e = np.zeros((2, 1))
        self.zigzag_count = 0
        self.zigzag_last = np.zeros((2, 1))

        self.t = 0
        # initialize evader's positions and headings
        self.target_position = np.random.random((2, 1))
        self.target_position[0] = self.target_position[0] * 3200 + 200
        self.target_position[1] = self.target_position[1] * 600 + 4200
        self.target_orientation = np.array([[0.], [1]])
        # initialize pursuers' positions and headings
        self.agent_position = np.random.random((2, 1))
        self.agent_position[0, :] = self.agent_position[0, :] * 2400 + 200
        self.agent_position[1, :] = self.agent_position[1, :] * 600 + 200
        self.agent_position = self.agent_position.repeat(3, axis=1) + np.array([[0, 400, 800], [0, 0, 0]])
        self.agent_orientation = np.vstack((np.zeros((1, self.num_agent)), np.ones((1, self.num_agent))))

        self.agent_position_origin = self.agent_position  # original positions
        self.agent_orientation_origin = self.agent_orientation  # original headings

        self.update_feature()
        self.update_state()  # update environment's state

        return self.state

    def reward(self):
        reward = np.zeros((1, self.num_agent))  # reward buffer
        done = np.zeros((1, self.num_agent))  # done buffer
        position_buffer = copy.deepcopy(self.agent_position)
        success_flag = np.any(self.done)  # whether the evader is inactive

        for i in range(self.num_agent):
            reward2 = 0  # r_col_1
            reward3 = 0  # r_col_2
            reward4 = 0  # r_app
            if success_flag:  # if the evader is inactive
                success_range = 300  # d_c
            else:
                success_range = 200
            if np.linalg.norm(self.agent_position[:,
                              i:i + 1] - self.target_position) < success_range:  # if the distance the evader is less than d_c
                reward1 = 20  # r_main
                done_temp = 1.  # the pursuer captures the evader successfully
            else:
                reward1 = 0
                done_temp = 0.
                if self.t == 1000:
                    done_temp = 2.  # pursuers failed though no collision

                if np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) > 150:
                    # if the distance from the nearest obstacle exceeds 150 mm
                    reward2 = 0
                elif np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
                    # if the distance from the nearest obstacle is less than 100 mm
                    reward2 = -20
                    if self.mode == 'Train':
                        # the pursuer collides and be moved to its original position
                        position_buffer[:, i:i + 1] = self.agent_position_origin[:, i:i + 1]
                    if self.mode == 'Valid':
                        # the pursuer is inactive
                        done_temp = 3.
                else:
                    reward2 = -2

                if np.amin(np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                                          axis=0)) > 200:
                    # if the distance from the nearest teammate exceeds 200 mm
                    reward3 = 0
                else:
                    reward3 = -20
                    if self.mode == 'Train':
                        # the pursuer collides and be moved to its original position
                        position_buffer[:, i:i + 1] = self.agent_position_origin[:, i:i + 1]
                    if self.mode == 'Valid':
                        # the pursuer is inactive
                        done_temp = 3.

                reward4 = (self.distance_from_target_last[0, i] - self.distance_from_target[0, i]) / 200

            reward[0, i] = reward1 + reward2 + reward3 + reward4  # the total reward
            done[0, i] = done_temp
        # in the training mode, initialize the collided pursuer, in validation mode, do nothing
        self.agent_position = position_buffer
        self.done = done

        return reward, done

    def step(self, action, attention_score_array=np.array([0])):
        if attention_score_array.size == 1:
            attention_score_array = np.zeros((self.num_agent, self.num_agent - 1))
        self.t += 1
        self.update_feature_last()
        ######agent#########
        if self.algorithm in ['attention2']:
            F, wall_following = self.from_action_to_APF(action, attention_score_array)
            F = np.round(F * 1000) / 1000
            for i in range(self.num_agent):
                F_APF = copy.deepcopy(F[:, i:i + 1])
                agent_orientation = self.agent_orientation[:, i:i + 1]
                temp = np.radians(30)
                if np.arccos(np.clip(np.dot(np.ravel(agent_orientation), np.ravel(F_APF)) / np.linalg.norm(
                        agent_orientation) / np.linalg.norm(F_APF), -1, 1)) > temp:
                    rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                    temp1 = np.matmul(rotate_matrix, agent_orientation)
                    rotate_matrix = np.array(
                        [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                    temp2 = np.matmul(rotate_matrix, agent_orientation)
                    if np.dot(np.ravel(temp1), np.ravel(F_APF)) > np.dot(np.ravel(temp2), np.ravel(F_APF)):
                        F_APF = temp1
                    else:
                        F_APF = temp2
                F[:, i:i + 1] = copy.deepcopy(F_APF)
            self.wall_following = wall_following

        agent_position_buffer = np.zeros((2, self.num_agent))
        for i in range(self.num_agent):
            if self.done[0, i]:  # if the pursuer is inactive, it will not move
                pass
            else:  # if the pursuer is active, calculate its displacement
                agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t

        #######escaper########
        # calculate the evader's displacement according to the escaping policy
        F_escaper, zigzag_count, zigzag_last, escaper_zigzag_flag, escaper_wall_following, escaper_slip_flag, distance_from_nearest_obstacle, last_e = escaper.escaper(
            self.agent_position, self.target_position,
            self.target_orientation, self.obstacle_total,
            self.num_agent, self.zigzag_count, self.zigzag_last, self.last_e, self.escaper_slip_flag)
        self.zigzag_last = zigzag_last
        self.zigzag_count = zigzag_count
        self.escaper_zigzag_flag = escaper_zigzag_flag
        self.escaper_wall_following = escaper_wall_following
        self.escaper_slip_flag = escaper_slip_flag
        self.last_e = last_e
        #####update#####
        self.agent_position = self.agent_position + agent_position_buffer  # update pursuers'positions
        self.agent_orientation = F  # update pursuers' headings

        if np.any(self.done) or distance_from_nearest_obstacle < 30:
            # if the evader is captured or collides with obstacles
            pass
        else:
            self.target_position = self.target_position + F_escaper * self.delta_t  # update the evader's position
            self.target_orientation = F_escaper  # update the evader's heading

        self.update_feature()
        reward, _ = self.reward()  # calculate reward function
        self.update_feature()
        self.update_state()  # update environment's state
        return self.state, reward, self.done

    def render(self):
        plt.figure(1)
        plt.cla()
        ax = plt.gca()
        plt.xlim([-100, 3700])
        plt.ylim([-100, 5100])
        ax.set_aspect(1)
        # plot obstacles and boundary
        plt.plot(self.obstacle1[0, :], self.obstacle1[1, :], 'black')
        plt.plot(self.obstacle2[0, :], self.obstacle2[1, :], 'black')
        plt.plot(self.obstacle3[0, :], self.obstacle3[1, :], 'black')
        plt.plot(self.obstacle4[0, :], self.obstacle4[1, :], 'black')
        plt.plot(self.obstacle5[0, :], self.obstacle5[1, :], 'black')
        plt.plot(self.boundary[0, :], self.boundary[1, :], 'black')
        # plot evader
        if self.escaper_slip_flag == 1:
            color = 'black'
        else:
            if self.escaper_wall_following == 1:
                color = 'green'
            else:
                if self.escaper_zigzag_flag == 1:
                    color = 'blue'
                else:
                    color = 'red'
        circle = mpatches.Circle(np.ravel(self.target_position), 100, facecolor=color)
        ax.add_patch(circle)
        # plot pursuers
        color = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(self.num_agent):
            circle = mpatches.Circle(self.agent_position[:, i], 100, facecolor=color[i])
            ax.add_patch(circle)
            if not self.done[0, i]:
                if self.wall_following[0, i]:
                    plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i],
                               self.agent_orientation[1, i], color='green', scale=10)
                else:
                    plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i],
                               self.agent_orientation[1, i], color='black', scale=10)

        plt.show(block=False)
        # plt.savefig(str(self.t))#whether save figures
        plt.pause(0.001)

    def update_state(self):
        '''
        Update the environment state (self.state and other class properties).
        '''
        self.state = np.zeros((self.num_state, self.num_agent))  # clear the environment state
        # clear the nearest obstacle list(considering virtual obstacles)
        for i in range(self.num_agent):
            # the distance form the nearest obstacle
            temp1 = self.obstacle_closest_with_other_agent[:, i:i + 1] - self.agent_position[:, i:i + 1]
            # the bearing of the nearest obstacle
            angle1 = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp1), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp1) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp1)) > 0:
                pass
            else:
                angle1 = -angle1
            # the distance from evader
            temp2 = self.target_position - self.agent_position[:, i:i + 1]
            # the bearing of evader
            angle2 = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp2), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp2) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp2)) > 0:
                pass
            else:
                angle2 = -angle2

            state = np.zeros((self.num_state,))  # state buffer
            state[:4] = np.array([np.linalg.norm(temp1) / 5000, angle1, np.linalg.norm(temp2) / 5000, angle2],
                                 dtype='float32')  # update state

            friends_position = np.delete(self.agent_position, i, axis=1)  # teammate positions
            for j in range(self.num_agent - 1):
                friend_position = friends_position[:, j:j + 1]  # teammate postion
                self_position = self.agent_position[:, i:i + 1]
                self_orientation = self.agent_orientation[:, i:i + 1]
                temp = friend_position - self_position
                distance = np.linalg.norm(temp)  # the distance from teammate
                # the bearing of teammate
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp), np.ravel(self_orientation)) / distance / np.linalg.norm(
                            self_orientation), -1, 1)) / np.pi
                if np.cross(np.ravel(self_orientation), np.ravel(temp)) > 0:
                    pass
                else:
                    angle = -angle
                # mask distant teammates and update state
                if distance < self.r_perception:
                    state[5 + 2 * j] = np.linalg.norm(temp) / 5000
                    state[6 + 2 * j] = np.array(angle)
                else:
                    state[5 + 2 * j] = 2
                    state[6 + 2 * j] = 0
            if np.any(self.done == 1):
                state[4] = 1
            else:
                state[4] = 0
            self.state[:, i] = state

    def update_feature(self):
        obstacle_closest = np.zeros((2, self.num_agent))
        for i in range(self.num_agent):
            # the index of nearest obstacle
            temp = np.argmin(np.linalg.norm(self.obstacle_total - self.agent_position[:, i:i + 1], axis=0))
            # the position of the nearest obstacle
            obstacle_closest[:, i:i + 1] = self.obstacle_total[:, temp:temp + 1]
        self.obstacle_closest = obstacle_closest

        virtual_obstacle = np.zeros((2, 0))  # virtual obstacle buffer
        for i in range(self.num_agent):
            if self.done[0, i]:  # if any pursuer is inactive
                virtual_obstacle = np.hstack(
                    (virtual_obstacle, self.agent_position[:, i:i + 1] + np.array([[1.], [1]])))
        # add virtual obstacles into obstacles list
        self.obstacle_with_other_agent = np.hstack((self.obstacle_total, virtual_obstacle))

        obstacle_closest_with_other_agent = np.zeros((2, self.num_agent))
        for i in range(self.num_agent):
            # the index of nearest obstacle (considering virtual obstacles)
            temp = np.argmin(np.linalg.norm(self.obstacle_with_other_agent - self.agent_position[:, i:i + 1], axis=0))
            # the position of nearest obstacle (considering virtual obstacles)
            obstacle_closest_with_other_agent[:, i:i + 1] = self.obstacle_with_other_agent[:, temp:temp + 1]
        self.obstacle_closest_with_other_agent = obstacle_closest_with_other_agent

        self.distance_from_target = np.linalg.norm(self.agent_position - self.target_position, axis=0,
                                                   keepdims=True)  # the distance between the evader and pursuers

    def update_feature_last(self):
        self.agent_position_last = copy.deepcopy(self.agent_position)
        self.agent_orientation_last = copy.deepcopy(self.agent_orientation)
        self.distance_from_target_last = copy.deepcopy(self.distance_from_target)

    def from_action_to_APF(self, action, attention_score_array):
        scale_repulse = np.zeros((1, self.num_agent))  # eta buffer
        individual_balance = np.zeros((1, self.num_agent))  # lambda buffer
        for i in range(self.num_agent):  # transform the action indexes into parameter pairs
            if action[0, i] < 8:
                scale_repulse[0, i] = 1e6
            elif action[0, i] < 16:
                scale_repulse[0, i] = 1e7
            else:
                scale_repulse[0, i] = 1e8
            if action[0, i] % 8 == 0:
                individual_balance[0, i] = 4000 / 7 * 0
            elif action[0, i] % 8 == 1:
                individual_balance[0, i] = 4000 / 7 * 1
            elif action[0, i] % 8 == 2:
                individual_balance[0, i] = 4000 / 7 * 2
            elif action[0, i] % 8 == 3:
                individual_balance[0, i] = 4000 / 7 * 3
            elif action[0, i] % 8 == 4:
                individual_balance[0, i] = 4000 / 7 * 4
            elif action[0, i] % 8 == 5:
                individual_balance[0, i] = 4000 / 7 * 5
            elif action[0, i] % 8 == 6:
                individual_balance[0, i] = 4000 / 7 * 6
            elif action[0, i] % 8 == 7:
                individual_balance[0, i] = 4000 / 7 * 7
        # calculate APF resultant forces
        F, wall_following = APF_function_for_DQN.total_decision(self.agent_position,
                                                                self.agent_orientation,
                                                                self.obstacle_closest_with_other_agent,
                                                                self.target_position,
                                                                scale_repulse,
                                                                individual_balance,
                                                                self.r_perception, attention_score_array)
        return F, wall_following
