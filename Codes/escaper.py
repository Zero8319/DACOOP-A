import numpy as np
import APF_function_for_DQN

v_max = 600  # the evader's speed
v_agent = 300  # the pursuers' speed
rcd = 200  # d_c
p_thres = 0.2  # the zigzag threshold
sensitivity_range = 2000  # the sensitivity range
r_zigzag = 300
influence_range = 800  # the influence range of obstacles
scale_repulse = 1e7  # eta
slip_range = 500  # the slip range used in external rules


def escaper(agent_position, target_position, target_orientation, obstacle_total, num_agent, zigzag_count, zigzag_last,
            last_e, slip_flag):
    # the distance from all pursuers
    distance_from_agent = np.linalg.norm(target_position - agent_position, axis=0, keepdims=True)
    # the distance from all obstacles
    distance_from_obstacle = np.linalg.norm(target_position - obstacle_total, axis=0, keepdims=True)
    # which pursuers is in the sensitivity range
    agent_in_sensitivity_range = distance_from_agent < sensitivity_range
    # which pursuers is in the slip range
    agent_in_slip_range = distance_from_agent < slip_range
    ########## F_escape or F_zigzag #############
    distance_from_nearest_agent = np.min(distance_from_agent)  # the distance from the nearest pursuer
    distance_from_nearest_obstacle = np.min(distance_from_obstacle)  # the distance from the nearest obstacle

    if distance_from_nearest_agent > sensitivity_range:
        p_panic = 0  # the degree of panic
    else:
        p_panic = 1 / (np.exp(1) - 1) * (np.exp(-distance_from_nearest_agent / sensitivity_range + 1) - 1)

    if (p_panic < p_thres) and (distance_from_nearest_obstacle > r_zigzag):
        # if the degree of panic exceeds threshold and there is no obstacle nearby
        zigzag_flag = 1
        if zigzag_count > np.random.randint(10, 15):
            zigzag_count = 0
        if zigzag_count == 0:
            # if the evader didn't zigzag at the last timestep, choose an orientation randomly
            while True:
                temp = np.random.random() * 2 * np.pi - np.pi
                F_escape = np.array([[np.cos(temp)],
                                     [np.sin(temp)]])
                if np.dot(np.ravel(F_escape), np.ravel(target_orientation)) > 0:
                    zigzag_last = F_escape
                    break
            zigzag_count += 1
        else:
            # if the evader zigzagged at the last timestep, keep moving along the orientation
            F_escape = zigzag_last
            zigzag_count += 1
    else:
        # if the evader do not zigzag, calculate F_escape
        zigzag_flag = 0
        zigzag_count = 0
        F_escape = np.full((2, num_agent), np.nan)

        for i in range(num_agent):
            if np.ravel(agent_in_sensitivity_range)[i]:
                F_escape[:, i:i + 1] = (target_position - agent_position)[:, i:i + 1] / np.linalg.norm(
                    distance_from_agent[:, i]) ** 2

        if np.all(np.isnan(F_escape)):
            F_escape = np.zeros((2, 1))
        else:
            F_escape = np.nanmean(F_escape, axis=1, keepdims=True) / np.linalg.norm(np.nanmean(F_escape, axis=1))
    ########### F_r ############
    # the nearest obstacle
    obstacle_sort_index = np.argsort(distance_from_obstacle)
    obstacle_closest = obstacle_total[:, np.ravel(obstacle_sort_index)[0:10]]
    # calculate the repulsive force
    F_repulse = np.zeros((2, 10))
    for i in range(10):
        temp = scale_repulse * (1 / (
                np.linalg.norm(target_position - obstacle_closest[:, i:i + 1]) - 100) - 1 / influence_range) / (
                       np.linalg.norm(target_position - obstacle_closest[:, i:i + 1]) - 100) ** 2 * (
                       target_position - obstacle_closest[:, i:i + 1]) / np.linalg.norm(
            target_position - obstacle_closest[:, i:i + 1])

        if np.linalg.norm(target_position - obstacle_closest[:, i:i + 1]) < influence_range:
            # if the pursuer is within the obstacle's influence range
            F_repulse[:, i:i + 1] = temp
        else:
            F_repulse[:, i:i + 1] = np.array([[0], [0]])
    F_repulse = np.mean(F_repulse, axis=1, keepdims=True)
    ############# resultance force ############
    vector1 = np.ravel(F_escape + F_repulse)  # F_ar
    vector2 = np.ravel(F_escape)  # F_a

    if np.dot(vector1, vector2) < 0:
        # if the angle between F_ar and F_a exceeds 90 degree, move according to wall following rules
        F_temp = APF_function_for_DQN.wall_follow_for_escaper(F_repulse, target_orientation,
                                                              distance_from_nearest_agent, F_escape,
                                                              distance_from_nearest_obstacle)
        F_total = F_temp / np.linalg.norm(np.ravel(F_temp))  # normalization
        wall_following = 1
    else:
        #  else, calculate the resultant force
        F_temp = F_repulse + F_escape
        F_total = F_temp / np.linalg.norm(np.ravel(F_temp))  # normalization
        wall_following = 0

    ############ slip ############
    if np.sum(agent_in_slip_range) > 1:  # if there are more than one pursuer in the slip range
        # calculate the positions of two nearest pursuers
        agent_closest_index = np.argsort(distance_from_agent)
        agent_closest1 = agent_position[:, agent_closest_index[0, 0]].reshape(2, -1)
        agent_closest2 = agent_position[:, agent_closest_index[0, 1]].reshape(2, -1)
        # calculate the slip orientation according to <Group chasing tactics: how to catch a faster prey>
        e_temp = (agent_closest1 - target_position) + (agent_closest2 - target_position)
        e = e_temp / np.linalg.norm(np.ravel(e_temp))
        rp1 = e * np.dot(np.ravel(e), np.ravel(agent_closest1 - target_position))
        rp2 = e * np.dot(np.ravel(e), np.ravel(agent_closest2 - target_position))
        taue1 = np.linalg.norm(np.ravel(rp1)) / v_max
        taue2 = np.linalg.norm(np.ravel(rp2)) / v_max
        tauc1 = (np.linalg.norm(np.ravel(rp1 - agent_closest1 + target_position)) - rcd) / v_agent
        tauc2 = (np.linalg.norm(np.ravel(rp2 - agent_closest2 + target_position)) - rcd) / v_agent
        if slip_flag == 0:
            if taue1 < tauc1 and taue2 < tauc2 and distance_from_nearest_obstacle < 300 and np.dot(np.ravel(F_repulse),
                                                                                                   np.ravel(e)) >= 0:
                F_total = e
                last_e = e
                slip_flag = 1
            else:
                slip_flag = 0
        else:
            if taue1 < tauc1 and taue2 < tauc2 and distance_from_nearest_obstacle > 100 and np.dot(np.ravel(e),
                                                                                                   np.ravel(
                                                                                                       last_e)) >= 0:
                F_total = e
                slip_flag = 1
            elif taue1 < tauc1 and taue2 < tauc2 and distance_from_nearest_obstacle > 100 and np.dot(np.ravel(e),
                                                                                                     np.ravel(
                                                                                                         last_e)) < 0:
                F_total = -e
                slip_flag = 1
            else:
                slip_flag = 0

    F_total = F_total * v_max  # calculate the displacement of evader
    return F_total, zigzag_count, zigzag_last, zigzag_flag, wall_following, slip_flag, distance_from_nearest_obstacle, last_e
