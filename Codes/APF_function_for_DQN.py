import numpy as np


def attract(self_position, target_position):
    F = (target_position - self_position) / np.linalg.norm(target_position - self_position)
    return F


def repulse(self_position, obstacle_closest, influence_range, scale_repulse):
    F = scale_repulse * (1 / (np.linalg.norm(self_position - obstacle_closest) - 100) - 1 / influence_range) / (
            np.linalg.norm(self_position - obstacle_closest) - 100) ** 2 * (
                self_position - obstacle_closest) / np.linalg.norm(self_position - obstacle_closest)
    if np.linalg.norm(self_position - obstacle_closest) < influence_range:
        return F
    else:
        return np.zeros((2, 1))


def individual(self_position, friend_position, individual_balance, r_perception, attention_score):
    F = np.zeros((2, 0))
    num_neighbor = 0
    for i in range(friend_position.shape[1]):
        if np.linalg.norm(friend_position[:, i:i + 1] - self_position) < r_perception:
            num_neighbor += 1
    for i in range(friend_position.shape[1]):
        temp = (friend_position[:, i:i + 1] - self_position) / np.linalg.norm(
            friend_position[:, i:i + 1] - self_position) * (0.5 - individual_balance / (
                np.linalg.norm(friend_position[:, i:i + 1] - self_position) - 200))
        if np.any(attention_score) and num_neighbor > 1:
            temp = temp * attention_score[0, i] * num_neighbor
        if np.linalg.norm(friend_position[:, i:i + 1] - self_position) < r_perception:
            F = np.hstack((F, temp))
    if F.size == 0:
        F = np.zeros((2, 1))
    return np.mean(F, axis=1, keepdims=True)


def generate_boundary(point1, point2, point3, point4, step):
    temp1 = np.ravel(np.arange(point1[0], point2[0] + 1, step))
    temp2 = np.ravel(np.arange(point2[1], point3[1] + 1, step))
    boundary12 = np.vstack((temp1, np.ones_like(temp1) * point1[1]))
    boundary23 = np.vstack((np.ones_like(temp2) * point2[0], temp2))
    boundary34 = np.vstack((np.flipud(temp1), np.ones_like(temp1) * point3[1]))
    boundary41 = np.vstack((np.ones_like(temp2) * point4[0], np.flipud(temp2)))

    boundary = np.hstack((boundary12, boundary23, boundary34, boundary41))
    return boundary


def wall_follow(self_orientation, F_repulse, F_individual):
    # calculate n_1 and n_2
    self_orientation = np.ravel(self_orientation)
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1
    # choose between n_1 and n_2
    temp1 = np.linalg.norm(np.ravel(rotate_vector1) - self_orientation)
    temp2 = np.linalg.norm(np.ravel(rotate_vector2) - self_orientation)
    if np.linalg.norm(F_individual) < 1:  # if inter-individual force is less threshold B
        if temp1 > temp2:  # choose according to the heading
            return rotate_vector2
        else:
            return rotate_vector1
    else:  # if inter-individual force exceeds threshold B,choose according to the inter-individual force
        if np.dot(np.ravel(rotate_vector1), np.ravel(F_individual)) > 0:  #
            return rotate_vector1
        else:
            return rotate_vector2


def wall_follow_for_escaper(F_repulse, target_orientation, distance_from_nearest_agent, F_escape,
                            distance_from_nearest_obstacle):
    # calculate n_1 and n_2
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1
    # choose between n_1 and n_2
    if np.dot(np.ravel(target_orientation), np.ravel(rotate_vector1)) > 0:
        # if n_1 forms a smaller angle with the evader's heading
        final = rotate_vector1
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector2)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            # if any pursuer is close and it is in the front of evader, choose n_2
            final = rotate_vector2
    else:
        # if n_2 forms a smaller angle with the evader's heading
        final = rotate_vector2
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector1)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            # if any pursuer is close and it is in the front of evader, choose n_1
            final = rotate_vector1
    if distance_from_nearest_obstacle < 150:
        # if the evader is too close to obstacles, add another force to avoid collisions
        final = final + F_repulse
    return final


def APF_decision(self_position, friend_position, target_position, obstacle_closest, scale_repulse, individual_balance,
                 r_perception, attention_score):
    influence_range = 800  # the influence range of obstacles
    F_attract = attract(self_position, target_position)  # calculate the attractive force
    # calculate the repulsive force
    F_repulse = repulse(self_position, obstacle_closest, influence_range, scale_repulse)
    # calculate the inter-individual force
    F_individual = individual(self_position, friend_position, individual_balance, r_perception, attention_score)
    # calculate the resultant force
    F = F_attract + F_repulse + F_individual
    return F_attract, F_repulse, F_individual, F


def total_decision(agent_position, agent_orientation, obstacle_closest, target_position, scale_repulse,
                   individual_balance, r_perception, attention_score_array):
    F = np.zeros((2, 0))  # resultant force buffer
    wall_following = np.zeros((1, 0))  # flag of whether pursuers move according to the wall following rules
    for i in range(scale_repulse.size):
        self_position = agent_position[:, i:i + 1]
        friend_position = np.delete(agent_position, i, axis=1)
        self_orientation = agent_orientation[:, i:i + 1]
        # calculate APF forces
        F_attract, F_repulse, F_individual, F_total = APF_decision(self_position, friend_position, target_position,
                                                                   obstacle_closest[:, i:i + 1],
                                                                   scale_repulse[0, i],
                                                                   individual_balance[0, i], r_perception,
                                                                   attention_score_array[i:i + 1, :])
        vector1 = np.ravel(F_attract + F_repulse)
        vector2 = np.ravel(F_attract)

        if np.dot(vector1, vector2) < 0:  # if the angle between F_ar and F_a exceeds 90 degree
            # move according to wall following rules
            F_total = wall_follow(self_orientation, F_repulse, F_individual)
            wall_following = np.hstack((wall_following, np.array(True).reshape(1, 1)))
        else:
            wall_following = np.hstack((wall_following, np.array(False).reshape(1, 1)))
        F_total = F_total / np.linalg.norm(F_total)  # normalize the resultant force
        F = np.hstack((F, F_total))
    return F, wall_following
