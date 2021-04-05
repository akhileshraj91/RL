import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tqdm
import time


class arm:
    def __init__(self, num, mu, sig):
        self.id = num
        self.mean = mu
        self.sigma = sig
        self.rew = deque([])
        self.count = 0
        self.val = 0

    def initialize(self):
        self.rew = deque([])
        self.count = 0
        self.val = 0


def reward(action):
    mu, sig = action.mean, action.sigma
    reward = np.random.normal(mu, sig, 1)
    return reward


def check_val():
    value = np.zeros([num_of_bandits, 1])
    for i in range(num_of_bandits):
        value[i] = band[i].val
    return value


def execute_main(epsilon, c_val, oa, it):
    # total_reward = 0
    # for rob in band:
    #     rob.initialize()
    percentage_optimal_action = np.zeros([epochs, 1])
    average_reward = np.zeros([iterations, 1])
    average_total_reward = np.zeros([iterations + 1, 1])
    average_reward_per_step = np.zeros([iterations, 1])
    for epo in range(epochs):
        print(int((it + 1) / 3 * epo / epochs * 100), "%")
        for rob in band:
            rob.initialize()
        total_reward = np.zeros([iterations + 1, 1])
        reward_per_step = np.zeros([iterations, 1])
        R = np.zeros([iterations, 1])
        for iter in range(iterations):
            value_list = check_val()
            # print(value_list)
            # time.sleep(1)
            prob = np.random.random()
            if prob < epsilon:
                A = np.random.randint(0, 10)
                # print(prob,epsilon,A)
            else:
                A = np.argmax(value_list)
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",epsilon)
            action = band[A]
            R[iter] = reward(action)
            # plt.figure(1)
            # plt.scatter(A, R, s=2, color='k')
            action.rew.append(R[iter])
            action.count += 1
            alpha=0.1
            action.val = action.val + alpha * (R[iter] - action.val)
            # print(action.val)
            total_reward[iter + 1] += total_reward[iter] + R[iter]
            reward_per_step[iter] = total_reward[iter + 1] / (iter + 1)
        total = 0
        for act in band:
            total += act.count
        percentage_optimal_action[epo] = (band[oa].count / total) * 100
        average_reward += R
        average_total_reward += total_reward
        average_reward_per_step += reward_per_step
    average_reward = average_reward / epochs
    average_total_reward = average_total_reward / epochs
    average_reward_per_step = average_reward_per_step / epochs

    plt.figure(1)
    plt.plot(average_reward, color=c_val, label='eps = % r' % round(epsilon, 2))
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(average_total_reward, color=c_val, label='eps = % r' % round(epsilon, 2))
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(average_reward_per_step, color=c_val, label='eps = % r' % round(epsilon, 2))
    plt.legend()
    plt.show()

    plt.figure(4)
    plt.scatter(range(len(percentage_optimal_action)),percentage_optimal_action, color=c_val, label='eps = % r' % round(epsilon, 2))
    plt.legend()
    plt.show()


# if __name__ == "main":
num_of_bandits = 10
mu_q, sigma_q = 0, 1
s = np.random.normal(mu_q, sigma_q, num_of_bandits)
optimal_action = np.argmax(s)
band = deque([])
for i in range(num_of_bandits):
    q_star = s[i]
    mu_r, sigma_r = q_star, 1
    band.append(arm(i, mu_r, sigma_r))
    plt.figure(0)
    plt.scatter(i, q_star)
plt.xlabel('Bandit arm')
plt.show()
iterations = 1000

reward_per_step = np.zeros([iterations, 1])
epsi_set = [0, 0.01, 0.1]
color = ['k', 'g', 'r']
epochs = 2000
for i in range(3):
    execute_main(epsi_set[i], color[i], optimal_action, i)
