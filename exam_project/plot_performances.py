from __future__ import annotations

import sys
sys.path.insert(1, '/home/marco/active_object_tracking_modelling/deep_rl_code/')
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import threading
import random
from numpy.random import default_rng
rng = default_rng()
from Utils import *
import math
from matplotlib import animation
from matplotlib.animation import PillowWriter 
from Agents import *
from WorldFeatures import *
import itertools
from numpy import linalg as alg
import tensorflow as T
import os
from matplotlib import animation
from matplotlib.animation import PillowWriter 
import os.path

episode = input("Enter episode: ")
agents = [0, 1]

lr_pars = {'a': 0.025, 'expa': 0.9, 'eps': 0.1, 'expe': .9, 'p_angle': 12}

reward = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/performances/rewards_up_to_{}_double_{}.npy'.format(episode, lr_pars), allow_pickle=True).item()
time = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/performances/time_up_to_{}_double_{}.npy'.format(episode, lr_pars), allow_pickle=True)

fig, ax = plt.subplots(figsize=(10,10))
plt.suptitle('duration')
ax.set_xlabel('episode')
ax.set_ylabel('time')

average_time = moving_average(time, mode='valid', periods=500)
ax.plot(range(len(average_time)), average_time, label='average duration')
ax.legend()

for i in agents:
    reward_i = reward[i]
    reward_i = reward_i[:int(episode)]

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    plt.suptitle('Agent {}'.format(i))

    ylabels=['episode reward']

    ax.set_xlabel('episode')
    ax.set_ylabel('episode reward')

    average_reward_i = moving_average(reward_i, mode='valid', periods=500)

    ax.plot(range(len(average_reward_i)), average_reward_i, label='average reword')

    ax.legend()
    plt.show()