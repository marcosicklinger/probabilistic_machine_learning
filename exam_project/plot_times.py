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

# episode = input("Enter episode: ")
episode_1 = 99999
episode_2 = 49999
n_trials = 7
n_agents = 2

time = []

# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_single.npy'.format(episode_2), allow_pickle=True)]

lr_pars_1 = {'a': 0.001, 'expa': 0.999, 'eps': 0.01, 'expe': 1, 'p_angle': 8}
time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_singleNoTDist_{}.npy'.format(episode_2, lr_pars_1), allow_pickle=True)]
# lr_pars_2 = {'a': 0.005, 'expa': 0.9, 'eps': 0.1, 'expe': 1.05, 'p_angle': 8}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_double_{}.npy'.format(episode_2, lr_pars_2), allow_pickle=True)]
# lr_pars_3 = {'a': 0.002, 'expa': 0.8, 'eps': 0.05, 'expe': 0.8, 'p_angle': 8}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_double_{}.npy'.format(episode_2, lr_pars_3), allow_pickle=True)]
# lr_pars_4 = {'a': 0.0075, 'expa': 0.9999, 'eps': 0.05, 'expe': 1.25, 'p_angle': 12}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDist_{}.npy'.format(episode_2, lr_pars_4), allow_pickle=True)]
# lr_pars_5 = {'a': 0.001, 'expa': 0.99, 'eps': 0.025, 'expe': 1, 'p_angle': 12}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDist_{}.npy'.format(episode_2, lr_pars_5), allow_pickle=True)]
# lr_pars_6 = {'a': 0.0015, 'expa': 0.99, 'eps': 0.03, 'expe': 1, 'p_angle': 16}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDist_{}.npy'.format(episode_2, lr_pars_6), allow_pickle=True)]
# lr_pars_7 = {'a': 0.002, 'expa': 0.97, 'eps': 0.033, 'expe': 0.99, 'p_angle': 20}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDist_{}.npy'.format(episode_2, lr_pars_7), allow_pickle=True)]
# lr_pars_8 = {'a': 0.01, 'expa': 0.99, 'eps': 0.05, 'expe': 1.25, 'p_angle': 12}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_singleNoTDist_{}.npy'.format(episode_2, lr_pars_8), allow_pickle=True)]
# lr_pars_9 = {'a': 0.005, 'expa': 0.99, 'eps': 0.05, 'expe': 1.15, 'p_angle': 16}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_singleNoTDist_{}.npy'.format(episode_2, lr_pars_9), allow_pickle=True)]
# lr_pars_10 = {'a': 0.003, 'expa': 0.999, 'eps': 0.025, 'expe': 1.05, 'p_angle': 12}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDistMDP_{}.npy'.format(episode_2, lr_pars_10), allow_pickle=True)]

periods = 500

fig, ax = plt.subplots(figsize=(10,10))
plt.suptitle('Duration comparison (moving average over {} episodes)'.format(periods))
ax.set_xlabel('episode')
ax.set_ylabel('time')

i = 0
time_array = np.array(time[i])
long_episodes = time_array[time_array > 200]
print('number of long episodes: {}'.format(len(long_episodes)))
average_time = moving_average(time[i], periods=periods, mode='valid')
ax.plot(range(len(average_time)), average_time, label='{} agents'.format(n_agents))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(2, lr_pars_2))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(2, lr_pars_3))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(2, lr_pars_4))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agents, angle aggregation {}'.format(2, lr_pars_5['p_angle']))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agents, angle aggregation {}'.format(2, lr_pars_6['p_angle']))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agent, angle aggregation {}'.format(2, lr_pars_7['p_angle']))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agent, angle aggregation {}'.format(1, lr_pars_8['p_angle']))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agent, angle aggregation {}'.format(1, lr_pars_9['p_angle']))
# i += 1
# average_time = moving_average(time[i], periods=periods, mode='valid')
# ax.plot(range(len(average_time)), average_time, label='{} agents, angle aggregation {}, MDP'.format(1, lr_pars_10['p_angle']))

ax.legend()
plt.show()

