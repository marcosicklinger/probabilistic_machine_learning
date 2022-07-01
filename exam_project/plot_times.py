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

episode_1 = 99999
episode_2 = 199999

time = []
time += [np.load('/home/marco/active_object_tracking_modelling/constvel_trials/performances/time_up_to_{}_single.npy'.format(episode_1), allow_pickle=True)]
time += [np.load('/home/marco/active_object_tracking_modelling/constvel_trials/performances/time_up_to_{}_double.npy'.format(episode_1), allow_pickle=True)]
lr_pars = {'a': 0.025, 'expa': 0.9, 'eps': 0.1, 'expe': 0.9}
time += [np.load('/home/marco/active_object_tracking_modelling/constvel_trials/performances/time_up_to_{}_double_{}.npy'.format(episode_1, lr_pars), allow_pickle=True)]
lr_pars = {'a': 0.035, 'expa': 0.9, 'eps': 0.1, 'expe': 1.05, 'p_angle': 12}
time += [np.load('/home/marco/active_object_tracking_modelling/constvel_trials/performances/time_up_to_{}_double_{}.npy'.format(episode_2, lr_pars), allow_pickle=True)]

periods = 500

fig, ax = plt.subplots(figsize=(10,10))
plt.suptitle('Duration comparison (moving average over {} episodes)'.format(periods))
ax.set_xlabel('episode')
ax.set_ylabel('time')

average_time = moving_average(time[0], periods=periods, mode='valid')
ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(1, '(precision 8)'))
average_time = moving_average(time[1], periods=periods, mode='valid')
ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(2, '(precision 8)'))
average_time = moving_average(time[2], periods=periods, mode='valid')
ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(2, '(precision 16)'))
average_time = moving_average(time[3], periods=periods, mode='valid')
ax.plot(range(len(average_time)), average_time, label='{} agents {}'.format(2, '(precision 12)'))

ax.legend()
plt.show()
