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

time = []
# n_trial = 3

lr_pars_1 = {'a': 0.005, 'expa': 0.9999, 'eps': 0.005, 'expe': 1, 'p_angle': 8}
time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDist_{}.npy'.format(episode_2, lr_pars_1), allow_pickle=True)]
# lr_pars_2 = {'a': 0.001, 'expa': 0.99, 'eps': 0.025, 'expe': 1, 'p_angle': 12}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/ave_time_{}_doubleNoTDist_{}.npy'.format(n_trial, lr_pars_2), allow_pickle=True)]
# lr_pars_3 = {'a': 0.0015, 'expa': 0.99, 'eps': 0.03, 'expe': 1, 'p_angle': 16}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/ave_time_{}_doubleNoTDist_{}.npy'.format(n_trial, lr_pars_3), allow_pickle=True)]
# lr_pars_4 = {'a': 0.002, 'expa': 0.97, 'eps': 0.033, 'expe': 0.99, 'p_angle': 16}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/ave_time_{}_doubleNoTDist_{}.npy'.format(n_trial, lr_pars_4), allow_pickle=True)]
# lr_pars_MDP = {'a': 0.003, 'expa': 0.999, 'eps': 0.025, 'expe': 1.05, 'p_angle': 12}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_doubleNoTDistMDP_{}.npy'.format(episode_2, lr_pars_MDP), allow_pickle=True)]
# lr_pars_4 = {'a': 0.0125, 'expa': 0.9, 'eps': 0.1, 'expe': 1, 'p_angle': 16}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_double_{}.npy'.format(episode_2, lr_pars_4), allow_pickle=True)]
# lr_pars_5 = {'a': 0.005, 'expa': 0.9, 'eps': 0.1, 'expe': 1, 'p_angle': 16}
# time += [np.load('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_double_{}.npy'.format(episode_2, lr_pars_5), allow_pickle=True)]

n_bins = 50

fig, ax = plt.subplots(figsize=(10,10))

ax.set_xlabel('episode duration')
ax.set_title('episode duration distribution')

hist_range = (0, 800)

ax.hist(time[0][-5000:], n_bins, range=hist_range, histtype='step', label='{} agents'.format(2))
# ax.hist(time[1][-5000:], n_bins, range=hist_range, histtype='step', label='{} agents {}'.format(2, lr_pars_2['p_angle']))
# ax.hist(time[2][-5000:], n_bins, range=hist_range, histtype='step', label='{} agents {}'.format(2, lr_pars_3['p_angle']))
# ax.hist(time[3][-5000:], n_bins, range=hist_range, histtype='step', label='{} agents {}'.format(2, lr_pars_4['p_angle']))
# ax.hist(time[4][-5000:], n_bins, range=hist_range, histtype='step', label='{} agents {} (MDP)'.format(1, lr_pars_MDP['p_angle']))
# ax.hist(time[3][-25000:], n_bins, histtype='step', label='{} agents {}'.format(2, lr_pars_4))
# ax.hist(time[4][-25000:], n_bins, histtype='step', label='{} agents {}'.format(2, lr_pars_5))

ax.legend()
plt.show()