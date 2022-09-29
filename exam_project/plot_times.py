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
import os
from matplotlib import animation
from matplotlib.animation import PillowWriter 
import os.path

n_agents = 1
periods = 150


fig, ax = plt.subplots(figsize=(10,10))
plt.suptitle('Duration comparison (moving average over {} episodes)'.format(periods))
ax.set_xlabel('episode')
ax.set_ylabel('time')

lr_pars_discrete = {'gamma': .9, 'alpha_0': .01, 'alpha': .01, 'alpha_min': 1e-7, 'exp_alpha': 0.9999, 'eps_0': .05, 'eps': .05, 'eps_min': 1e-6, 'exp_eps': 1.5, 'start_update': 5e4, 'cut': 1e50}
current_directory = os.getcwd()
name_instance_directory = str(lr_pars_discrete).replace("'", '').replace(" ", '').replace("{", '').replace("}", '').replace(",", '_').replace(":", '_').replace(".", '_')
time_discrete = np.zeros(5001)
instances = [5]
for i in instances:
    time_discrete += np.load(current_directory + '/trials/performance/{}/time_up_to_{}_{}_{}_discrete.npy'.format(name_instance_directory,5000, n_agents, i), allow_pickle=True)
time_discrete = time_discrete/len(instances)
average_time_discrete = moving_average(time_discrete, periods=periods, mode='valid')
ax.plot(range(len(average_time_discrete)), average_time_discrete, label=r'$\epsilon$-greedy tabular')

lr_pars = {'gamma': .925, 'LQ': 9, 'epsilon': 1, 'exploration': 0.1, 'RBF_length_scale': 0.05, 'delta':0.9999, 'kernel': 'RationalQuadratic2.3', 'start_update': 1e7}
current_directory = os.getcwd()
name_instance_directory = str(lr_pars).replace("'", '').replace(" ", '').replace("{", '').replace("}", '').replace(",", '_').replace(":", '_').replace(".", '_')
time1 = np.zeros(5001)
instances = [13]
for i in instances:
    time1 += np.load(current_directory + '/trials/performance/{}/time_up_to_{}_{}_{}.npy'.format(name_instance_directory,5000, n_agents, i), allow_pickle=True)
time1 = time1/len(instances)
average_time1 = moving_average(time1, periods=periods, mode='valid')
ax.plot(range(len(average_time1)), average_time1, label=r'rational quadratic ($\alpha$=2.3, l=1.0)')

lr_pars = {'gamma': .925, 'LQ': 9, 'epsilon': 1, 'exploration': 0.1, 'RBF_length_scale': 0.04, 'delta':0.9999, 'kernel': 'RationalQuadratic2.4', 'start_update': 1e7}
current_directory = os.getcwd()
name_instance_directory = str(lr_pars).replace("'", '').replace(" ", '').replace("{", '').replace("}", '').replace(",", '_').replace(":", '_').replace(".", '_') 
time2 = np.zeros(5001)
instances = [12]
for i in instances:
    time2 += np.load(current_directory + '/trials/performance/{}/time_up_to_{}_{}_{}.npy'.format(name_instance_directory, 5000, n_agents, i), allow_pickle=True)
time2 = time2/len(instances)
average_time2 = moving_average(time2, periods=periods, mode='valid')
ax.plot(range(len(average_time2)), average_time2, label=r'RBF (l=0.05)')

ax.legend()
plt.show()

