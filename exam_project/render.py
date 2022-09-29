from __future__ import annotations

import sys
sys.path.insert(1, '/home/marco/probabilistic_machine_learning/exam_project/')
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

low_boundaries = [0, 0]
high_boundaries = [1,1]

N = 1
lr_pars = {'gamma': .925, 'LQ': 9, 'epsilon': 1, 'exploration': 0.1, 'RBF_length_scale': 0.05, 'delta':0.9999, 'kernel': 'RationalQuadratic2.3', 'start_update': 1e7}
current_directory = os.getcwd()
name_instance_directory = str(lr_pars).replace("'", '').replace(" ", '').replace("{", '').replace("}", '').replace(",", '_').replace(":", '_').replace(".", '_')
time = np.zeros(5001)
instance = 5
episode = int(input('insert episode to study: '))
saved_episodes_dict = np.load(current_directory + '/trials/episode/{}/renderepisodes_{}_{}.npy'.format(name_instance_directory, N, instance), allow_pickle=True).item()
episode_dict = saved_episodes_dict[episode]

duration = episode_dict['length']
time = np.arange(duration)
trackers_positions_history = episode_dict['tracker_trajectory']
target_positions_history = episode_dict['target_trajectory']

X_trackers_coords = {n: np.array( [pos[0] for pos in trackers_positions_history[n]] ) for n in range(N)}
Y_trackers_coords = {n: np.array( [pos[1] for pos in trackers_positions_history[n]] ) for n in range(N)}
X_target_coords = np.array( [pos[0] for pos in target_positions_history] )
Y_target_coords = np.array( [pos[1] for pos in target_positions_history] )

def motion(t):

    ax.clear()

    ax.set_xlim([low_boundaries[0], high_boundaries[0]])
    ax.set_ylim([low_boundaries[1], high_boundaries[1]])

    for n in range(N):
        detection_zone = plt.Circle((X_trackers_coords[n][t], Y_trackers_coords[n][t]), radius=Tracker.DETECTION_RANGE, alpha=0.1, color='k')
        catching_zone = plt.Circle((X_trackers_coords[n][t], Y_trackers_coords[n][t]), radius=3, fill=False, linestyle='--', color='k')
        tracker_path_x = X_trackers_coords[n][:t+1]
        tracker_trail_x = tracker_path_x[-7:]
        tracker_path_y = Y_trackers_coords[n][:t+1]
        tracker_trail_y = tracker_path_y[-7:]
        target_path_x = X_target_coords[:t+1]
        target_trail_x = target_path_x[-7:]
        target_path_y = Y_target_coords[:t+1]
        target_trail_y = target_path_y[-7:]
        ax.plot( tracker_trail_x, tracker_trail_y, color= 'red', linestyle='--', alpha=0.33 )
        marker = matplotlib.markers.MarkerStyle('o')
        ax.scatter(X_trackers_coords[n][t], Y_trackers_coords[n][t], color='red', marker=marker, s=200, linewidth=3)

    ax.text(X_target_coords[t]+0.025, Y_target_coords[t]+0.025, 'G', color='k',  ha='center', va='center', fontsize=30)
    ax.set_title('Episode {} \nStep = '.format(episode) + str(time[t]))
    return ax

fig, ax = plt.subplots(figsize=(20,20))
plt.gca().set_aspect("equal", adjustable="box")
line_ani = animation.FuncAnimation( fig, motion, interval=10, frames=int(duration), repeat=False )

f = '/home/marco/probabilistic_machine_learning/exam_project/trials/episode/{}/episode_{}_{}.gif'.format(name_instance_directory, episode, instance)
writergif = animation.PillowWriter(fps=4)
line_ani.save(f, writer=writergif)
del line_ani
