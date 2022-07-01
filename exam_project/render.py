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

episode = input('Enter episode to render:\t')
low_boundaries = [0, 0]
high_boundaries = [100,100]

N = 2
lr_pars = {'a': 0.025, 'expa': 0.9, 'eps': 0.1, 'expe': 1.05, 'p_angle': 12}


duration = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/duration_double_{}_{}.npy'.format(lr_pars, episode))
trackers_positions_history = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_positions_history_double_{}_{}.npy'.format(lr_pars, episode))
trackers_orientation_history = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_orientation_history_double_{}_{}.npy'.format(lr_pars, episode))
trackers_velocities_history = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_velocities_history_double_{}_{}.npy'.format(lr_pars, episode))
trackers_target_observation_history = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_target_observation_history_double_{}_{}.npy'.format(lr_pars, episode))
target_positions_history = np.load('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/target_positions_history_double_{}_{}.npy'.format(lr_pars, episode))

time = np.arange(duration)

X_trackers_coords = {n: np.array( [pos[0] for pos in trackers_positions_history[n]] ) for n in range(N)}
Y_trackers_coords = {n: np.array( [pos[1] for pos in trackers_positions_history[n]] ) for n in range(N)}
THETA_trackers = {n: np.array( [theta for theta in trackers_orientation_history[n]] ) for n in range(N)}
Vx_trackers_magnitude = {n: np.array( [vel[0] for vel in trackers_velocities_history[n]] ) for n in range(N)}
Vy_trackers_magnitude = {n: np.array( [vel[1] for vel in trackers_velocities_history[n]] ) for n in range(N)}
target_observation = {n: np.array( [obs for obs in trackers_target_observation_history[n]] ) for n in range(N)}


X_target_coords = np.array( [pos[0] for pos in target_positions_history] )
Y_target_coords = np.array( [pos[1] for pos in target_positions_history] )


def motion(t):

    ax.clear()

    ax.set_xlim([low_boundaries[0], high_boundaries[0]])
    ax.set_ylim([low_boundaries[1], high_boundaries[1]])

    for n in range(N):
        detection_zone = plt.Circle((X_trackers_coords[n][t], Y_trackers_coords[n][t]), radius=Tracker.DETECTION_RANGE, alpha=0.1, color='k')
        catching_zone = plt.Circle((X_trackers_coords[n][t], Y_trackers_coords[n][t]), radius=3, fill=False, linestyle='--', color='k', alpha=0.5)
        ax.add_artist(detection_zone)
        ax.add_artist(catching_zone)
        tracker_path_x = X_trackers_coords[n][:t+1]
        tracker_trail_x = tracker_path_x[-10:]
        tracker_path_y = Y_trackers_coords[n][:t+1]
        tracker_trail_y = tracker_path_y[-10:]
        target_path_x = X_target_coords[:t+1]
        target_trail_x = target_path_x[-10:]
        target_path_y = Y_target_coords[:t+1]
        target_trail_y = target_path_y[-10:]
        ax.plot( tracker_trail_x, tracker_trail_y, color= 'g' if target_observation[n][t] else 'k', linestyle='--', alpha=0.33 )
        ax.plot( target_trail_x, target_trail_y, color= 'k', linestyle='--', alpha=0.33 )
        marker = matplotlib.markers.MarkerStyle('+')
        marker._transform = marker.get_transform().rotate_deg(THETA_trackers[n][t] * 180/np.pi)
        ax.scatter(X_trackers_coords[n][t], Y_trackers_coords[n][t], color='g' if target_observation[n][t] else 'k', marker=marker, s=100)
        ax.text(X_trackers_coords[n][t]+0.5, Y_trackers_coords[n][t]+0.5, '{:.2f}'.format(np.sqrt(Vx_trackers_magnitude[n][t]**2 + Vy_trackers_magnitude[n][t]**2)), 
                color='g' if target_observation[n][t] else 'k', ha='center', va='center', fontsize=15)

    ax.scatter(X_target_coords[t], Y_target_coords[t], color='k', marker='o', s=100, label='Target')

    ax.legend( loc='lower center', bbox_to_anchor=(0.5, -3) )
    ax.set_title('Episode {} \nStep = '.format(episode) + str(time[t]))

    return ax

fig, ax = plt.subplots(figsize=(20,20))
plt.gca().set_aspect("equal", adjustable="box")
line_ani = animation.FuncAnimation( fig, motion, interval=10, frames=int(duration), repeat=False )

f = '/home/marco/active_object_tracking_modelling/constvel_trials/gifs/episode_{}.gif'.format(episode)
writergif = animation.PillowWriter(fps=4)
line_ani.save(f, writer=writergif)

# plt.show()

del line_ani
# plt.close()