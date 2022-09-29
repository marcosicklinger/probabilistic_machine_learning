from __future__ import annotations

from Agents import *
from Utils import *
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

low_boundaries = [0, 0]
high_boundaries = [1,1]
Vmax = 10
N = 1
lr_pars = {'gamma': .925, 'LQ': 9, 'epsilon': 1, 'exploration': 0.1, 'RBF_length_scale': 0.05, 'delta':0.9999, 'kernel': 'RationalQuadratic2.3', 'start_update': 1e7}
current_directory = os.getcwd()
name_instance_directory = str(lr_pars).replace("'", '').replace(" ", '').replace("{", '').replace("}", '').replace(",", '_').replace(":", '_').replace(".", '_')
instance = 5
episode = 5000
BVset = np.load(current_directory + '/trials/performance/{}/BV_{}_{}_{}.npy'.format(name_instance_directory, episode, N, instance), allow_pickle=True)

n_grid = 20
X = np.linspace(low_boundaries[0] + np.abs(low_boundaries[0] -  high_boundaries[0])/(2*n_grid), high_boundaries[0]*1 - np.abs(low_boundaries[0] -  high_boundaries[0])/(2*n_grid), n_grid)
Y = np.linspace(low_boundaries[1] + np.abs(low_boundaries[1] -  high_boundaries[1])/(2*n_grid), high_boundaries[1]*1 - np.abs(low_boundaries[1] -  high_boundaries[1])/(2*n_grid), n_grid)
Qopt = np.zeros((len(X), len(Y)))
policy = np.zeros((len(X), len(Y), 2))
Q = lambda s: [np.min([np.min([bv['mu'] + lr_pars['LQ']*SADistance(s, a, bv['s'], a) for bv in BVset[a]]), Vmax]) for a in range(len(DGPQTracker.ACTIONS))]
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        Qopt[i, j] += np.max(Q(np.array([x,y])))
        policy[i, j, :] = DGPQTracker.ACTIONS[np.argmax(Q(np.array([x,y])))] 
U, V = policy[:,:,1], -policy[:,:,0]

fig, ax = plt.subplots(figsize=(10,10))
ax.set_xticks(np.arange(Qopt.shape[0]))
ax.set_xticklabels(np.round(X, 2), rotation=45)
ax.set_yticks(np.arange(Qopt.shape[1]))
ax.set_yticklabels(np.round(Y, 2))
im = ax.imshow(Qopt, cmap=plt.get_cmap("Spectral"))
q = ax.quiver(n_grid*X-.5, n_grid*Y-.5, U, V, color="black")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()