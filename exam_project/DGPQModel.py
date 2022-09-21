from Agents import *
from Environments import *
from Utils import *

n_trackers = 1
low = [0, 0]
high = [1, 1]
lr_pars = {'gamma': .8, 'LQ': 4, 'epsilon': 5, 'exploration': 0.1, 'RBF_length_scale':0.005,  'start_update': 1e7}
hyper_pars = {'time_importance': 1, 'max_time': 200}
current = RandomCurrentVelocity(-3, 3)
environment = DGPQEnvironment(n_trackers, low, high, lr_pars, hyper_pars, current=current, memoryless_trackers=True)

# print('STATE SPACE SETTING:')
# print('\t- No distance to target')
# print('\t- {} angles state aggregation'.format(2*environment.orientation_slices))
# print('----------------------------------------------------------')

n_episodes = 5001
environment.train(n_episodes, render=[1000, 4900, 4950, 5000], remember=False)