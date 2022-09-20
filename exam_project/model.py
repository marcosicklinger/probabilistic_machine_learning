from Agents import *
from Environments import *
from Utils import *

n_trackers = 1
low = [0, 0]
high = [1, 1]
lr_pars = {'gamma': .9999, 'alpha_0': .09, 'alpha': .09, 'alpha_min': 1e-7, 'exp_alpha': 0.9999, 'eps_0': .14, 'eps': .14, 'eps_min': 1e-6, 'exp_eps': 1.5, 'start_update': 5e4, 'cut': 1e50}
hyper_pars = {'time_importance': 1, 'max_time': 800}
current = RandomCurrentVelocity(-3, 3)

environment = Environment(n_trackers, low, high, lr_pars, hyper_pars, current=current, memoryless_trackers=True)

print('STATE SPACE SETTING:')
print('\t- No distance to target')
print('\t- {} angles state aggregation'.format(2*environment.orientation_slices))
print('----------------------------------------------------------')

n_episodes = 5001
environment.train(n_episodes, render=[4900, 4950, 5000], remember=False)


