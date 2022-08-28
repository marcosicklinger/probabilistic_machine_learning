from Agents import *
from Environments import *
from Utils import *

n_trackers = 1
low = [0, 0]
high = [100, 100]
lr_pars = {'gamma': .99, 'alpha_0': .001, 'alpha': .001, 'alpha_min': 1e-7, 'exp_alpha': 0.9999, 'eps_0': .01, 'eps': .01, 'eps_min': 1e-6, 'exp_eps': 1.05, 'start_update': 1e6, 'cut': 1e50}
hyper_pars = {'time_importance': 1, 'max_time': 800}
current = RandomCurrentVelocity(-3, 3)

environment = Environment(n_trackers, low, high, lr_pars, hyper_pars, current=current, memoryless_trackers=True)

print('STATE SPACE SETTING:')
print('\t- No distance to target')
print('\t- {} angles state aggregation'.format(2*environment.orientation_slices))
print('----------------------------------------------------------')

n_episodes = 50000
environment.train(n_episodes, render=[46000, 47000, 48000, 49000, 49999], remember=False)


