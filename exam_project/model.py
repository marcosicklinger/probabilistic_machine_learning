from Agents import *
from Environments import *
from Utils import *

n_trackers = 2
low = [0, 0]
high = [100, 100]
lr_pars = {'gamma': 1., 'alpha_0': .0075, 'alpha': .0075, 'alpha_min': 1e-7, 'exp_alpha': .9, 'eps_0': .1, 'eps': .1, 'eps_min': 0.001, 'exp_eps': .9, 'start_update': 2e7}
hyper_pars = {'time_importance': 1, 'max_time': 400}
current = RandomCurrentVelocity(-3, 3)

environment = Environment(n_trackers, low, high, lr_pars, hyper_pars, current=current, memoryless_trackers=True)

n_episodes = 200000
environment.train(n_episodes, render=[199900, 199950, 199990, 199999], remember=False)


