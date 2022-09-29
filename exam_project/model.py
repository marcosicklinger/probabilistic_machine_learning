from Agents import *
from Environments import *
from Utils import *
import multiprocessing as mp
from random import randint

def DoRun(i):
    np.random.seed(i)
    n_trackers = 1
    low = [0, 0]
    high = [1, 1]
    lr_pars = {'gamma': .9, 'alpha_0': .01, 'alpha': .01, 'alpha_min': 1e-7, 'exp_alpha': 0.9999, 'eps_0': .05, 'eps': .05, 'eps_min': 1e-6, 'exp_eps': 1.5, 'start_update': 5e4, 'cut': 1e50}
    hyper_pars = {'time_importance': 1, 'max_time': 200}
    environment = Environment(n_trackers, low, high, lr_pars, hyper_pars, instance=i)
    n_episodes = 5001
    environment.train(n_episodes, render=[500, 4900, 4950, 5000])

instances = 14
process_number = 14
pool = mp.Pool(process_number)
[pool.apply_async(DoRun, (instance,)) for instance in range(instances)]
pool.close()
pool.join()