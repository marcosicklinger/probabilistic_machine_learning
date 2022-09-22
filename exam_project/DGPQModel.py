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
    lr_pars = {'gamma': .925, 'LQ': 9, 'epsilon': 1, 'exploration': 0.1, 'RBF_length_scale': 0.04, 'start_update': 1e7, 'delta':0.9999}
    hyper_pars = {'time_importance': 1, 'max_time': 200}
    current = RandomCurrentVelocity(-3, 3)
    environment = DGPQEnvironment(n_trackers, low, high, lr_pars, hyper_pars, current=current, memoryless_trackers=True, instance=i)
    n_episodes = 5001
    environment.train(n_episodes, render=[1000, 4900, 4950, 5000], remember=False)

instances = 2
process_number = 1
pool = mp.Pool(process_number)


[pool.apply_async(DoRun, (instance,)) for instance in range(instances)]
pool.close()
pool.join()
