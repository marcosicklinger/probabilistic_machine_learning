from __future__ import annotations
import numpy as np
import itertools
from Utils import *
from collections import deque
import random
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF,  ConstantKernel, WhiteKernel, Matern, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# ------
# ------
# AGENTS
# ------
# ------

# -------------
# TARCKER CLASS
# -------------
class Tracker:

    # ID = itertools.count()

    # ACTIONS
    # -------
    STEP_SIZE = 0.1
    MOVEMENTS = [np.array([-STEP_SIZE,0]), np.array([STEP_SIZE,0]), np.array([0,-STEP_SIZE]), np.array([0,STEP_SIZE])]
    ACTIONS = np.array(MOVEMENTS)
    # -------

    # SWARMAGENT PROPERTIES
    # ------------------
    DETECTION_RANGE = 30
    DETECTION_PROBABILITY = lambda distance: 1 if distance < Tracker.DETECTION_RANGE else 0
    INTERACTION_RANGE = 60
    # ------------------



    def __init__(self, 
                lr_pars,
                state_space_size,
                init_position=None
    ):

        self.state_space_size = state_space_size
        self.position = init_position
        self.actions = DiscreteActionSpace( len(Tracker.ACTIONS) )
        self.neighborhood = []
        self.observation = np.empty((len(self.state_space_size),))

        self.lr_pars = deepcopy(lr_pars)
        self.Q = np.zeros((*self.state_space_size, self.actions.n_actions))



    def seesTarget(self, distance):

        sees_target = np.random.choice([0, 1], 1, p=[1-Tracker.DETECTION_PROBABILITY(distance), Tracker.DETECTION_PROBABILITY(distance)])

        return sees_target



    def policy(self, observation):

        best_actions = self.actions.getModelBestActions( self.Q[ (*tuple(observation), ) ] )
        # e-greedy policy
        policy = self.lr_pars['eps'] * np.ones( self.actions.n_actions ) / self.actions.n_actions + \
                (1 - self.lr_pars['eps']) * best_actions / np.sum(best_actions)
        return policy



    def act(self, observation:"state or observation"):

        return self.actions.sample(self.policy(observation))



    def learn(self, observation, action, reward, next_observation, done, step):

        dQ = reward - self.Q[ (*observation, action) ]
        if not done: dQ += self.lr_pars['gamma'] * np.max(self.Q[ (*next_observation, )])
       
        if step < self.lr_pars['start_update']:
            self.Q[ (*observation, action) ] += self.lr_pars['alpha_0']*dQ
        else:
            self.Q[ (*observation, action) ] += self.lr_pars['alpha']*dQ



# -------------

# ------------------------------
# GAUSSIAN PROCESS TARCKER CLASS
# ------------------------------
class DGPQTracker:

    # ID = itertools.count()

    # ACTIONS
    # -------
    STEP_SIZE = 0.1
    MOVEMENTS = [np.array([-STEP_SIZE,0]), np.array([STEP_SIZE,0]), np.array([0,-STEP_SIZE]), np.array([0,STEP_SIZE])]
    ACTIONS = np.array(MOVEMENTS)
    # -------

    # SWARMAGENT PROPERTIES
    # ------------------
    DETECTION_RANGE = 0
    DETECTION_PROBABILITY = lambda distance: 1 if distance < Tracker.DETECTION_RANGE else 0 # lambda distance: np.exp( - 0.1 * (distance / Tracker.DETECTION_RANGE)**2 ) if distance < Tracker.DETECTION_RANGE else 0
    INTERACTION_RANGE = 0
    # ------------------



    def __init__(self, 
                init_lr_pars,
                state_dimensions,
                low, 
                high,
                init_position=None,
                init_speed=None,
                init_orientation=None,
                noise_diagonal_var=None,
                Rmax = 1,
                Vmax = 1
    ):
        self.state_dimensions = state_dimensions 
        self.low_boundaries = low
        self.high_boundaries = high
        self.position = init_position
        self.speed = init_speed
        self.orientation = init_orientation
        self.actions = DiscreteActionSpace( len(Tracker.ACTIONS) )
        self.neighborhood = []
        self.observation = np.empty(self.state_dimensions)
        self.init_lr_pars = deepcopy(init_lr_pars)
        self.gamma = self.init_lr_pars['gamma']
        self.LQ = self.init_lr_pars['LQ']
        self.Vmax = Vmax
        self.Rmax = Rmax
        self.noise_level = 0.1
        self.epsilon = self.init_lr_pars['epsilon']
        self.eps_1 = (1./3)*self.epsilon*(1 - self.gamma)
        self.delta = self.init_lr_pars['delta']
        self.tolerance2_numerator = 2*self.noise_level*(self.epsilon**2)*(1 - self.gamma)**4 
        self.Ns = CoverigNumber([self.low_boundaries, self.high_boundaries], self.epsilon*(1-self.gamma)/(3*self.LQ))
        self.K = self.actions.n_actions*self.Ns*( (3*self.Rmax)/(( (1 - self.gamma)**2 )*self.epsilon) + 1)
        self.tolerance2 = self.tolerance2_numerator/( 9*(self.Rmax**2)*np.log( (6/self.delta)*self.actions.n_actions*self.Ns*(1 + self.K)) )
        self.exploration = self.init_lr_pars['exploration']
        self.update_rate = 1.
        self.length_scale = self.init_lr_pars['length_scale']
        self.BVset = [[] for _ in range(self.actions.n_actions)]
        self.kernel_name = self.init_lr_pars['kernel']
        self.kernel_component = None
        self.GP_kernel = None
        self.reset_kernel()
        self.GPR = [GPR(self.GP_kernel, normalize_y=True) for i in range(self.actions.n_actions)]



    def reset_kernel(self):
        if self.kernel_name == 'Matern0.5':
            self.kernel_component = Matern(length_scale=self.length_scale, nu=1.5)
        if self.kernel_name == 'Matern1.5':
            self.kernel_component = Matern(length_scale=self.length_scale, nu=1.5)
        if self.kernel_name == 'Matern2.5':
            self.kernel_component = Matern(length_scale=self.length_scale, nu=1.5)
        if self.kernel_name == 'RBF':
            self.kernel_component = RBF(length_scale=self.length_scale)
        if self.kernel_name == 'RationalQuadratic1':
            self.kernel_component =  RationalQuadratic(length_scale=self.length_scale, alpha=1)
        if self.kernel_name == 'RationalQuadratic2.25':
            self.kernel_component =  RationalQuadratic(length_scale=self.length_scale, alpha=2.25)
        self.GP_kernel = self.kernel_component + WhiteKernel(noise_level=self.noise_level)


    
    def approximator(self, s, a):

        return np.min( [np.min( [bv['mu'] + self.LQ * SADistance(s, a, bv['s'], a) 
                        for bv in self.BVset[a]] ), self.Vmax] ) 



    def prior_mean(self, s, a):

        if not bool(self.BVset[a]):
            return self.Rmax/(1 - self.gamma)     
        return self.approximator(s, a)



    def Qa(self, observation, action):

        if not bool(self.BVset[action]): 
            return 0
        return self.approximator(observation, action)


    
    def Q(self, observation):

        Q = np.empty(self.actions.n_actions)
        for action in self.actions.action_indeces:
            Q[action] = self.Qa(observation, action)
        return Q



    def updateBasisVectorSet(self, mu, observation, action):

        self.BVset[action] += [{'mu': mu, 's': observation}]
        if bool(self.BVset[action]):
            reduntant_basis_vectors = []
            for a in self.actions.action_indeces:
                for j, bv_j in reversed(list(enumerate(self.BVset[a]))):
                    if j != len(self.BVset[a]) - 1:
                        if mu + self.LQ * SADistance(observation, action, bv_j['s'], a) <= self.GPR[a].predict(bv_j['s'].reshape(1,-1))[0]:#bv_j['mu']: 
                                del self.BVset[a][j]



    def policy(self, observation:"state or observation"):

        '''computes the agent's policy as the average of Qa and Qb'''
        best_actions = self.actions.getModelBestActions( self.Q(observation) ) 
        policy = best_actions / np.sum(best_actions)
        return policy



    def act(self, observation:"state or observation"):

        return self.actions.sample(self.policy(observation))

    def learn(self, observation, action, reward, next_observation, done, step):

        q = reward + self.gamma * np.max(self.Q(next_observation))
        Qa_hat = self.Qa(observation, action)

        _, std_dev_1 = self.GPR[action].predict(observation.reshape(1, -1), return_std=True)
        if std_dev_1[0] ** 2 > self.tolerance2:
            mu_prior = Qa_hat if bool(self.BVset[action]) else self.Rmax / (1 - self.gamma)
            self.GPR[action].fit(observation.reshape(1, -1), np.array([q - mu_prior]))

        mean_2, std_dev_2 = self.GPR[action].predict(observation.reshape(1, -1), return_std=True)
        # std_dev_2[0] -= std_dev_2[0]
        if std_dev_1[0] ** 2 > self.tolerance2 >= std_dev_2[0] ** 2:
            if Qa_hat - mean_2[0] > 2 * self.eps_1:
                self.updateBasisVectorSet(mean_2[0] + self.eps_1 + Qa_hat, observation, action)
                self.reset_kernel()
                self.GPR = [GPR(self.GP_kernel, normalize_y=True) for i in range(self.actions.n_actions)]


# ------------------
# DUMMY TARGET CLASS
# ------------------
class DummyTarget:

    def __init__(self,
                init_position=None,
                init_speed=None
    ):

        self.position = init_position
        self.speed = init_speed

# ------------------

