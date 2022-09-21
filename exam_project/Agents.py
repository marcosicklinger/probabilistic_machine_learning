from __future__ import annotations
import numpy as np
import itertools
from Utils import *
from collections import deque
import random
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF,  ConstantKernel, WhiteKernel
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
    # ROTATION_P = np.pi/6
    # ROTATION_N = -np.pi/6
    # NO_ROTATION = 0
    # ROT_ACT = [ROTATION_N, NO_ROTATION, ROTATION_P]
    # ACTIONS = np.array(ROT_ACT)
    STEP_SIZE = 0.1
    MOVEMENTS = [np.array([-STEP_SIZE,0]), np.array([STEP_SIZE,0]), np.array([0,-STEP_SIZE]), np.array([0,STEP_SIZE])]
    ACTIONS = np.array(MOVEMENTS)
    # -------

    # SWARMAGENT PROPERTIES
    # ------------------
    SPEED = 3
    DETECTION_RANGE = 30
    DETECTION_PROBABILITY = lambda distance: 1 if distance < Tracker.DETECTION_RANGE else 0 # lambda distance: np.exp( - 0.1 * (distance / Tracker.DETECTION_RANGE)**2 ) if distance < Tracker.DETECTION_RANGE else 0
    INTERACTION_RANGE = 60
    # ------------------

    # @classmethod
    # def resetID(cls):
    #     cls.ID = itertools.count()

    @classmethod
    def noiseKinematicMeasure(cls, accuracy):

            mean = (0, 0)
            cov = [[accuracy, 0], [0, accuracy]]
            return np.random.multivariate_normal(mean, cov)



    @classmethod
    def noiseOrientationMeasure(cls, importance):

            return np.random.normal(-np.pi*(importance*.01), np.pi*(importance*.01), 1)[0]



    @classmethod
    def noiseDistanceMeasure(cls, importance, distance):
            return np.random.normal(0, distance*(importance*.01), 1)[0]



    def __init__(self, 
                lr_pars:"learning parameters dictionary",
                state_space_size,
                init_position=None,
                init_speed=None,
                init_orientation=None,
                memoryless=True
    ):

        self.state_space_size = state_space_size

        self.position = init_position
        self.speed = init_speed
        self.orientation = init_orientation
        self.actions = DiscreteActionSpace( len(Tracker.ACTIONS) )
        self.neighborhood = []
        self.observation = np.empty((len(self.state_space_size),))
        self.memoryless = memoryless
        if not self.memoryless: 
            self.memory = deque(maxlen=2000)
            self.replay_size = 32

        self.lr_pars = deepcopy(lr_pars)
        self.Q = np.zeros((*self.state_space_size, self.actions.n_actions))
    def seesTarget(self, distance):

        sees_target = np.random.choice([0, 1], 1, p=[1-Tracker.DETECTION_PROBABILITY(distance), Tracker.DETECTION_PROBABILITY(distance)])

        return sees_target



    def observeNeighborhood(self, new_neighbors):

        del self.neighborhood
        self.neighborhood = new_neighbors



    def observeLocalProperties(self, local_observation):

        del self.local_observation
        self.local_observation = local_observation
        return np.append(np.array( [self.position, self.speed] ).flatten(), self.orientation)


    
    def remember(self, experience):
        assert not self.memoryless, "memoryless trackers cannot remember"

        self.memory.append(experience)



    def policy(self, observation:"state or observation"):

        '''computes the agent's policy as the average of Qa and Qb'''
        # best_actions = ( self.q_hat.predict(observation) == np.max( self.q_hat.predict(observation) ) )
        best_actions = self.actions.getModelBestActions( self.Q[ (*tuple(observation), ) ] )
        policy = self.lr_pars['eps'] * np.ones( self.actions.n_actions ) / self.actions.n_actions + \
                (1 - self.lr_pars['eps']) * best_actions / np.sum(best_actions)
        # print(best_actions.shape)
        # print(policy.shape)
        return policy



    def act(self, observation:"state or observation"):

        return self.actions.sample(self.policy(observation))
        # return np.random.choice( range(Tracker.ACTIONS.shape[0]), 1, p=self.policy(z) )



    def expectedSarsaStep(self, observation, action, reward, next_observation, done, step):

        dQ = reward - self.Q[ (*observation, action) ]
        if not done: dQ += self.lr_pars['gamma'] * ( np.dot(self.policy(next_observation), self.Q[ (*next_observation, ) ]) ) 
       
        if step < self.lr_pars['start_update']:
            self.Q[ (*observation, action) ] += self.lr_pars['alpha_0']*dQ
        else:
            self.Q[ (*observation, action) ] += self.lr_pars['alpha']*dQ



    def memoryReplayStep(self, replay_size):

        memory_set = random.sample(self.memory, replay_size)

        for observation, action, reward, next_observation, done in memory_set:
            self.expectedSarsaStep(observation, action, reward, next_observation, done)
        


# -------------

# ------------------------------
# GAUSSIAN PROCESS TARCKER CLASS
# ------------------------------
class DGPQTracker:

    # ID = itertools.count()

    # ACTIONS
    # -------
    # ROTATION_P = np.pi/6
    # ROTATION_N = -np.pi/6
    # NO_ROTATION = 0
    # ROT_ACT = [ROTATION_N, NO_ROTATION, ROTATION_P]
    # ACTIONS = np.array(ROT_ACT)
    STEP_SIZE = 0.1
    MOVEMENTS = [np.array([-STEP_SIZE,0]), np.array([STEP_SIZE,0]), np.array([0,-STEP_SIZE]), np.array([0,STEP_SIZE])]
    ACTIONS = np.array(MOVEMENTS)
    # -------

    # SWARMAGENT PROPERTIES
    # ------------------
    SPEED = 3
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
        self.delta = 0.99
        self.tolerance2_numerator = 2*self.noise_level*(self.epsilon**2)*(1 - self.gamma)**4 
        self.Ns = CoverigNumber([self.low_boundaries, self.high_boundaries], self.epsilon*(1-self.gamma)/(3*self.LQ))
        self.K = self.actions.n_actions*self.Ns*( (3*self.Rmax)/(( (1 - self.gamma)**2 )*self.epsilon) + 1)
        self.tolerance2 = self.tolerance2_numerator/( 9*(self.Rmax**2)*np.log( (6/self.delta)*self.actions.n_actions*self.Ns*(1 + self.K)) )
        self.exploration = self.init_lr_pars['exploration']
        self.update_rate = 1.
        self.Qinit = 0
        self.length_scale = self.init_lr_pars['RBF_length_scale']
        self.BVset = [[] for _ in range(self.actions.n_actions)]
        self.frozen_BVset = deepcopy(self.BVset)
        self.kernel = RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.noise_level)
        # self.kernel.set_params(**{'length_scale_bounds': (1e-8, 100000.0)})
        self.GP_kernel = self.kernel
        self.GPR = [GPR(self.GP_kernel, normalize_y=True) for i in range(self.actions.n_actions)]
        # self.prior_mean = lambda s, a: self.Qa(s, a) 


    
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
        # print(len(self.BVset[action]))
        # print(self.approximator(observation, action))
        return self.approximator(observation, action)


    
    def Q(self, observation):

        Q = np.empty(self.actions.n_actions)

        for action in self.actions.action_indeces:
            Q[action] = self.Qa(observation, action)
        
        return Q



    def updateBasisVectorSet(self, mu, observation, action):

        # self.BVset[action] += [{'mu': mu, 's': observation}]
        len_before = len(self.BVset[action])
        # print(len_before, end='\r')

        self.BVset[action] += [{'mu': mu, 's': observation}]
        if bool(self.BVset[action]):
            reduntant_basis_vectors = []
            for j, bv_j in reversed(list(enumerate(self.BVset[action]))):
                if j != len(self.BVset[action]) - 1:
                    if mu + self.LQ * SADistance(observation, action, bv_j['s'], action) <= bv_j['mu']: #bv_j['mu']:
                    # if SADistance(observation, action, bv_j['s'], action) < 0.2:
                    #     if mu <= bv_j['mu']:
                        # print( SADistance(observation, action, bv_j['s'], action))
                            # reduntant_basis_vectors += [j]
                            del self.BVset[action][j]
            # print(bool(reduntant_basis_vectors), len(reduntant_basis_vectors))
            # for idx in sorted(reduntant_basis_vectors, reverse=True):
            #     del self.BVset[action][idx]

        # self.BVset[action] += [{'mu': mu, 's': observation}]

        # if len(self.BVset[action]) > 500:
        #     del self.BVset[action][0]
        # print(len(self.BVset[action]))


    def policy(self, observation:"state or observation"):

        '''computes the agent's policy as the average of Qa and Qb'''
        best_actions = self.actions.getModelBestActions( self.Q(observation) ) 
        # best_actions = self.actions.getModelBestActions( np.array( [self.GPR[action].predict(observation.reshape(1, -1), return_std=False)[0]
        #                                                             for action in range(self.actions.n_actions)] ) )
        policy = self.exploration * np.ones( self.actions.n_actions ) / self.actions.n_actions + (1 - self.exploration) * best_actions / np.sum(best_actions)
        # policy = best_actions / np.sum(best_actions)
        # # print(best_actions.shape)
        # # print(policy.shape)
        return policy



    def act(self, observation:"state or observation"):

        return self.actions.sample(self.policy(observation))
        # return np.random.choice( range(Tracker.ACTIONS.shape[0]), 1, p=self.policy(z) )



    def learn(self, observation, action, reward, next_observation, done, step):

        # print('obs ', observation, '  -  act ', action, '  -  check ', observation == next_observation)

        q = reward + self.gamma * np.max(self.Q(next_observation)) 
        # q = reward + self.gamma * np.dot( self.Q(next_observation), self.policy(next_observation) )
        # print(q)

        mu_prior = self.prior_mean(observation, action)
       
        _, std_dev_1 = self.GPR[action].predict(observation.reshape(1,-1), return_std=True)
        # print('std1', std_dev_1[0])
        if std_dev_1[0]**2  > self.tolerance2:
            # print('FIRST GP UPDATE')
            self.GPR[action].fit(observation.reshape(1,-1), np.array([q - mu_prior]))
        # if np.abs(mean_1 - q) > std_dev_1bv_j['mu']:
            # print('here')
            # self.GPR[action].fit(observation.reshape(1,-1), np.array([q]))

        mean_2, std_dev_2 = self.GPR[action].predict(observation.reshape(1,-1), return_std=True)
        # mean_2[0] += mu_prior
        # print('Qa: ', self.Qa(observation, action), '   mean: ', mean_2[0])
        # print('std2', std_dev)
        # print(std_dev_1[0]**2  > self.tolerance >= std_dev_2[0]**2, np.abs(self.Qa(observation, action) - mean_2[0]) > 2*self.eps_1)
        print(std_dev_1[0]**2, self.tolerance2, std_dev_2[0]**2, std_dev_1[0]**2 > self.tolerance2, self.tolerance2 >= std_dev_2[0]**2, self.Qa(observation, action) - mean_2[0] > 2*self.eps_1)
        # print(self.Qa(observation, action) - mean_2[0], 2*self.eps_1)
        if std_dev_1[0]**2 > self.tolerance2 >= std_dev_2[0]**2: #and self.Qa(observation, action) - mean_2[0] > 2*self.eps_1:
            if self.Qa(observation, action) - mean_2[0] > 2*self.eps_1:
                # print('SECOND GP UPDATE')
                self.updateBasisVectorSet(mean_2[0] + self.eps_1, observation, action)
                # print('woy')
                self.GPR = [GPR(self.GP_kernel, normalize_y=True) for i in range(self.actions.n_actions)]     

# -------------

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

