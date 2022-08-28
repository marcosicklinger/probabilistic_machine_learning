from __future__ import annotations
import numpy as np
import itertools
from tensorflow import keras as K
from tensorflow.keras import backend 
import tensorflow as T
from Utils import FeatureEmbedding, DiscreteActionSpace
from collections import deque
import random
from copy import deepcopy

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
    ROTATION_P = np.pi/6
    ROTATION_N = -np.pi/6
    NO_ROTATION = 0
    ROT_ACT = [ROTATION_N, NO_ROTATION, ROTATION_P]
    ACTIONS = np.array(ROT_ACT)
    # -------

    # SWARMAGENT PROPERTIES
    # ------------------
    SPEED = 3
    DETECTION_RANGE = 10
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

