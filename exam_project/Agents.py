from __future__ import annotations
import numpy as np
import itertools
from tensorflow import keras as K
from tensorflow.keras import backend 
import tensorflow as T
from Utils import FeatureEmbedding, DiscreteActionSpace
from collections import deque
import random



class Tracker:

    ROTATION_1 = np.pi/4
    ROTATION_2 = np.pi/12
    ROTATION_3 = 5*np.pi/12
    ROTATION_5 = -np.pi/4
    ROTATION_6 = -np.pi/12
    ROTATION_7 = -5*np.pi/12
    NO_ROTATION = 0

    ROT_ACT = [ROTATION_1, ROTATION_2, ROTATION_3, ROTATION_5, ROTATION_6, ROTATION_7, NO_ROTATION]
    ACTIONS = np.array( ROT_ACT )

    SPEED = 3
    DETECTION_RANGE = 30
    DETECTION_PROBABILITY = lambda distance: 1 if distance < Tracker.DETECTION_RANGE else 0 
    INTRASWARM_INTERACTION_RANGE = 60

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
        self.observation = np.empty((5,))
        self.memoryless = memoryless
        if not self.memoryless: 
            self.memory = deque(maxlen=2000)
            self.replay_size = 32

        self.lr_pars = lr_pars
        self.Q = 'gaussian process'


    
    def remember(self, experience):
        assert not self.memoryless, "memoryless trackers cannot remember"

        self.memory.append(experience)



    def policy(self, observation:"state or observation"):
        '''computes the agent's policy as the average of Qa and Qb'''

        best_actions = self.actions.getModelBestActions( 'model' )
        policy = self.lr_pars['eps'] * np.ones( self.actions.n_actions ) / self.actions.n_actions + \
                (1 - self.lr_pars['eps']) * best_actions / np.sum(best_actions)

        return policy



    def act(self, observation:"state or observation"):
        pass



    def model():
        pass



    def learn():
        pass



    def memoryReplayStep(self, replay_size):

        memory_set = random.sample(self.memory, replay_size)

        for observation, action, reward, next_observation, done in memory_set:
            # update model
            pass

        

class DummyTarget:

    def __init__(self,
                init_position=None,
                init_speed=None
    ):

        self.position = init_position
        self.speed = init_speed


