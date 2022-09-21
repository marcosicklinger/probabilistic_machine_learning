from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import threading
import random
from numpy.random import default_rng
rng = default_rng()
from Utils import *
import math
from matplotlib import animation
from matplotlib.animation import PillowWriter 
from Agents import *
from WorldFeatures import *
import itertools
from numpy import linalg as alg
import os
from matplotlib import animation
from matplotlib.animation import PillowWriter 
import os.path


class Environment:

    def __init__(self, 
                n_trackers, 
                low,
                high,
                lr_pars,
                hyper_pars, 
                environment_noise=None,
                current=None,
                memoryless_trackers=True
    ):
        
        self.n_trackers = n_trackers
        self.n_targets = 1
        self.low_boundaries = low
        self.high_boundaries = high
        self.lr_pars = lr_pars
        self.hyper_pars = hyper_pars
        self.xside = np.abs(self.high_boundaries[0] - self.low_boundaries[0])
        self.yside = np.abs(self.high_boundaries[1] - self.low_boundaries[1])
        if current is not None:
            self.current = current
        if environment_noise is not None:
            self.environment_noise = environment_noise

        self.target = DummyTarget() 

        self.catching_distance = 0.1

        self.x_binning = math.ceil( np.abs(self.low_boundaries[0]-self.high_boundaries[0]) / (self.catching_distance) )
        self.y_binning = math.ceil( np.abs(self.low_boundaries[1]-self.high_boundaries[1]) / (self.catching_distance) )
        self.position_binning = [self.x_binning, self.y_binning]
        self.neighbor_x_binning = math.ceil( np.abs(self.low_boundaries[0]-self.high_boundaries[0]) / (self.catching_distance) )
        self.neighbor_y_binning = math.ceil( np.abs(self.low_boundaries[1]-self.high_boundaries[1]) / (self.catching_distance) )
        self.neighbor_position_binning = [self.neighbor_x_binning, self.neighbor_y_binning]
        self.orientation_slices = 4
        self.orientation_binning = 2*self.orientation_slices + 1
        self.neighbor_bearing_slices = 1
        self.neighbor_bearing_binning = 2*self.neighbor_bearing_slices + 1
        max_distance = Tracker.DETECTION_RANGE
        self.target_distance_binning = math.floor(max_distance/self.catching_distance)
        max_distance_communication = Tracker.INTERACTION_RANGE
        self.neighbor_distance_binning = math.floor(max_distance_communication/10)
        self.boundary_binning = 5
        # self.state_space_size = (self.position_binning[0]+1, self.position_binning[1]+1, self.orientation_binning, 
        #                         self.target_distance_binning+1, self.orientation_binning, 
        #                         self.target_distance_binning+1, self.orientation_binning)
        self.tracker_action_space_size = len(Tracker.ACTIONS)
        self.state_space_size = (
                                self.position_binning[0]+1,
                                self.position_binning[1]+1,
                                # self.neighbor_position_binning[0]+1,
                                # self.neighbor_position_binning[1]+1,
                                # self.orientation_binning,  
                                # self.boundary_binning,
                                # self.neighbor_distance_binning+1, 
                                # self.neighbor_bearing_binning, 
                                # self.orientation_binning,
                                # self.target_distance_binning+1, 
                                # self.orientation_binning
        )

        self.trackers_starting_area = (self.xside/4, self.yside/4) 

        zones = [0,1,2,3]
        n_zones = len(zones)
        zone_xamplitude = self.xside/n_zones
        zone_yamplitude = self.yside/n_zones
    
        target_starting_zone_x = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        target_starting_zone_y = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        target_low_x = target_starting_zone_x*zone_xamplitude
        target_high_x = (target_starting_zone_x+1)*zone_xamplitude
        target_low_y = target_starting_zone_y*zone_yamplitude
        target_high_y = (target_starting_zone_y+1)*zone_yamplitude
        self.target_position = np.array([0.95, 0.95]) # self.__initializeAgentsPosition__(self.n_targets, [target_low_x, target_low_y], [target_high_x, target_high_y])[0]

        self.memoryless_trackers = memoryless_trackers
        self.trackers_network = dict()
        for n in range(n_trackers):
            self.trackers_network[n] = Tracker(self.lr_pars, self.state_space_size, memoryless=self.memoryless_trackers)
        
        self.actions = {n: None for n in self.trackers_network.keys()}



    def __aggregatePosition__(self, position):

        pos_state = [None, None]
        
        for k in [0, 1]:
            for x in range(self.position_binning[k]):
                if self.low_boundaries[k] + x*self.catching_distance <= position[k] < self.low_boundaries[k] + (x+1)*self.catching_distance:
                    pos_state[k] = x
            if pos_state[k] is None:
                pos_state[k] = self.position_binning[k]

        return pos_state

    

    def __aggregateOrientation__(self, orientation, slices):

        if orientation is None: 
            return 0
        elif np.abs(orientation) == np.pi:
            return 2*slices

        orientation_state = None

        for a in range(1, slices+1):
            if orientation >= 0:
                if (a-1) * np.pi/slices <= np.abs(orientation) < a * np.pi/slices:
                    orientation_state = a
                    if orientation < 0: 
                        orientation_state *= -1
            else:
                if (a-1) * np.pi/slices < np.abs(orientation) <=  a * np.pi/slices:
                    orientation_state = a
                    if orientation < 0: 
                        orientation_state *= -1
        
        if orientation_state is None:
            return 0

        return orientation_state + (slices+1) if orientation_state < 0 else orientation_state + slices


    
    def __aggregateDistance__(self, distance, binning):
        
        distance_state = None

        for d in range(binning):
            if d*self.catching_distance <= distance < (d+1)*self.catching_distance:
                distance_state = d

        if distance_state is None:
            distance_state = binning

        return distance_state

    def __aggregateBoundary__(self, position, target_position=None):
        
        x_distances_from_boundary = [np.abs(position[0] - self.low_boundaries[0]), np.abs(position[0] - self.high_boundaries[0])]
        x_nearest_boundary = np.argmin(x_distances_from_boundary)
        y_distances_from_boundary = [np.abs(position[1] - self.low_boundaries[1]), np.abs(position[1] - self.high_boundaries[1])]
        y_nearest_boundary = np.argmin(y_distances_from_boundary)

        nearest_boundary_encoding = None
        if (x_distances_from_boundary[x_nearest_boundary] < Tracker.DETECTION_RANGE/2 or y_distances_from_boundary[y_nearest_boundary] < Tracker.DETECTION_RANGE/2):
            if x_distances_from_boundary[x_nearest_boundary] < y_distances_from_boundary[y_nearest_boundary]:
                if x_nearest_boundary == 0:
                    nearest_boundary_encoding = 1
                else:
                    nearest_boundary_encoding = 3
            else: 
                if y_nearest_boundary == 0:
                    nearest_boundary_encoding = 4
                else:
                    nearest_boundary_encoding = 2
        else:
            nearest_boundary_encoding = 0

        return nearest_boundary_encoding


    def __initializeAgentsPosition__(self, N, low, high):

        return [ np.random.uniform(low=low, high=high, size=(2,)) for _ in range(N) ]
        # return [ np.array([np.random.random() * agents_starting_area[0] + low[0], np.random.random() * agents_starting_area[1] + low[1]]) for _ in range(n) ]


        
    def __initializeAgentsVelocity__(self, n):

        # return [ np.random.random((2,)) for _ in range(n) ]
        return [ np.array([0, 0]) for _ in range(n) ]



    def __initializeAgentsOrientation__(self, n, group_orientation = np.pi/4):

        initial_orientation = np.random.uniform(-np.pi, np.pi, 1)[0]

        return [ initial_orientation for _ in range(n) ]



    def __computeDistance__(self, x1, x2):

        return alg.norm(x1-x2, ord=2) 

    

    def __computeRelativePosition(self, x1, x2):

        return x1-x2


    
    def __computeRelativeVelocity__(self, v1, v2):

        return v1-v2



    def __computeRelativeOrientation__(self, v1, v2):

        u1 = v1/alg.norm(v1, ord=2)
        u2 = v2/alg.norm(v2, ord=2)
        return np.arccos( np.dot(u1, u2) ) 



    def __computeBearing__(self, dx, v, v_orientation):
        

        # dx_rotated = np.dot(rotationMatrix(-v_orientation), dx)
        # dx_rotated_ycomponent_sign = dx_rotated[1]/np.abs(dx_rotated[1])

        # dx_normalized = dx/alg.norm(dx, ord=2)
        # if alg.norm(v, ord=2) > 0: 
        #     v_normalized = v/alg.norm(v, ord=2)
        # else:
        #     v_normalized = np.array( [np.cos(v_orientation), np.sin(v_orientation)] )

        # # print(dx_rotated, dx_rotated_ycomponent_sign, dx_normalized, v_normalized)

        # return np.arccos( np.dot(v_normalized, dx_normalized) ) if dx_rotated_ycomponent_sign > 0 else -np.arccos( np.dot(v_normalized, dx_normalized) )

        return np.arctan2(dx[1], dx[0])
    

    def __normalizeObservation__(self, observation):

        tracker_position_norm = alg.norm(observation[0][:2], ord=2)
        if tracker_position_norm>0: observation[0][:2] /= tracker_position_norm
        target_relative_position_norm = alg.norm(observation[0][2:4], ord=2)
        if target_relative_position_norm>0: observation[0][2:4] /= target_relative_position_norm
        # print(observation, target_relative_position_norm)

        return observation



    def getTrackersObservation(self, n, noisy=[0, 0, 0, 0]):

        observed_position = self.__aggregatePosition__(self.trackers_true_positions[n])

        if self.trackers_true_orientations[n] != None:
            observed_orientation = self.__aggregateOrientation__(self.trackers_true_orientations[n] + np.random.normal(0, noisy[1]), self.orientation_slices)
        else:
            observed_orientation = self.__aggregateOrientation__(self.trackers_true_orientations[n], self.orientation_slices)

        tracker_target_distance = self.__computeDistance__(self.target_position, self.trackers_true_positions[n]) + np.random.normal(0, noisy[2])
        tracker_target_relative_position = self.target_position - self.trackers_true_positions[n]
        observed_target_distance = self.__aggregateDistance__(tracker_target_distance, self.target_distance_binning)
        bearing = None
        if tracker_target_distance < Tracker.DETECTION_RANGE:
            bearing = self.__computeBearing__(tracker_target_relative_position, self.trackers_true_velocities[n], self.trackers_true_orientations[n])
        if bearing !=None:
            bearing += np.random.normal(0, noisy[3])
            observed_bearing = self.__aggregateOrientation__(bearing, self.orientation_slices)
        else:
            observed_bearing = self.__aggregateOrientation__(bearing, self.orientation_slices)

        min_neighbor_dist = np.inf
        nearest_neighbor = None
        for m in set(self.trackers_network.keys()) - set([n]):
            neighbor_distance = self.__computeDistance__(self.trackers_true_positions[n], self.trackers_true_positions[m]) + np.random.normal(0, noisy[2])
            if neighbor_distance < min_neighbor_dist:
                min_neighbor_dist = neighbor_distance
                nearest_neighbor = m
        # if min_neighbor_dist < Tracker.INTERACTION_RANGE:

        observed_nearest_neighbor_distance = self.__aggregateDistance__(min_neighbor_dist, self.neighbor_distance_binning)
        nearest_neighbor_bearing = None
        if min_neighbor_dist <= Tracker.INTERACTION_RANGE:
            nearest_neighbor_relative_position = self.trackers_true_positions[nearest_neighbor] - self.trackers_true_positions[n]
            nearest_neighbor_bearing = self.__computeBearing__(nearest_neighbor_relative_position, self.trackers_true_velocities[n], self.trackers_true_orientations[n])
        observed_nearest_neighbor_bearing = self.__aggregateOrientation__(nearest_neighbor_bearing, self.neighbor_bearing_slices)

        if nearest_neighbor is not None:
            observed_nearest_neighbor_orientation = self.__aggregateOrientation__(np.abs(self.trackers_true_orientations[n] - self.trackers_true_orientations[nearest_neighbor]), self.orientation_slices)
        else:
            observed_nearest_neighbor_orientation = 0

        nearest_boundary = self.__aggregateBoundary__(self.trackers_true_positions[n])

        observation = np.array( [
                                observed_position[0],
                                observed_position[1],
                                # observed_orientation, 
                                # nearest_boundary,
                                # observed_nearest_neighbor_distance, 
                                # observed_nearest_neighbor_bearing, 
                                # observed_nearest_neighbor_orientation,
                                # observed_target_distance, 
                                # observed_bearing
                                ] )
        # print(observation)

        return observation


    def rewardFunction(self, done, time, next_observation):
        
        reward_dict = dict()
        # time_reward = self.hyper_pars['time_importance'] * time
        position_check = lambda n: self.low_boundaries[0] < self.trackers_true_positions[n][0] < self.high_boundaries[0] and self.low_boundaries[1] < self.trackers_true_positions[1] < self.high_boundaries[1] 

        reward_dict = {n: -1 for n in self.trackers_network.keys()} # -1 if time < self.hyper_pars['max_time'] else -100
        # reward_dict = {n: 0 for n in self.trackers_network.keys()}
        # reward_dict = {n: -time/self.hyper_pars['max_time'] for n in self.trackers_network.keys()}

        # min_distance = np.infty
        # nearest_tracker = None
        # for i in self.trackers_network.keys():
        #     for j in set(self.trackers_network.keys()) - set([i]):
        #         distance_ij = self.__aggregateDistance__(alg.norm(self.trackers_true_positions[i] - self.trackers_true_positions[j], ord=2), self.neighbor_distance_binning)
        #         if distance_ij == 0: distance_ij = 0.1
        #         if distance_ij < min_distance:
        #             min_distance = distance_ij
        #             nearest_tracker = j

        #     if min_distance < Tracker.INTERACTION_RANGE:
        #             # reward_dict[i] += -1./(self.__aggregateDistance__(min_distance, self.neighbor_distance_binning) + 1)
        #             reward_dict[i] += -1./min_distance

        #     if next_observation[i][1] != 0: 
        #         reward_dict[i] += -1

        # min_neighbor_dist = np.inf
        # nearest_neighbor = None
        # for n in self.trackers_network.keys():
        #     for m in set(self.trackers_network.keys()) - set([n]):
        #         neighbor_distance = self.__computeDistance__(self.trackers_true_positions[n], self.trackers_true_positions[m])
        #         if neighbor_distance < min_neighbor_dist:
        #             min_neighbor_dist = neighbor_distance
        #             nearest_neighbor = m
        #     if min_neighbor_dist <= Tracker.INTERACTION_RANGE:
        #         reward_dict[n] += (np.cos(self.trackers_true_orientations[n] - self.trackers_true_orientations[m]) - 1) / min_neighbor_dist
            # if self.trackers_network[n].observation[2] != 5:
            #     reward_dict[n] -= 


        return reward_dict



    def reset(self):

        zones = [0,1,2,3]
        n_zones = len(zones)
        zone_xamplitude = self.xside/n_zones
        zone_yamplitude = self.yside/n_zones

        trackers_starting_zone_x = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        trackers_starting_zone_y = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        trackers_low_x = trackers_starting_zone_x*zone_xamplitude
        trackers_high_x = (trackers_starting_zone_x+1)*zone_xamplitude
        trackers_low_y = trackers_starting_zone_y*zone_yamplitude
        trackers_high_y = (trackers_starting_zone_y+1)*zone_yamplitude
        self.trackers_starting_positions = self.__initializeAgentsPosition__(self.n_trackers, [trackers_low_x, trackers_low_y], [trackers_high_x, trackers_high_y])
        self.trackers_true_positions = [self.trackers_starting_positions[n] for n in range(self.n_trackers)]

        self.trackers_starting_orientations = self.__initializeAgentsOrientation__(self.n_trackers)
        self.trackers_true_orientations = [self.trackers_starting_orientations[n] for n in range(self.n_trackers)]

        self.trackers_starting_velocities = [ Tracker.SPEED * np.array( [np.cos(self.trackers_starting_orientations[n]), np.sin(self.trackers_starting_orientations[n])] ) for n in range(self.n_trackers)]
        self.trackers_true_velocities = [self.trackers_starting_velocities[n] for n in range(self.n_trackers)]

        for n in self.trackers_network.keys():

            self.trackers_network[n].observation = self.getTrackersObservation(n)



    def __periodicPositionCorrection__(self, new_position):
        
        for coordinate in [0,1]:
            if new_position[coordinate] < self.low_boundaries[coordinate]:
                new_position[coordinate] = self.high_boundaries[coordinate] - np.abs(new_position[coordinate] - self.low_boundaries[coordinate])
            elif new_position[coordinate] > self.high_boundaries[coordinate]:
                new_position[coordinate] = self.low_boundaries[coordinate] + np.abs(new_position[coordinate] - self.high_boundaries[coordinate])

        return new_position

        
    def step(self, action_dict, t):

        is_rl_step = True

        # PERIODIC BC on target
        # dx_target = self.current.generateCurrent()
        # new_target_position = self.target_position + self.current.generateCurrent()
        # self.target_position = self.__periodicPositionCorrection__(new_target_position)

        # INFINITE BC on target
        # self.target_position = self.target_position + self.current.generateCurrent()
        
        next_observation = [None]*self.n_trackers

        # IF BOUNCES ON BOUDARY, SKIP RL STEP (EBC ONLY)
        # for i in self.trackers_network.keys():

        #     if self.trackers_true_positions[i][0] < self.low_boundaries[0] or self.trackers_true_positions[i][0] > self.high_boundaries[0]:
        #         self.trackers_true_velocities[i][0] *= -1
        #         # CORRECT ORIENTATION
        #         if alg.norm( self.trackers_true_velocities[i] ) > 0:
        #             self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
        #                                         - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
        #         self.trackers_true_positions[i] = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
        #         is_rl_step = False
        #     elif self.trackers_true_positions[i][1] < self.low_boundaries[1] or self.trackers_true_positions[i][1] > self.high_boundaries[1]:
        #         self.trackers_true_velocities[i][1] *= -1
        #         # CORRECT ORIENTATION
        #         if alg.norm( self.trackers_true_velocities[i] ) > 0:
        #             self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
        #                                             - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
        #         self.trackers_true_positions[i] = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
        #         is_rl_step = False
            
        # if not is_rl_step:
        #     reward_dict = {n: 0 for n in self.trackers_network.keys()}

        #     for i in self.trackers_network.keys(): next_observation[i] = self.getTrackersObservation(i)
        #     done = False
    
        #     for i in self.trackers_network.keys():
        #         if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance: 
        #             done = True
        #             reward_dict[i] = 100

        #     return next_observation, reward_dict, done, is_rl_step


        for i in self.trackers_network.keys():
            
            # rotation_angle = Tracker.ACTIONS[action_dict[i]]
            movement = Tracker.ACTIONS[action_dict[i]] + np.random.multivariate_normal([0,0], [[0.01,0],[0,0.01]])

            # self.trackers_true_velocities[i] = np.dot(rotationMatrix(rotation_angle), self.trackers_true_velocities[i]) #+ np.random.normal(0, 0.15)

            # if alg.norm( self.trackers_true_velocities[i] ) > 0:
            #     # self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
            #     #                                    - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
            #     self.trackers_true_orientations[i] = np.arctan2(self.trackers_true_velocities[i][1], self.trackers_true_velocities[i][0])

            # new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            new_position = self.trackers_true_positions[i] + movement
            if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
                self.trackers_true_positions[i] = new_position


            # TRIVIAL BC: agent stops at boundary if it tries to go out of the search region
            # if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
            #     self.trackers_true_positions[i] = new_position

            # SOFT ELASTIC BC: agent bounces back softly when it tries to go out of the search region ("softly referring to the fact that it doesnt bounce 
            # back exacly at the physical boundary of the search region but whenever it steps outside of it the velocity is multiplied by -1")
            # if self.low_boundaries[0] - 5 < new_position[0]  < self.high_boundaries[0] + 5 and self.low_boundaries[1] - 5 < new_position[1] < self.high_boundaries[1] + 5:
            #     self.trackers_true_positions[i] = new_position
            # if new_position[0] < self.low_boundaries[0] - 5 or new_position[0] > self.high_boundaries[0] + 5:
            #     self.trackers_true_velocities[i] = np.dot(rotationMatrix(-rotation_angle), self.trackers_true_velocities[i])
            #     self.trackers_true_velocities[i][0] *= -1
            #     # CORRECT ORIENTATION
            #     if alg.norm( self.trackers_true_velocities[i] ) > 0:
            #         # self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
            #         #                                - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
            #         self.trackers_true_orientations[i] = np.arctan2(self.trackers_true_velocities[i][1], self.trackers_true_velocities[i][0])
            #     new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            # elif new_position[1] < self.low_boundaries[1] - 5 or new_position[1] > self.high_boundaries[1] + 5:
            #     self.trackers_true_velocities[i] = np.dot(rotationMatrix(-rotation_angle), self.trackers_true_velocities[i])
            #     self.trackers_true_velocities[i][1] *= -1
            #     # CORRECT ORIENTATION
            #     if alg.norm( self.trackers_true_velocities[i] ) > 0:
            #         # self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
            #         #                                - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
            #         self.trackers_true_orientations[i] = np.arctan2(self.trackers_true_velocities[i][1], self.trackers_true_velocities[i][0])
            #     new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            # self.trackers_true_positions[i] = new_position

            # PERIODIC BC
            # self.trackers_true_positions[i] = self.__periodicPositionCorrection__(new_position)

            # INFINITE BOUNDARY
            # self.trackers_true_positions[i] = new_position

            # self.trackers_network[i].last_target_observation = self.trackers_network[i].target_observation

        for i in self.trackers_network.keys(): 
            next_observation[i] = self.getTrackersObservation(i)

        # status_of_search = [self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance for i in self.trackers_network.keys()]
        # done = np.sum( status_of_search ) > 0 
        # winners = [i for i, e in enumerate(status_of_search) if e == True]

        done = False
        rewards = self.rewardFunction(done, t, next_observation)
    
        for i in self.trackers_network.keys():
            if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance: 
                done = True
                # rewards[i] = 100 # self.hyper_pars['max_time'] - t

        return next_observation, rewards, done, is_rl_step



    def train(self, n_episodes, render=[], remember=False):

        self.reward_trajectories = {n: [0]*n_episodes for n in self.trackers_network.keys()}
        self.time_trajectories = [] 
        self.inefficient_coverage = [[0]*n_episodes for n in self.trackers_network.keys()]
        self.efficient_coverage = [[0]*n_episodes for n in self.trackers_network.keys()]
        render_episodes = {}
        long_episodes = {}

        Na_t = np.zeros( (self.n_trackers, len(Tracker.ACTIONS)) )
        Na_tstar = None

        Mao_t = np.zeros( (self.n_trackers, *self.state_space_size, len(Tracker.ACTIONS)) )
        Mao_tstar = None

        step = 0
        succ_episodes = 0
        for episode in range(n_episodes):

            self.reset()

            # print(self.target_position)

            # if episode in render:
                # print(episode, '----------------------------------')
                # print(self.trackers_network[0].position)
            trackers_positions_history = [[self.trackers_true_positions[n]] for n in self.trackers_network.keys()]
            trackers_velocities_history = [[self.trackers_true_velocities[n]] for n in self.trackers_network.keys()]
            trackers_orientation_history = [[self.trackers_true_orientations[n]] for n in self.trackers_network.keys()]
            trackers_target_observation_history = [[1 if self.__computeDistance__(self.trackers_true_positions[n], self.target_position) <= Tracker.DETECTION_RANGE else 0] for n in self.trackers_network.keys()]
            trackers_action_history = [[] for n in self.trackers_network.keys()]
            coverage_overlappings = [[] for n in self.trackers_network.keys()]
            nonoverlapped_coverage = [[] for n in self.trackers_network.keys()]

            target_positions_history = [self.target_position]

            done = False
            time = 1
            while not done and time < self.hyper_pars['max_time']: 
                
                for i in self.trackers_network.keys():
                    # print(self.trackers_network[i].observation)
                    self.actions[i] = self.trackers_network[i].act(self.trackers_network[i].observation)
                    # if episode in render: 
                    trackers_action_history[i] += [self.actions[i]]
                    # if time == 1: self.trackers_network[i].observation[2] = self.actions[i]
                    Na_t[ i,  self.actions[i] ] += 1
                    Mao_t[(i, *self.trackers_network[i].observation, self.actions[i])] += 1

                
                next_observation, rewards, done, is_rl_step = self.step(self.actions, time)
                # print(next_observation)
                
                for i in self.trackers_network.keys():

                    # self.trackers_network[i].deepExpectedSarsaStep(T.stack(self.environment_observation[i]/alg.norm(self.environment_observation[i], ord=2)), self.actions[i], rewards[i], T.stack(next_observation[i]/alg.norm(next_observation[i], ord=2)), done, step)
                    if remember:
                        self.trackers_network[i].remember((T.stack( self.__normalizeObservation__(self.environment_observation[i]) ), self.actions[i], rewards[i], T.stack( self.__normalizeObservation__(next_observation[i]) ), done))
                        if len(self.trackers_network[i].memory) > self.trackers_network[i].replay_size:
                            self.trackers_network[i].memoryReplayStep(self.trackers_network[i].replay_size)
                    else:
                        if is_rl_step:
                            # print( self.trackers_network[i].observation, self.actions, rewards, next_observation, done, step )
                            self.trackers_network[i].expectedSarsaStep( self.trackers_network[i].observation, self.actions[i], rewards[i], next_observation[i], done, step )

                    if step == self.trackers_network[i].lr_pars['start_update']: 
                        Na_tstar = deepcopy(Na_t)
                        Mao_tstar = deepcopy(Mao_t)

                    if step == self.lr_pars['cut']:
                        self.trackers_network[i].lr_pars['alpha_0'] *= 0.05
                        # self.lr_pars['alpha'] = self.lr_pars['alpha_0']
                        # self.lr_pars['eps_0'] *= 0.1
                        # self.lr_pars['eps'] = self.lr_pars['eps_0']

                    if step > self.trackers_network[i].lr_pars['start_update'] and episode < n_episodes - n_episodes//20:
                        if self.trackers_network[i].lr_pars['eps'] > self.trackers_network[i].lr_pars['eps_min']:
                            self.trackers_network[i].lr_pars['eps'] = self.trackers_network[i].lr_pars['eps_0'] / ( (step / self.trackers_network[i].lr_pars['start_update']) ** self.trackers_network[i].lr_pars['exp_eps'] )
                            # self.trackers_network[i].lr_pars['eps'] = self.trackers_network[i].lr_pars['eps_0'] / ( (Na_t[ i, self.actions[i] ] / Na_tstar[ i, self.actions[i] ]) ** self.trackers_network[i].lr_pars['exp_eps'] )
                            # self.trackers_network[i].lr_pars['eps'] = self.trackers_network[i].lr_pars['eps_0'] / ( (Mao_t[ (i, *self.trackers_network[i].observation, self.actions[i]) ] / Mao_tstar[ (i, *self.trackers_network[i].observation, self.actions[i]) ]) ** self.trackers_network[i].lr_pars['exp_eps'] )
                        else:
                            self.trackers_network[i].lr_pars['eps'] = 0

                        if self.trackers_network[i].lr_pars['alpha'] > self.trackers_network[i].lr_pars['alpha_min']:
                            self.trackers_network[i].lr_pars['alpha'] = self.trackers_network[i].lr_pars['alpha_0'] / ( (step / self.trackers_network[i].lr_pars['start_update']) ** self.trackers_network[i].lr_pars['exp_alpha'] )
                            # self.trackers_network[i].lr_pars['alpha'] = self.trackers_network[i].lr_pars['alpha_0'] / ( (Na_t[ i, self.actions[i] ] / Na_tstar[ i, self.actions[i] ]) ** self.trackers_network[i].lr_pars['exp_alpha'] )
                            # self.trackers_network[i].lr_pars['alpha'] = self.trackers_network[i].lr_pars['alpha_0'] / ( (Mao_t[ (i, *self.trackers_network[i].observation, self.actions[i]) ] / Mao_tstar[ (i, *self.trackers_network[i].observation, self.actions[i]) ]) ** self.trackers_network[i].lr_pars['exp_alpha'] )
                    elif episode > n_episodes - n_episodes//20:
                        self.trackers_network[i].lr_pars['eps'] = 0

                    self.reward_trajectories[i][episode] += rewards[i]

                for i in self.trackers_network.keys():
                    self.trackers_network[i].observation = next_observation[i]
              
                # if episode in render:
                    # print(done)
                for i in self.trackers_network.keys(): 
                    # print(self.__computeDistance__(self.trackers_true_positions[i], self.target_position))                               
                    trackers_positions_history[i] += [self.trackers_true_positions[i]] #[list(self.trackers_network[i].position)]
                    trackers_velocities_history[i] += [self.trackers_true_velocities[i]]
                    trackers_orientation_history[i] += [self.trackers_true_orientations[i]]
                    trackers_target_observation_history[i] += [1 if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= Tracker.DETECTION_RANGE else 0]
                    
                    for j in set(self.trackers_network.keys()) - set([i]):
                        relative_position_ij = self.trackers_true_positions[i] - self.trackers_true_positions[j]
                        distance_ij = alg.norm(relative_position_ij, ord=2)
                        if distance_ij < Tracker.DETECTION_RANGE:
                            self.inefficient_coverage[i][episode] += 1
                        else:
                            self.efficient_coverage[i][episode] += 1

                target_positions_history += [self.target_position]

                # print(self.trackers_network[i].position)  
                # print(self.trackers_starting_positions)
                # print(self.trackers_starting_velocities)

                time += 1
                step += 1
                if(done): 
                    succ_episodes += 1
                
                print(episode, end='\r')


            if episode % 500 == 0:
                print('ep = ', episode, ', step = ', step, ', t = ', time, 'ratio of successful episodes = {:.2f}'.format(succ_episodes/(episode+1)), ', eps = ', [self.trackers_network[n].lr_pars['eps'] for n in self.trackers_network.keys()], ', alpha = ', [self.trackers_network[n].lr_pars['alpha'] for n in self.trackers_network.keys()])

            self.time_trajectories += [time]
            for i in self.trackers_network.keys(): 
                self.inefficient_coverage[i][episode] /= time
                self.efficient_coverage[i][episode] /= time

            lr_pars = {'a': self.lr_pars['alpha_0'], 'expa': self.lr_pars['exp_alpha'], 'eps': self.lr_pars['eps_0'], 'expe': self.lr_pars['exp_eps'], 'p_angle': 2*self.orientation_slices}
            if (episode in render) or (time > 200 and episode > n_episodes-n_episodes//20):
                episode_dict = {'episode': episode,
                            'length': time,
                            'tracker_trajectory': deepcopy(trackers_positions_history),
                            'tracker_velocity': deepcopy(trackers_velocities_history),
                            'tracker_orientation': deepcopy(trackers_orientation_history),
                            'observation_history': deepcopy(trackers_target_observation_history),
                            'action_history': deepcopy(trackers_action_history),
                            'target_trajectory': deepcopy(target_positions_history)
            }
            if episode in render:
                render_episodes[episode] = deepcopy(episode_dict)

                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_positions_history_doubleNoTDist_{}_{}'.format(lr_pars, episode), trackers_positions_history) 
                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_velocities_history_doubleNoTDist_{}_{}'.format(lr_pars, episode), trackers_velocities_history) 
                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_orientation_history_doubleNoTDist_{}_{}'.format(lr_pars, episode), trackers_orientation_history) 
                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_target_observation_history_doubleNoTDist_{}_{}'.format(lr_pars, episode), trackers_target_observation_history) 
                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/target_positions_history_doubleNoTDist_{}_{}'.format(lr_pars, episode), target_positions_history)
                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/duration_doubleNoTDist_{}_{}'.format(lr_pars, episode), time)
                # np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/actions_{}_doubleNoTDist_{}'.format(episode, lr_pars), trackers_action_history)

                np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/rewards_up_to_{}_singleNoTDist_{}'.format(episode, lr_pars), self.reward_trajectories)
                np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_singleNoTDist_{}'.format(episode, lr_pars), self.time_trajectories)

            elif (time > 200 and episode > n_episodes-n_episodes//20):
                long_episodes[episode] = deepcopy(episode_dict)
                
            del trackers_positions_history
            del trackers_velocities_history
            del trackers_orientation_history
            del trackers_target_observation_history
            del target_positions_history
            del trackers_action_history
        
        np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/episode/renderepisodes_singleNoTDist_{}'.format(lr_pars), render_episodes)
        np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/episode/longepisodes_singleNoTDist_{}'.format(lr_pars), long_episodes)

        for i in self.trackers_network.keys():
            np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/Q/q_matrix_{}_singleNoTDist_{}'.format(i, lr_pars), self.trackers_network[i].Q)
            


##############################################################
# ENVIRONMENT UILIZING DELAYED GAUSSIAN PROCESSES Q-LEARNING #
##############################################################

class DGPQEnvironment:

    def __init__(self, 
                n_trackers, 
                low,
                high,
                lr_pars,
                hyper_pars, 
                environment_noise=None,
                current=None,
                memoryless_trackers=True
    ):
        
        self.state_dimensions = 2
        self.lr_pars = lr_pars
        self.noise_diagonal_var = None

        self.n_trackers = n_trackers
        self.n_targets = 1
        
        self.low_boundaries = low
        self.high_boundaries = high
        self.hyper_pars = hyper_pars
        self.xside = np.abs(self.high_boundaries[0] - self.low_boundaries[0])
        self.yside = np.abs(self.high_boundaries[1] - self.low_boundaries[1])
        if current is not None:
            self.current = current
        if environment_noise is not None:
            self.environment_noise = environment_noise

        self.target = DummyTarget() 

        self.catching_distance = 0.1

        self.trackers_starting_area = (self.xside/4, self.yside/4) 

        zones = [0,1,2,3]
        n_zones = len(zones)
        zone_xamplitude = self.xside/n_zones
        zone_yamplitude = self.yside/n_zones
    
        target_starting_zone_x = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        target_starting_zone_y = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        target_low_x = target_starting_zone_x*zone_xamplitude
        target_high_x = (target_starting_zone_x+1)*zone_xamplitude
        target_low_y = target_starting_zone_y*zone_yamplitude
        target_high_y = (target_starting_zone_y+1)*zone_yamplitude
        self.target_position = np.array([0.95, 0.95]) #self.__initializeAgentsPosition__(self.n_targets, [target_low_x, target_low_y], [target_high_x, target_high_y])[0]

        self.memoryless_trackers = memoryless_trackers
        self.trackers_network = dict()
        for n in range(n_trackers):
            self.trackers_network[n] = DGPQTracker(self.lr_pars, self.state_dimensions, self.low_boundaries, self.high_boundaries)
        
        self.actions = {n: None for n in self.trackers_network.keys()}



    def __initializeAgentsPosition__(self, N, low, high):

        return [ np.random.uniform(low=low, high=high, size=(2,)) for _ in range(N) ]
        # return [ np.array([np.random.random() * agents_starting_area[0] + low[0], np.random.random() * agents_starting_area[1] + low[1]]) for _ in range(n) ]


        
    def __initializeAgentsVelocity__(self, n):

        # return [ np.random.random((2,)) for _ in range(n) ]
        return [ np.array([0, 0]) for _ in range(n) ]



    def __initializeAgentsOrientation__(self, n, group_orientation = np.pi/4):

        initial_orientation = np.random.uniform(-np.pi, np.pi, 1)[0]

        return [ initial_orientation for _ in range(n) ]



    def __computeDistance__(self, x1, x2):

        return alg.norm(x1-x2, ord=2) 

    

    def __computeRelativePosition(self, x1, x2):

        return x1-x2


    
    def __computeRelativeVelocity__(self, v1, v2):

        return v1-v2



    def __computeRelativeOrientation__(self, v1, v2):

        u1 = v1/alg.norm(v1, ord=2)
        u2 = v2/alg.norm(v2, ord=2)
        return np.arccos( np.dot(u1, u2) ) 



    def __computeBearing__(self, dx, v, v_orientation):
        

        # dx_rotated = np.dot(rotationMatrix(-v_orientation), dx)
        # dx_rotated_ycomponent_sign = dx_rotated[1]/np.abs(dx_rotated[1])

        # dx_normalized = dx/alg.norm(dx, ord=2)
        # if alg.norm(v, ord=2) > 0: 
        #     v_normalized = v/alg.norm(v, ord=2)
        # else:
        #     v_normalized = np.array( [np.cos(v_orientation), np.sin(v_orientation)] )

        # # print(dx_rotated, dx_rotated_ycomponent_sign, dx_normalized, v_normalized)

        # return np.arccos( np.dot(v_normalized, dx_normalized) ) if dx_rotated_ycomponent_sign > 0 else -np.arccos( np.dot(v_normalized, dx_normalized) )

        return np.arctan2(dx[1], dx[0])
    

    def __normalizeObservation__(self, observation):

        tracker_position_norm = alg.norm(observation[0][:2], ord=2)
        if tracker_position_norm>0: observation[0][:2] /= tracker_position_norm
        target_relative_position_norm = alg.norm(observation[0][2:4], ord=2)
        if target_relative_position_norm>0: observation[0][2:4] /= target_relative_position_norm
        # print(observation, target_relative_position_norm)

        return observation



    def getTrackersObservation(self, n, noisy=[0, 0, 0, 0]):

        observation = np.array( [
                                self.trackers_true_positions[n][0],
                                self.trackers_true_positions[n][1],
                                # self.trackers_true_orientations[n], 
                                # nearest_boundary,
                                # observed_nearest_neighbor_distance, 
                                # observed_nearest_neighbor_bearing, 
                                # observed_nearest_neighbor_orientation,
                                # observed_target_distance, 
                                # observed_bearing
                                ] )
        # print(observation)

        return observation


    def rewardFunction(self, done, time, next_observation):
        
        reward_dict = dict()
        # time_reward = self.hyper_pars['time_importance'] * time
        position_check = lambda n: self.low_boundaries[0] < self.trackers_true_positions[n][0] < self.high_boundaries[0] and self.low_boundaries[1] < self.trackers_true_positions[1] < self.high_boundaries[1] 

        reward_dict = {n: 0 for n in self.trackers_network.keys()} # -1 if time < self.hyper_pars['max_time'] else -100
        # reward_dict = {n: 0 for n in self.trackers_network.keys()}
        # reward_dict = {n: -time/self.hyper_pars['max_time'] for n in self.trackers_network.keys()}

        # min_distance = np.infty
        # nearest_tracker = None
        # for i in self.trackers_network.keys():
        #     for j in set(self.trackers_network.keys()) - set([i]):
        #         distance_ij = self.__aggregateDistance__(alg.norm(self.trackers_true_positions[i] - self.trackers_true_positions[j], ord=2), self.neighbor_distance_binning)
        #         if distance_ij == 0: distance_ij = 0.1
        #         if distance_ij < min_distance:
        #             min_distance = distance_ij
        #             nearest_tracker = j

        #     if min_distance < Tracker.INTERACTION_RANGE:
        #             # reward_dict[i] += -1./(self.__aggregateDistance__(min_distance, self.neighbor_distance_binning) + 1)
        #             reward_dict[i] += -1./min_distance

        #     if next_observation[i][1] != 0: 
        #         reward_dict[i] += -1

        # min_neighbor_dist = np.inf
        # nearest_neighbor = None
        # for n in self.trackers_network.keys():
        #     for m in set(self.trackers_network.keys()) - set([n]):
        #         neighbor_distance = self.__computeDistance__(self.trackers_true_positions[n], self.trackers_true_positions[m])
        #         if neighbor_distance < min_neighbor_dist:
        #             min_neighbor_dist = neighbor_distance
        #             nearest_neighbor = m
        #     if min_neighbor_dist <= Tracker.INTERACTION_RANGE:
        #         reward_dict[n] += (np.cos(self.trackers_true_orientations[n] - self.trackers_true_orientations[m]) - 1) / min_neighbor_dist
            # if self.trackers_network[n].observation[2] != 5:
            #     reward_dict[n] -= 


        return reward_dict



    def reset(self):

        zones = [0,1,2,3]
        n_zones = len(zones)
        zone_xamplitude = self.xside/n_zones
        zone_yamplitude = self.yside/n_zones

        trackers_starting_zone_x = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        trackers_starting_zone_y = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        trackers_low_x = trackers_starting_zone_x*zone_xamplitude
        trackers_high_x = (trackers_starting_zone_x+1)*zone_xamplitude
        trackers_low_y = trackers_starting_zone_y*zone_yamplitude
        trackers_high_y = (trackers_starting_zone_y+1)*zone_yamplitude
        # self.trackers_starting_positions = self.__initializeAgentsPosition__(self.n_trackers, [trackers_low_x, trackers_low_y], [trackers_high_x, trackers_high_y])
        self.trackers_starting_positions = [np.array([0,0])]
        self.trackers_true_positions = [self.trackers_starting_positions[n] for n in range(self.n_trackers)]

        self.trackers_starting_orientations = self.__initializeAgentsOrientation__(self.n_trackers)
        self.trackers_true_orientations = [self.trackers_starting_orientations[n] for n in range(self.n_trackers)]

        self.trackers_starting_velocities = [ Tracker.SPEED * np.array( [np.cos(self.trackers_starting_orientations[n]), np.sin(self.trackers_starting_orientations[n])] ) for n in range(self.n_trackers)]
        self.trackers_true_velocities = [self.trackers_starting_velocities[n] for n in range(self.n_trackers)]

        for n in self.trackers_network.keys():
            self.trackers_network[n].observation = self.getTrackersObservation(n)



    def __periodicPositionCorrection__(self, new_position):
        
        for coordinate in [0,1]:
            if new_position[coordinate] < self.low_boundaries[coordinate]:
                new_position[coordinate] = self.high_boundaries[coordinate] - np.abs(new_position[coordinate] - self.low_boundaries[coordinate])
            elif new_position[coordinate] > self.high_boundaries[coordinate]:
                new_position[coordinate] = self.low_boundaries[coordinate] + np.abs(new_position[coordinate] - self.high_boundaries[coordinate])

        return new_position

        
    def step(self, action_dict, t):

        # PERIODIC BC on target
        # dx_target = self.current.generateCurrent()
        # new_target_position = self.target_position + self.current.generateCurrent()
        # self.target_position = self.__periodicPositionCorrection__(new_target_position)

        # INFINITE BC on target
        # self.target_position = self.target_position + self.current.generateCurrent()
        
        next_observation = [None]*self.n_trackers

        # IF BOUNCES ON BOUDARY, SKIP RL STEP (EBC ONLY)
        # for i in self.trackers_network.keys():

        #     if self.trackers_true_positions[i][0] < self.low_boundaries[0] or self.trackers_true_positions[i][0] > self.high_boundaries[0]:
        #         self.trackers_true_velocities[i][0] *= -1
        #         # CORRECT ORIENTATION
        #         if alg.norm( self.trackers_true_velocities[i] ) > 0:
        #             self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
        #                                         - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
        #         self.trackers_true_positions[i] = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
        #         is_rl_step = False
        #     elif self.trackers_true_positions[i][1] < self.low_boundaries[1] or self.trackers_true_positions[i][1] > self.high_boundaries[1]:
        #         self.trackers_true_velocities[i][1] *= -1
        #         # CORRECT ORIENTATION
        #         if alg.norm( self.trackers_true_velocities[i] ) > 0:
        #             self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
        #                                             - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
        #         self.trackers_true_positions[i] = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
        #         is_rl_step = False
            
        # if not is_rl_step:
        #     reward_dict = {n: 0 for n in self.trackers_network.keys()}

        #     for i in self.trackers_network.keys(): next_observation[i] = self.getTrackersObservation(i)
        #     done = False
    
        #     for i in self.trackers_network.keys():
        #         if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance: 
        #             done = True
        #             reward_dict[i] = 100

        #     return next_observation, reward_dict, done, is_rl_step


        for i in self.trackers_network.keys():
            
            # rotation_angle = Tracker.ACTIONS[action_dict[i]]
            movement = Tracker.ACTIONS[action_dict[i]] + np.random.multivariate_normal([0,0], [[0.01,0],[0,0.01]])
            # print(movement)
            # self.trackers_true_velocities[i] = np.dot(rotationMatrix(rotation_angle), self.trackers_true_velocities[i]) #+ np.random.normal(0, 0.15)

            # if alg.norm( self.trackers_true_velocities[i] ) > 0:
            #     self.trackers_true_orientations[i] = np.arctan2(self.trackers_true_velocities[i][1], self.trackers_true_velocities[i][0])

            # new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            new_position = self.trackers_true_positions[i] + movement
            if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
                self.trackers_true_positions[i] = deepcopy(new_position)

            # TRIVIAL BC: agent stops at boundary if it tries to go out of the search region
            # if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
            #     self.trackers_true_positions[i] = new_position

            # SOFT ELASTIC BC: agent bounces back softly when it tries to go out of the search region ("softly referring to the fact that it doesnt bounce 
            # back exacly at the physical boundary of the search region but whenever it steps outside of it the velocity is multiplied by -1")
            # if self.low_boundaries[0] - 5 < new_position[0]  < self.high_boundaries[0] + 5 and self.low_boundaries[1] - 5 < new_position[1] < self.high_boundaries[1] + 5:
            #     self.trackers_true_positions[i] = new_position
            # if new_position[0] < self.low_boundaries[0] - 5 or new_position[0] > self.high_boundaries[0] + 5:
            #     self.trackers_true_velocities[i] = np.dot(rotationMatrix(-rotation_angle), self.trackers_true_velocities[i])
            #     self.trackers_true_velocities[i][0] *= -1
            #     # CORRECT ORIENTATION
            #     if alg.norm( self.trackers_true_velocities[i] ) > 0:
            #         # self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
            #         #                                - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
            #         self.trackers_true_orientations[i] = np.arctan2(self.trackers_true_velocities[i][1], self.trackers_true_velocities[i][0])
            #     new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            # elif new_position[1] < self.low_boundaries[1] - 5 or new_position[1] > self.high_boundaries[1] + 5:
            #     self.trackers_true_velocities[i] = np.dot(rotationMatrix(-rotation_angle), self.trackers_true_velocities[i])
            #     self.trackers_true_velocities[i][1] *= -1
            #     # CORRECT ORIENTATION
            #     if alg.norm( self.trackers_true_velocities[i] ) > 0:
            #         # self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
            #         #                                - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )
            #         self.trackers_true_orientations[i] = np.arctan2(self.trackers_true_velocities[i][1], self.trackers_true_velocities[i][0])
            #     new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            # self.trackers_true_positions[i] = new_position

            # PERIODIC BC
            # self.trackers_true_positions[i] = self.__periodicPositionCorrection__(new_position)

            # INFINITE BOUNDARY
            # self.trackers_true_positions[i] = new_position

            # self.trackers_network[i].last_target_observation = self.trackers_network[i].target_observation

        for i in self.trackers_network.keys(): 
            next_observation[i] = self.getTrackersObservation(i)

        # status_of_search = [self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance for i in self.trackers_network.keys()]
        # done = np.sum( status_of_search ) > 0 
        # winners = [i for i, e in enumerate(status_of_search) if e == True]

        done = False
        rewards = self.rewardFunction(done, t, next_observation)
    
        for i in self.trackers_network.keys():
            if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance: 
                done = True
                rewards[i] = 1 # self.hyper_pars['max_time'] - t

        return next_observation, rewards, done



    def train(self, n_episodes, render=[], remember=False):

        self.reward_trajectories = {n: [0]*n_episodes for n in self.trackers_network.keys()}
        self.time_trajectories = [] 
        self.inefficient_coverage = [[0]*n_episodes for n in self.trackers_network.keys()]
        self.efficient_coverage = [[0]*n_episodes for n in self.trackers_network.keys()]
        render_episodes = {}
        long_episodes = {}

        step = 0
        succ_episodes = 0
        for episode in range(n_episodes):

            self.reset()

            # print(self.target_position)

            # if episode in render:
                # print(episode, '----------------------------------')
                # print(self.trackers_network[0].position)
            trackers_positions_history = [[self.trackers_true_positions[n]] for n in self.trackers_network.keys()]
            trackers_velocities_history = [[self.trackers_true_velocities[n]] for n in self.trackers_network.keys()]
            trackers_orientation_history = [[self.trackers_true_orientations[n]] for n in self.trackers_network.keys()]
            trackers_target_observation_history = [[1 if self.__computeDistance__(self.trackers_true_positions[n], self.target_position) <= Tracker.DETECTION_RANGE else 0] for n in self.trackers_network.keys()]
            trackers_action_history = [[] for n in self.trackers_network.keys()]
            coverage_overlappings = [[] for n in self.trackers_network.keys()]
            nonoverlapped_coverage = [[] for n in self.trackers_network.keys()]

            target_positions_history = [self.target_position]

            done = False
            time = 1
            while not done and time < self.hyper_pars['max_time']: 
                
                for i in self.trackers_network.keys():
                    # print(self.trackers_network[i].observation)
                    self.actions[i] = self.trackers_network[i].act(self.trackers_network[i].observation)
                    # if episode in render: 
                    trackers_action_history[i] += [self.actions[i]]
                    # if time == 1: self.trackers_network[i].observation[2] = self.actions[i]
                
                next_observation, rewards, done = self.step(self.actions, time)
                # print(next_observation)
                
                for i in self.trackers_network.keys():

                    self.trackers_network[i].learn( self.trackers_network[i].observation, self.actions[i], rewards[i], next_observation[i], done, step )
                    if step > self.trackers_network[i].init_lr_pars['start_update']:
                        if self.trackers_network[i].exploration > 1e-5:
                            self.trackers_network[i].exploration = self.trackers_network[i].init_lr_pars['exploration'] / ( (step / self.trackers_network[i].init_lr_pars['start_update']) ** self.trackers_network[i].update_rate )
                        else:
                            self.trackers_network[i].exploration = 0
                    self.reward_trajectories[i][episode] += rewards[i]

                for i in self.trackers_network.keys():
                    self.trackers_network[i].observation = next_observation[i]
              
                # if episode in render:
                    # print(done)
                for i in self.trackers_network.keys(): 
                    # print(self.__computeDistance__(self.trackers_true_positions[i], self.target_position))                               
                    trackers_positions_history[i] += [self.trackers_true_positions[i]] #[list(self.trackers_network[i].position)]
                    trackers_velocities_history[i] += [self.trackers_true_velocities[i]]
                    trackers_orientation_history[i] += [self.trackers_true_orientations[i]]
                    trackers_target_observation_history[i] += [1 if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= Tracker.DETECTION_RANGE else 0]
                    
                    # for j in set(self.trackers_network.keys()) - set([i]):
                    #     relative_position_ij = self.trackers_true_positions[i] - self.trackers_true_positions[j]
                    #     distance_ij = alg.norm(relative_position_ij, ord=2)
                    #     if distance_ij < Tracker.DETECTION_RANGE:
                    #         self.inefficient_coverage[i][episode] += 1
                    #     else:
                    #         self.efficient_coverage[i][episode] += 1

                target_positions_history += [self.target_position]

                # print(self.trackers_network[i].position)  
                # print(self.trackers_starting_positions)
                # print(self.trackers_starting_velocities)

                time += 1
                step += 1
                if(done): 
                    succ_episodes += 1
                
                print(episode, [len(self.trackers_network[0].BVset[action]) for action in self.trackers_network[0].actions.action_indeces], self.trackers_network[0].policy(self.trackers_network[0].observation), succ_episodes/(episode+1), end='\r')

            # if done: break

            if episode % 1000 == 0:
                print( 'ep = ', episode, ', step = ', step, ', t = ', time, 'ratio of successful episodes = {:.4f}'.format(succ_episodes/(episode+1)))

            self.time_trajectories += [time]
            for i in self.trackers_network.keys(): 
                self.inefficient_coverage[i][episode] /= time
                self.efficient_coverage[i][episode] /= time

            if (episode in render) or (time > 75 and episode > n_episodes-5e3):
                episode_dict = {'episode': episode,
                            'length': time,
                            'tracker_trajectory': deepcopy(trackers_positions_history),
                            'tracker_velocity': deepcopy(trackers_velocities_history),
                            'tracker_orientation': deepcopy(trackers_orientation_history),
                            'observation_history': deepcopy(trackers_target_observation_history),
                            'action_history': deepcopy(trackers_action_history),
                            'target_trajectory': deepcopy(target_positions_history)
            }
            if episode in render:
                render_episodes[episode] = deepcopy(episode_dict)

                np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/rewards_up_to_{}_{}'.format(episode, self.n_trackers), self.reward_trajectories)
                np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/performance/time_up_to_{}_{}'.format(episode, self.n_trackers), self.time_trajectories)

            elif (time > 75 and episode > n_episodes-n_episodes//20):
                long_episodes[episode] = deepcopy(episode_dict)
                
            del trackers_positions_history
            del trackers_velocities_history
            del trackers_orientation_history
            del trackers_target_observation_history
            del target_positions_history
            del trackers_action_history
        
        np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/episode/renderepisodes_{}_{}'.format(self.n_trackers), render_episodes)
        np.save('/home/marco/probabilistic_machine_learning/exam_project/trials/episode/longepisodes_{}_{}'.format(self.n_trackers), long_episodes)

            


    
        
    


        