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
import tensorflow as T
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

        self.catching_distance = 3

        self.position_binning = [math.floor(self.xside/self.catching_distance), math.floor(self.yside/self.catching_distance)]
        self.orientation_slices = 6
        self.orientation_binning = 2*self.orientation_slices + 1
        max_distance = Tracker.DETECTION_RANGE
        self.target_distance_binning = math.floor(max_distance/self.catching_distance)
        max_distance_communication = Tracker.INTRASWARM_INTERACTION_RANGE
        self.neighbor_distance_binning = math.floor(max_distance_communication/self.catching_distance)

        self.state_space_size = (self.orientation_binning, 
                                self.neighbor_distance_binning+1, 
                                self.orientation_binning, 
                                self.target_distance_binning+1, 
                                self.orientation_binning)

        self.trackers_starting_area = (self.xside/4, self.yside/4) 

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

    

    def __aggregateSpeed(self, speed):

        magnitude = alg.norm(speed, ord=2)
        speed_state = None

        for s in range(self.speed_binning):
            if s*Tracker.INCREASE_SPEED < magnitude < (s+1)*Tracker.INCREASE_SPEED:
                speed_state = s

        if speed_state is None:
            speed_state = self.speed_binning

        return speed_state

    

    def __aggregateOrientation__(self, orientation):

        if orientation is None: 
            return 0
        elif np.abs(orientation) == np.pi:
            return 2*self.orientation_slices

        orientation_state = None

        for a in range(1, self.orientation_slices+1):
            if orientation >= 0:
                if (a-1) * np.pi/self.orientation_slices <= np.abs(orientation) < a * np.pi/self.orientation_slices:
                    orientation_state = a
                    if orientation < 0: 
                        orientation_state *= -1
            else:
                if (a-1) * np.pi/self.orientation_slices < np.abs(orientation) <=  a * np.pi/self.orientation_slices:
                    orientation_state = a
                    if orientation < 0: 
                        orientation_state *= -1
        
        if orientation_state is None:
            return 0

        return orientation_state + (self.orientation_slices+1) if orientation_state < 0 else orientation_state + self.orientation_slices


    
    def __aggregateDistance__(self, distance, binning):
        
        distance_state = None

        for d in range(binning):
            if d*self.catching_distance <= distance < (d+1)*self.catching_distance:
                distance_state = d

        if distance_state is None:
            distance_state = binning

        return distance_state



    def __initializeAgentsPosition__(self, N, low, high):

        return [ np.random.uniform(low=low, high=high, size=(2,)) for _ in range(N) ]


        
    def __initializeAgentsVelocity__(self, n):

        # return [ np.random.random((2,)) for _ in range(n) ]
        return [ np.array([0, 0]) for _ in range(n) ]



    def __initializeAgentsOrientation__(self, n, group_orientation = np.pi/4):

        return [ np.random.uniform(-np.pi, np.pi, 1)[0] for _ in range(n) ]



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
        

        dx_rotated = np.dot(rotationMatrix(-v_orientation), dx)
        dx_rotated_ycomponent_sign = dx_rotated[1]/np.abs(dx_rotated[1])

        dx_normalized = dx/alg.norm(dx, ord=2)
        if alg.norm(v, ord=2) > 0: 
            v_normalized = v/alg.norm(v, ord=2)
        else:
            v_normalized = np.array( [np.cos(v_orientation), np.sin(v_orientation)] )

        return np.arccos( np.dot(v_normalized, dx_normalized) ) if dx_rotated_ycomponent_sign > 0 else -np.arccos( np.dot(v_normalized, dx_normalized) )
    

    def __normalizeObservation__(self, observation):

        tracker_position_norm = alg.norm(observation[0][:2], ord=2)
        if tracker_position_norm>0: observation[0][:2] /= tracker_position_norm
        target_relative_position_norm = alg.norm(observation[0][2:4], ord=2)
        if target_relative_position_norm>0: observation[0][2:4] /= target_relative_position_norm

        return observation



    def getTrackersObservation(self, n, noisy=[0, 0, 0, 0]):

        if self.trackers_true_orientations[n] != None:
            observed_orientation = self.__aggregateOrientation__(self.trackers_true_orientations[n] + np.random.normal(0, noisy[1]))
        else:
            observed_orientation = self.__aggregateOrientation__(self.trackers_true_orientations[n])

        tracker_target_distance = self.__computeDistance__(self.target_position, self.trackers_true_positions[n]) + np.random.normal(0, noisy[2])
        tracker_target_relative_position = self.target_position - self.trackers_true_positions[n]
        observed_target_distance = self.__aggregateDistance__(tracker_target_distance, self.target_distance_binning)
        bearing = None
        if tracker_target_distance < Tracker.DETECTION_RANGE:
            bearing = self.__computeBearing__(tracker_target_relative_position, self.trackers_true_velocities[n], self.trackers_true_orientations[n])
        if bearing !=None:
            bearing += np.random.normal(0, noisy[3])
            observed_bearing = self.__aggregateOrientation__(bearing)
        else:
            observed_bearing = self.__aggregateOrientation__(bearing)

        min_neighbor_dist = np.inf
        nearest_neighbor = None
        for m in set(self.trackers_network.keys()) - set([n]):
            neighbor_distance = self.__computeDistance__(self.trackers_true_positions[n], self.trackers_true_positions[m]) + np.random.normal(0, noisy[2])
            if neighbor_distance < min_neighbor_dist:
                min_neighbor_dist = neighbor_distance
                nearest_neighbor = m
        observed_nearest_neighbor_distance = self.__aggregateDistance__(min_neighbor_dist, self.neighbor_distance_binning)
        nearest_neighbor_bearing = None
        if min_neighbor_dist <= Tracker.DETECTION_RANGE:
            nearest_neighbor_relative_position = self.trackers_true_positions[nearest_neighbor] - self.trackers_true_positions[n]
            nearest_neighbor_bearing = self.__computeBearing__(nearest_neighbor_relative_position, self.trackers_true_velocities[n], self.trackers_true_orientations[n])
        observed_nearest_neighbor_bearing = self.__aggregateOrientation__(nearest_neighbor_bearing)

        observation = np.array( [observed_orientation, 
                                observed_nearest_neighbor_distance, 
                                observed_nearest_neighbor_bearing, 
                                observed_target_distance, 
                                observed_bearing] )

        return observation


    def rewardFunction(self, done, time):
        
        reward_dict = dict()
        reward_dict = {n: -1 if time < self.hyper_pars['max_time'] else -100 for n in self.trackers_network.keys()} # -1 if time < self.hyper_pars['max_time'] else -100

        return reward_dict



    def reset(self):

        zones = [0,1,2]
        n_zones = len(zones)
        zone_xamplitude = self.xside/n_zones
        zone_yamplitude = self.yside/n_zones
    
        target_starting_zone_x = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        target_starting_zone_y = np.random.choice(zones, 1, p=[1/n_zones]*n_zones)[0]
        target_low_x = target_starting_zone_x*zone_xamplitude
        target_high_x = (target_starting_zone_x+1)*zone_xamplitude
        target_low_y = target_starting_zone_y*zone_yamplitude
        target_high_y = (target_starting_zone_y+1)*zone_yamplitude
        self.target_position = self.__initializeAgentsPosition__(self.n_targets, [target_low_x, target_low_y], [target_high_x, target_high_y])[0]
        self.target.speed = self.current.generateCurrent()

        trackers_starting_zone_x = (n_zones-1) - target_starting_zone_x
        trackers_starting_zone_y = (n_zones-1) - target_starting_zone_y
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


        
    def step(self, action_dict, t):

        while True:
            dx_target = self.current.generateCurrent()
            new_target_position = self.target_position + self.current.generateCurrent()
            if self.low_boundaries[0] < new_target_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_target_position[1] < self.high_boundaries[1]:
                self.target_position = new_target_position
                break
                
        
        next_observation = [None]*self.n_trackers
        for i in self.trackers_network.keys():
            
            rotation_angle = Tracker.ACTIONS[action_dict[i]]

            self.trackers_true_velocities[i] = np.dot(rotationMatrix(rotation_angle), self.trackers_true_velocities[i])

            new_position = self.trackers_true_positions[i] + self.trackers_true_velocities[i]
            if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
                self.trackers_true_positions[i] = new_position

            if alg.norm( self.trackers_true_velocities[i] ) > 0:
                self.trackers_true_orientations[i] = np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) ) if self.trackers_true_velocities[i][1] > 0 else\
                                                   - np.arccos( self.trackers_true_velocities[i][0] / alg.norm( self.trackers_true_velocities[i], ord=2 ) )

            next_observation[i] = self.getTrackersObservation(i)

        done = False
        rewards = self.rewardFunction(done, t)

        crashed = []
    
        end_for_collision = False
        for i in self.trackers_network.keys():
            if self.n_trackers > 1:
                for j in set(self.trackers_network.keys()) - set([i]):
                    if self.__computeDistance__(self.trackers_true_positions[i], self.trackers_true_positions[j]) <= self.catching_distance:
                        end_for_collision = True
                        rewards[i] = rewards[j] = -100
            if self.__computeDistance__(self.trackers_true_positions[i], self.target_position) <= self.catching_distance: 
                done = True
                rewards[i] = 100
            
        return next_observation, rewards, done, end_for_collision



    def train(self, n_episodes, render=[], remember=False):

        self.reward_trajectories = {n: [0]*n_episodes for n in self.trackers_network.keys()}
        self.time_trajectories = [] 

        Na_t = np.zeros( (self.n_trackers ,len(Tracker.ACTIONS)) )
        Na_tstar = None

        step = 0
        succ_episodes = 0
        for episode in range(n_episodes):

            self.reset()

            if episode in render:
                trackers_positions_history = [[self.trackers_true_positions[n]] for n in self.trackers_network.keys()]
                trackers_velocities_history = [[self.trackers_true_velocities[n]] for n in self.trackers_network.keys()]
                trackers_orientation_history = [[self.trackers_true_orientations[n]] for n in self.trackers_network.keys()]
                trackers_target_observation_history = [[1 if self.trackers_network[n].observation[-2] < self.target_distance_binning else 0] for n in self.trackers_network.keys()]
                target_positions_history = [self.target_position]


            done = False
            end_for_collision = False
            time = 1
            while (not done and not end_for_collision) and time < self.hyper_pars['max_time']: #  or not end_for_collision
                
                for i in self.trackers_network.keys():
                    self.actions[i] = self.trackers_network[i].act(self.trackers_network[i].observation)
                    Na_t[ i,  self.actions[i] ] += 1

                
                next_observation, rewards, done, end_for_collision = self.step(self.actions, time)
                
                for i in self.trackers_network.keys():

                    if remember:

                        if len(self.trackers_network[i].memory) > self.trackers_network[i].replay_size:
                            pass

                    else:
                        pass

                    if step == self.trackers_network[i].lr_pars['start_update']: Na_tstar = deepcopy(Na_t)

                    if step > self.trackers_network[i].lr_pars['start_update']:
                        if self.trackers_network[i].lr_pars['eps'] > self.trackers_network[i].lr_pars['eps_min']:
                            self.trackers_network[i].lr_pars['eps'] = self.trackers_network[i].lr_pars['eps_0'] / ( (Na_t[ i, self.actions[i] ] / Na_tstar[ i, self.actions[i] ]) ** self.trackers_network[i].lr_pars['exp_eps'] )
                        else:
                            self.trackers_network[i].lr_pars['eps'] = 0

                        if self.trackers_network[i].lr_pars['alpha'] > self.trackers_network[i].lr_pars['alpha_min']:
                            self.trackers_network[i].lr_pars['alpha'] = self.trackers_network[i].lr_pars['alpha_0'] / ( (Na_t[ i, self.actions[i] ] / Na_tstar[ i, self.actions[i] ]) ** self.trackers_network[i].lr_pars['exp_alpha'] )

                    self.reward_trajectories[i][episode] += rewards[i]

                for i in self.trackers_network.keys():
                    self.trackers_network[i].observation = next_observation[i]
              
                if episode in render:
                    for i in self.trackers_network.keys():                                
                        trackers_positions_history[i] += [self.trackers_true_positions[i]]
                        trackers_velocities_history[i] += [self.trackers_true_velocities[i]]
                        trackers_orientation_history[i] += [self.trackers_true_orientations[i]]
                        trackers_target_observation_history[i] += [1 if self.trackers_network[i].observation[-2] < self.target_distance_binning else 0]
                    target_positions_history += [self.target_position]

                time += 1
                step += 1
                if(done): 
                    succ_episodes += 1
                
                print(episode, end='\r')


            if episode % 1000 == 0:
                print('ep = ', episode, ', step = ', step, ', t = ', time, 'successful episodes = ', succ_episodes, ', eps = ', [self.trackers_network[n].lr_pars['eps'] for n in self.trackers_network.keys()], ', alpha = ', [self.trackers_network[n].lr_pars['alpha'] for n in self.trackers_network.keys()])

            self.time_trajectories += [time]

            if episode in render:
                lr_pars = {'a': self.lr_pars['alpha_0'], 'expa': self.lr_pars['exp_alpha'], 'eps': self.lr_pars['eps_0'], 'expe': self.lr_pars['exp_eps'], 'p_angle': 2*self.orientation_slices}
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_positions_history_double_{}_{}'.format(lr_pars, episode), trackers_positions_history) 
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_velocities_history_double_{}_{}'.format(lr_pars, episode), trackers_velocities_history) 
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_orientation_history_double_{}_{}'.format(lr_pars, episode), trackers_orientation_history) 
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/trackers_target_observation_history_double_{}_{}'.format(lr_pars, episode), trackers_target_observation_history) 
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/target_positions_history_double_{}_{}'.format(lr_pars, episode), target_positions_history)
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/episodes/duration_double_{}_{}'.format(lr_pars, episode), time)

                del trackers_positions_history
                del trackers_velocities_history
                del trackers_orientation_history
                del trackers_target_observation_history
                del target_positions_history

                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/performances/rewards_up_to_{}_double_{}'.format(episode, lr_pars), self.reward_trajectories)
                np.save('/home/marco/active_object_tracking_modelling/constvel_trials/performances/time_up_to_{}_double_{}'.format(episode, lr_pars), self.time_trajectories)



        






        


        
    


        