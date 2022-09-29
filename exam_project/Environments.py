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
                instance=0
    ):
        self.instance = instance
        self.n_trackers = n_trackers
        self.n_targets = 1
        self.low_boundaries = low
        self.high_boundaries = high
        self.lr_pars = lr_pars
        self.hyper_pars = hyper_pars
        self.xside = np.abs(self.high_boundaries[0] - self.low_boundaries[0])
        self.yside = np.abs(self.high_boundaries[1] - self.low_boundaries[1])
        self.catching_distance = 0.15
        self.x_binning = math.ceil( np.abs(self.low_boundaries[0]-self.high_boundaries[0]) / 0.1 )
        self.y_binning = math.ceil( np.abs(self.low_boundaries[1]-self.high_boundaries[1]) / 0.1 )
        self.position_binning = [self.x_binning, self.y_binning]
        self.tracker_action_space_size = len(Tracker.ACTIONS)
        self.state_space_size = (
                                self.position_binning[0]+1,
                                self.position_binning[1]+1
        )
        self.target_position = np.array([1, 1])
        self.trackers_network = dict()
        for n in range(n_trackers):
            self.trackers_network[n] = Tracker(self.lr_pars, self.state_space_size) 
        self.actions = {n: None for n in self.trackers_network.keys()}

        self.current_directory = os.getcwd()
        self.name_instance_directory = str(self.lr_pars).replace("'", '').replace(" ", '').replace("{", '').replace("}",'').replace(",", '_').replace(":", '_').replace(".", '_')
        episode_directory = os.path.join(self.current_directory + '/trials/episode/', self.name_instance_directory)
        performance_directory = os.path.join(self.current_directory + '/trials/performance/', self.name_instance_directory)
        if not os.path.exists(episode_directory):
            os.makedirs(episode_directory)
        if not os.path.exists(performance_directory):
            os.makedirs(performance_directory)



    def __aggregatePosition__(self, position):

        pos_state = [None, None]
        
        for k in [0, 1]:
            for x in range(self.position_binning[k]):
                if self.low_boundaries[k] + x*self.catching_distance <= position[k] < self.low_boundaries[k] + (x+1)*self.catching_distance:
                    pos_state[k] = x
            if pos_state[k] is None:
                pos_state[k] = self.position_binning[k]

        return pos_state


    
    def __aggregateDistance__(self, distance, binning):
        
        distance_state = None

        for d in range(binning):
            if d*self.catching_distance <= distance < (d+1)*self.catching_distance:
                distance_state = d

        if distance_state is None:
            distance_state = binning

        return distance_state



    def getTrackersObservation(self, n, noisy=[0, 0, 0, 0]):

        observed_position = self.__aggregatePosition__(self.trackers_true_positions[n])
        observation = np.array( [
                                observed_position[0],
                                observed_position[1],
                                ] )
        return observation


    def rewardFunction(self, done, time, next_observation):
        
        reward_dict = dict()
        reward_dict = {n: 0 for n in self.trackers_network.keys()} # -1 if time < self.hyper_pars['max_time'] else -100
        return reward_dict



    def reset(self):

        self.trackers_starting_positions = [np.array([0,0])]
        self.trackers_true_positions = [self.trackers_starting_positions[n] for n in range(self.n_trackers)]
        for n in self.trackers_network.keys():

            self.trackers_network[n].observation = self.getTrackersObservation(n)


        
    def step(self, action_dict, t):

        next_observation = [None]*self.n_trackers



        for i in self.trackers_network.keys():
            
            movement = Tracker.ACTIONS[action_dict[i]] + np.random.multivariate_normal([0,0], [[0.003,0],[0,0.003]])
            new_position = self.trackers_true_positions[i] + movement
            if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
                self.trackers_true_positions[i] = new_position

        for i in self.trackers_network.keys(): 
            next_observation[i] = self.getTrackersObservation(i)

        done = False
        rewards = self.rewardFunction(done, t, next_observation)
    
        for i in self.trackers_network.keys():
            if np.linalg.norm(self.trackers_true_positions[i] - self.target_position, ord=2) <= self.catching_distance: 
                done = True
                rewards[i] = 1 

        return next_observation, rewards, done



    def train(self, n_episodes, render=[]):

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

            trackers_positions_history = [[self.trackers_true_positions[n]] for n in self.trackers_network.keys()]
            trackers_action_history = [[] for n in self.trackers_network.keys()]
            target_positions_history = [self.target_position]

            done = False
            time = 1
            while not done and time < self.hyper_pars['max_time']: 
                
                for i in self.trackers_network.keys():
                    self.actions[i] = self.trackers_network[i].act(self.trackers_network[i].observation)
                    trackers_action_history[i] += [self.actions[i]]

                next_observation, rewards, done = self.step(self.actions, time)
        
                for i in self.trackers_network.keys():
                   
                    self.trackers_network[i].learn( self.trackers_network[i].observation, self.actions[i], rewards[i], next_observation[i], done, step )

                    if step == self.trackers_network[i].lr_pars['start_update']: 
                        Na_tstar = deepcopy(Na_t)
                        Mao_tstar = deepcopy(Mao_t)

                    if step == self.lr_pars['cut']:
                        self.trackers_network[i].lr_pars['alpha_0'] *= 0.05

                    if step > self.trackers_network[i].lr_pars['start_update'] and episode < n_episodes - n_episodes//20:
                        if self.trackers_network[i].lr_pars['eps'] > self.trackers_network[i].lr_pars['eps_min']:
                            self.trackers_network[i].lr_pars['eps'] = self.trackers_network[i].lr_pars['eps_0'] / ( (step / self.trackers_network[i].lr_pars['start_update']) ** self.trackers_network[i].lr_pars['exp_eps'] )
                        else:
                            self.trackers_network[i].lr_pars['eps'] = 0

                        if self.trackers_network[i].lr_pars['alpha'] > self.trackers_network[i].lr_pars['alpha_min']:
                            self.trackers_network[i].lr_pars['alpha'] = self.trackers_network[i].lr_pars['alpha_0'] / ( (step / self.trackers_network[i].lr_pars['start_update']) ** self.trackers_network[i].lr_pars['exp_alpha'] )     
                    elif episode > n_episodes - n_episodes//20:
                        self.trackers_network[i].lr_pars['eps'] = 0

                    self.reward_trajectories[i][episode] += rewards[i]

                for i in self.trackers_network.keys():
                    self.trackers_network[i].observation = next_observation[i]
              
                for i in self.trackers_network.keys(): 
                    trackers_positions_history[i] += [self.trackers_true_positions[i]] 

                target_positions_history += [self.target_position]

                time += 1
                step += 1
                if(done): 
                    succ_episodes += 1

            if episode % 500 == 0:
                print('ep = ', episode, ', step = ', step, ', t = ', time, 'ratio of successful episodes = {:.2f}'.format(succ_episodes/(episode+1)), ', eps = ', [self.trackers_network[n].lr_pars['eps'] for n in self.trackers_network.keys()], ', alpha = ', [self.trackers_network[n].lr_pars['alpha'] for n in self.trackers_network.keys()])

            self.time_trajectories += [time]

            if (episode in render) or (time > 75 and episode > n_episodes-n_episodes//20):
                episode_dict = {'episode': episode,
                            'length': time,
                            'tracker_trajectory': deepcopy(trackers_positions_history),
                            'action_history': deepcopy(trackers_action_history),
            }
            if episode in render:
                render_episodes[episode] = deepcopy(episode_dict)

                np.save(self.current_directory +'/trials/performance/{}/rewards_up_to_{}_{}_{}_discrete'.format(self.name_instance_directory,episode, self.n_trackers,self.instance), self.reward_trajectories)
                np.save(self.current_directory +'/trials/performance/{}/time_up_to_{}_{}_{}_discrete'.format(self.name_instance_directory,episode, self.n_trackers,self.instance), self.time_trajectories)


            elif (time > 75 and episode > n_episodes-n_episodes//20):
                long_episodes[episode] = deepcopy(episode_dict)
                
            del trackers_positions_history
            del trackers_action_history
        
        np.save(self.current_directory +'/trials/episode/{}/renderepisodes_{}_{}_discrete'.format(self.name_instance_directory, self.n_trackers, self.instance), render_episodes)
        np.save(self.current_directory +'/trials/episode/{}/longepisodes_{}_{}_discrete'.format(self.name_instance_directory, self.n_trackers, self.instance), long_episodes)
            
            


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
                instance = 0

    ):
        self.instance = instance
        self.state_dimensions = 2
        self.lr_pars = lr_pars
        self.n_trackers = n_trackers
        self.n_targets = 1
        self.low_boundaries = low
        self.high_boundaries = high
        self.hyper_pars = hyper_pars
        self.catching_distance = 0.15
        self.target_position = np.array([1, 1])
        self.trackers_network = dict()
        for n in range(n_trackers):
            self.trackers_network[n] = DGPQTracker(self.lr_pars, self.state_dimensions, self.low_boundaries, self.high_boundaries)
        self.actions = {n: None for n in self.trackers_network.keys()}

        self.current_directory = os.getcwd()
        self.name_instance_directory = str(self.lr_pars).replace("'", '').replace(" ", '').replace("{", '').replace("}",'').replace(",", '_').replace(":", '_').replace(".", '_')
        episode_directory = os.path.join(self.current_directory + '/trials/episode/', self.name_instance_directory)
        performance_directory = os.path.join(self.current_directory + '/trials/performance/', self.name_instance_directory)
        if not os.path.exists(episode_directory):
            os.makedirs(episode_directory)
        if not os.path.exists(performance_directory):
            os.makedirs(performance_directory)



    def getTrackersObservation(self, n, noisy=[0, 0, 0, 0]):

        observation = np.array( [
                                self.trackers_true_positions[n][0],
                                self.trackers_true_positions[n][1],
                                ] )
        return observation


    def rewardFunction(self, done, time, next_observation):
        
        reward_dict = dict()
        position_check = lambda n: self.low_boundaries[0] < self.trackers_true_positions[n][0] < self.high_boundaries[0] and self.low_boundaries[1] < self.trackers_true_positions[1] < self.high_boundaries[1] 
        reward_dict = {n: 0 for n in self.trackers_network.keys()} 
        return reward_dict



    def reset(self):

        self.trackers_starting_positions = [np.array([0,0])]
        self.trackers_true_positions = [self.trackers_starting_positions[n] for n in range(self.n_trackers)]

        for n in self.trackers_network.keys():
            self.trackers_network[n].observation = self.getTrackersObservation(n)


        
    def step(self, action_dict, t):
        
        next_observation = [None]*self.n_trackers

        for i in self.trackers_network.keys():
            
            movement = Tracker.ACTIONS[action_dict[i]] + np.random.multivariate_normal([0,0], [[0.003,0],[0,0.003]])
            new_position = self.trackers_true_positions[i] + movement
            if self.low_boundaries[0] < new_position[0] < self.high_boundaries[0] and self.low_boundaries[1] < new_position[1] < self.high_boundaries[1]:
                self.trackers_true_positions[i] = deepcopy(new_position)

        for i in self.trackers_network.keys(): 
            next_observation[i] = self.getTrackersObservation(i)

        done = False
        rewards = self.rewardFunction(done, t, next_observation)
    
        for i in self.trackers_network.keys():
            if np.linalg.norm(self.trackers_true_positions[i] - self.target_position, ord=2) <= self.catching_distance: 
                done = True
                rewards[i] = 1 

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
            trackers_positions_history = [[self.trackers_true_positions[n]] for n in self.trackers_network.keys()]
            trackers_action_history = [[] for n in self.trackers_network.keys()]

            target_positions_history = [self.target_position]

            done = False
            time = 1
            while not done and time < self.hyper_pars['max_time']: 
                
                for i in self.trackers_network.keys():
                    self.actions[i] = self.trackers_network[i].act(self.trackers_network[i].observation)
                    trackers_action_history[i] += [self.actions[i]]
                
                next_observation, rewards, done = self.step(self.actions, time)
                
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

                for i in self.trackers_network.keys(): 
                    trackers_positions_history[i] += [self.trackers_true_positions[i]]
                target_positions_history += [self.target_position]

                time += 1
                step += 1
                if(done): 
                    succ_episodes += 1
                
                print(episode, [len(self.trackers_network[0].BVset[action]) for action in self.trackers_network[0].actions.action_indeces], self.trackers_network[0].policy(self.trackers_network[0].observation), succ_episodes/(episode+1), end='\r')

            if episode % 1000 == 0:
                print('instance', self.instance, 'ep = ', episode, ', step = ', step, ', t = ', time, 'exploration = ', self.trackers_network[0].exploration, 'ratio of successful episodes = {:.4f}'.format(succ_episodes/(episode+1)))

            self.time_trajectories += [time]
            for i in self.trackers_network.keys(): 
                self.inefficient_coverage[i][episode] /= time
                self.efficient_coverage[i][episode] /= time

            if (episode in render) or (time > 75 and episode > n_episodes-n_episodes//20):
                episode_dict = {'episode': episode,
                            'length': time,
                            'tracker_trajectory': deepcopy(trackers_positions_history),
                            'action_history': deepcopy(trackers_action_history)
            }
            if episode in render:
                render_episodes[episode] = deepcopy(episode_dict)

                np.save(self.current_directory +'/trials/performance/{}/rewards_up_to_{}_{}_{}'.format(self.name_instance_directory,episode, self.n_trackers,self.instance), self.reward_trajectories)
                np.save(self.current_directory +'/trials/performance/{}/time_up_to_{}_{}_{}'.format(self.name_instance_directory,episode, self.n_trackers,self.instance), self.time_trajectories)
                np.save(self.current_directory +'/trials/performance/{}/BV_{}_{}_{}'.format(self.name_instance_directory,episode, self.n_trackers,self.instance), getattr(self.trackers_network[0],'BVset'))


            elif (time > 75 and episode > n_episodes-n_episodes//20):
                long_episodes[episode] = deepcopy(episode_dict)
                
            del trackers_positions_history
            del trackers_action_history
        
        np.save(self.current_directory +'/trials/episode/{}/renderepisodes_{}_{}'.format(self.name_instance_directory, self.n_trackers, self.instance), render_episodes)
        np.save(self.current_directory +'/trials/episode/{}/longepisodes_{}_{}'.format(self.name_instance_directory, self.n_trackers, self.instance), long_episodes)

            


    
        
    


        