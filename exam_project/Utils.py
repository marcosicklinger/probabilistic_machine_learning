import keras as K
import tensorflow as T
import numpy as np

def moving_average(data_set, periods=100, mode='same'):

    weights = np.ones(periods) / periods

    return np.convolve(data_set, weights, mode=mode)



def rotationMatrix(angle):
    
    matrix = np.array([[np.cos(angle), -np.sin(angle)] , [np.sin(angle), np.cos(angle)]])
    
    return matrix



class DiscreteActionSpace:

    def __init__(self, n):

        self.n_actions = n
        self.action_indeces = np.array( [i for i in range(self.n_actions)] )



    def sample(self, p):

        return np.random.choice( self.action_indeces, 1, p=p )[0]



    def getModelBestActions(self, model_prediction):

        return (model_prediction == np.max( model_prediction ))

        
