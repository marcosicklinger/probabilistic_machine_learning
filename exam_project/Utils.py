import numpy as np

def moving_average(data_set, periods=100, mode='same'):

    weights = np.ones(periods) / periods

    return np.convolve(data_set, weights, mode=mode)



def rotationMatrix(angle):
    
    matrix = np.array([[np.cos(angle), -np.sin(angle)] , [np.sin(angle), np.cos(angle)]])
    
    return matrix



def distanceFromNearestBoundaries(position, low, high):

    x_smaller_distance = np.min([np.abs(position[0] - low[0]), np.abs(position[0] - high[0])])
    y_smaller_distance = np.min([np.abs(position[1] - low[1]), np.abs(position[1] - high[1])])
    distances = np.array([x_smaller_distance, y_smaller_distance])

    return distances
    


class FeatureEmbedding:

    def __init__(self):

    #     self.observation = None
        pass

    # def updateObservation(self, observation):

    #     self.observation = observation



    def __call__(self, t):

        mean_feature_embedding = T.math.reduce_mean(t, axis=0)
        # if self.observation is not None:
        #     return T.concat( [mean_feature_embedding, self.observation], axis=0 )
        return T.stack([mean_feature_embedding])



class DiscreteActionSpace:

    def __init__(self, n):

        self.n_actions = n
        self.action_indeces = np.array( [i for i in range(self.n_actions)] )
        self.uniform_choice = np.array([1./self.n_actions for i in self.action_indeces])



    def sample(self, p):

        return np.random.choice( self.action_indeces, 1, p=p )[0]



    def getModelBestActions(self, model_prediction):
        # print(model_prediction)
        if all(model_prediction == model_prediction[0]):
            # print('here')
            best_action = np.random.choice(self.action_indeces, size=1, p=self.uniform_choice)[0]
            return np.array([1. if i==best_action else 0. for i in self.action_indeces])
        return (model_prediction == np.max( model_prediction ))

        
def SADistance(s1, a1, s2, a2):
    # print(s1, s2)
    s1_s2_L1Distance = np.linalg.norm(s1-s2, ord=1)
    # print(s1_s2_L1Distance)
    # a1_a2_L1Distance = 1 if a1 != a2 else 0
    SAL1distance = s1_s2_L1Distance #+ a1_a2_L1Distance

    return SAL1distance

def CoverigNumber(domain_dimensions, ball_radius):
    return np.ceil((domain_dimensions[1][0]-domain_dimensions[0][0])/(ball_radius*2))*np.ceil((domain_dimensions[1][1]-domain_dimensions[0][1])/(ball_radius*2))