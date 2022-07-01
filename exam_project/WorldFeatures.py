from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import threading
import random
from Utils import *
import math
import numpy as np
from matplotlib import animation
from matplotlib.animation import PillowWriter 

class RandomCurrentVelocity:

    def __init__(self, low, high):

        self.intensity_lower_bound = low
        self.intensity_upper_bound = high

    def generateCurrent(self):

        return (self.intensity_upper_bound - self.intensity_lower_bound) * np.random.random( (2,) ) + self.intensity_lower_bound










