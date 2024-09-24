import numpy as np
import functions as f
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

""" File creating dataset so we use the same throughout the project"""

def load_normal_data(N_samples: int= 100, noise:float = 0.1, seed:int = 42):
    np.random.seed(seed)
    x = np.random.rand(N_samples)
    y = np.random.randn(N_samples)
    z = f.FrankeFunction(x,y) 
    z = z + np.random.normal(0, 0.1, z.shape) 
    return x, y, z


def load_range_data(step_size: float= 0.05, noise:float = 0.1, seed:int = 42):
    np.random.seed(seed)
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)
    z = f.FrankeFunction(x,y) 
    z = z + np.random.normal(0, 0.1, z.shape) 
    return x, y, z