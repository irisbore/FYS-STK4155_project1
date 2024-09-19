import numpy as np
from functions import FrankeFunction, create_design_matrix
# from sklearn.preprocessing import StandardScaler, 
#from sklearn. import train_test_split

# Create data
degree = 4
np.random.seed(2024)
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
xv, yv = np.meshgrid(x,y)
z = FrankeFunction(xv, yv) #Add noise here

x = np.ravel(xv) 
y = np.ravel(yv) 
z = np.ravel(z) 

# Train-test-split
X = create_design_matrix(x, y, degree)
X_train, X_test, z_train, z_test = train_test_split(X, z)

# Standardize
