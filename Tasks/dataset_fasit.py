import numpy as np
import functions as f
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

""" File showing how the current dataset should look so we use the same for every part"""

# Create data
np.random.seed(2024)
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
z = f.FrankeFunction(x, y)
z = z + np.random.normal(0, 0.1, z.shape)

N = 75
x = np.random.rand(N)
y = np.random.randn(N)
z = f.FrankeFunction(x,y) #Using x,y and z when doing regression
z = z + np.random.normal(0, 0.1, z.shape) #the noise was too high, tried sligtly less

# Polynomial degrees
degrees = np.arange(0, 7)
# Mesh for plotting in 3D
#xv, yv = np.meshgrid(x,y)
#z = FrankeFunction(xv, yv) #Add noise here
#z = z + np.random.randn(z.shape[0], z.shape[1])
degree = 0

# Train-test-split
X = f.create_design_matrix(x, y, degree)
#OLS_beta = f.OLS_beta(X, z)
#ztilde = f.z_predict(X, OLS_beta)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state = 42)

# Scale and center the data
scaler = StandardScaler(with_std=True, with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

