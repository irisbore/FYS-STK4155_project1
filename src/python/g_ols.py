import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.style.use('seaborn-v0_8-whitegrid')
import git
import sys
sys.path.append("../")
import functions as f
path_to_root = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(path_to_root+'src'+'/')

# Load the terrain data
terrain_data = imread(path_to_root+"/data/SRTM_data_Norway_1.tif")

# Define the x and y coordinates

x = np.arange(0, terrain_data.shape[1], 10)
y = np.arange(0, terrain_data.shape[0], 10)

# Create a meshgrid of the x and y coordinates
xv, yv = np.meshgrid(x, y)

# Sample z
zv = terrain_data[::10, ::10]

x = xv.flatten()
y = yv.flatten()
z = zv.flatten().reshape(-1, 1)

# Set the best polynomial degree
degree = 14

poly_features = PolynomialFeatures(degree=degree)
X = poly_features.fit_transform(np.column_stack((x, y)))

# Split the data into training and test data
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

# Scale and center the data
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

z_scaler = StandardScaler()
z_train = z_scaler.fit_transform(z_train)
z_test = z_scaler.transform(z_test)

#Calculating OLSbeta, ztilde, mse and R2
OLSbeta = f.beta_OLS(X_train, z_train)
ztilde = f.z_predict(X_test, OLSbeta)
mse = f.mse(z_test, ztilde)
R2 = f.r2(z_test, ztilde)

# Predict z with whole dataset
X = X_scaler.transform(X)
ztilde = f.z_predict(X, OLSbeta)

ztilde = z_scaler.inverse_transform(ztilde)

ztilde_mesh = ztilde.reshape(zv.shape)

heatmap = plt.pcolormesh(xv, yv, ztilde_mesh, cmap='viridis')

# Add color bar
plt.colorbar(heatmap)

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heatmap of sampled terrain data')
plt.gca().invert_yaxis()

plt.show()