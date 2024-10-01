
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
plt.style.use('seaborn-v0_8-whitegrid')
import git
import sys
sys.path.append("../")
path_to_root = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(path_to_root+'/src'+'/')
print(sys.path)
import functions as f

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

# Set best degree
degree = 4

lambda_values = np.logspace(-5, -3, 2)
lambda_value = 0.1

# Create polynomial features
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

mse_temp = np.zeros(len(lambda_values))

for j, lambda_ in enumerate(lambda_values):
    # Create and fit the linear regression model
    model = Lasso(alpha = lambda_values[j], tol = 1e-3, selection = 'random', precompute=True, fit_intercept=False, max_iter=10000) #selection = "random"
    model.fit(X_train, z_train)

    # Make predictions for training and test data
    z_train_pred = model.predict(X_train)
    z_test_pred = model.predict(X_test)

    mse_test = f.r2(z_test, z_test_pred)

    mse_temp[j] = mse_test

j = np.argmin(mse_temp)
print(lambda_values[j])

# Create and fit the linear regression model with the optimal lambda value
model = Lasso(alpha = lambda_value, tol = 1e-3, selection = 'random', precompute=True, fit_intercept=False, max_iter=10000)
model.fit(X_train, z_train)


# Predict z with whole dataset
X = X_scaler.transform(X)
beta_lasso = model.coef_
ztilde = f.z_predict(X, beta_lasso).reshape(-1,1)

ztilde = z_scaler.inverse_transform(ztilde)

ztilde_mesh = ztilde.reshape(zv.shape)

# Plot heatmap
aspect_ratio = xv.shape[1] / yv.shape[0]
plt.figure(figsize=(5, 5 / aspect_ratio))
heatmap = plt.pcolormesh(xv, yv, ztilde_mesh, cmap='viridis')

# Add color bar
plt.colorbar(heatmap)

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heatmap of Lasso regression on sampled terrain data')
plt.gca().invert_yaxis()
#f.save_to_results(filename = "lasso_terrain_heatmap.png")
plt.show()
