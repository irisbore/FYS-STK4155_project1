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
import git
import sys
path_to_root = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(path_to_root+'src'+'/')
import functions as f
plt.style.use('seaborn-v0_8-whitegrid')


# Load the terrain data
terrain_data = imread(path_to_root+"/data/SRTM_data_Norway_1.tif")

# Define the x and y coordinates
#x = np.arange(0, terrain_data[2000:3601,:].shape[1], 10)
x = np.arange(0, terrain_data.shape[1], 10)
#y = np.arange(0, terrain_data[2000:3601,:].shape[0], 10)
y = np.arange(0, terrain_data.shape[0], 10)
# Create a meshgrid of the x and y coordinates
xv, yv = np.meshgrid(x, y)

# Sample z
#zv = terrain_data[2000:3601:10, ::10]
zv = terrain_data[::10, ::10]


degrees = np.arange(1, 15)

beta_values = []
mse_values = np.zeros(len(degrees))
R2_values = np.zeros(len(degrees))

for i, degree in enumerate(degrees): 
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X = poly_features.fit_transform(np.column_stack((x, y)))

    # Split the data into training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    # Scale and center the data
    X_train, X_test = f.scale_train_test(train = X_train, test = X_test)
    z_train, z_test = f.scale_train_test(train = z_train, test = z_test)

    #Calculating OLSbeta, ztilde, mse and R2
    OLSbeta = f.beta_OLS(X_train, z_train)
    ztilde = f.z_predict(X_test, OLSbeta)
    mse = f.mse(z_test, ztilde)
    R2 = f.r2(z_test, ztilde)

    #Adding the values to the arrays

    beta_values.append(OLSbeta)
    mse_values[i] = mse
    R2_values[i] = R2


ztilde = f.z_predict(X, OLSbeta)

heatmap = plt.pcolormesh(xv, yv, ztildev, cmap='viridis')

# Add color bar
plt.colorbar(heatmap)

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heatmap of sampled terrain data')
plt.gca().invert_yaxis()