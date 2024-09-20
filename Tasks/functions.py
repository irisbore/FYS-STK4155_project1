import numpy as np

# Franke function
def FrankeFunction(x: np.ndarray,y: np.ndarray) -> np.ndarray:
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4


# Design matrix
def create_design_matrix(x: np.ndarray, y: np.ndarray, n:int ) -> np.ndarray:
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
    
	return X 

# Calculate OLSbeta
def beta_OLS(X: np.ndarray, z: np.ndarray) -> np.ndarray:
	beta = np.linalg.pinv(X.T @ X) @ X.T @ z
	return beta

def beta_ridge(X: np.ndarray, z: np.ndarray, lamda: float) -> np.ndarray:
	I = np.eye(X.shape[1])
	betaRidge =  np.linalg.inv(X.T @ X + I*lamda) @ X.T @ z
	return betaRidge

# Predict z from a design matrix and a beta array
def z_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
	ztilde = X @ beta
	return ztilde
	
# Calculating MSE and R2
def mse(true: np.ndarray, pred) -> np.ndarray:
	mse = np.mean((true - pred)**2)
	return mse

def r2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
	r2 = 1 - (np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2))
	return r2