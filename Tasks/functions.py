import numpy as np

# Franke function

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    #Noise term
    term5 = 0 #np.random.normal(0, 0.1, len(x))
    return term1 + term2 + term3 + term4 + term5

# Design matrix
def create_design_matrix(x, y, degree):
    X = np.zeros((len(x), degree+1))
    if degree == 0:
        X[:, 0] = x
    elif degree == 1:
        X[:, 0] = x
        X[:, 1] = y
    elif degree == 2:
        X[:, 0] = x
        X[:, 1] = y
        X[:, 2] = x@x.T
    elif degree == 3:
        X[:, 0] = x
        X[:, 1] = y
        X[:, 2] = x@x.T
        X[:, 3] = y@y.T
    elif degree == 4:
        X[:, 0] = x
        X[:, 1] = y
        X[:, 2] = x@x.T
        X[:, 3] = y@y.T
        X[:, 4] = x@y.T
    else:
        print("Degree not implemented")
    return X