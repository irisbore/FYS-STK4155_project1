import numpy as np

# Franke function

def FrankeFunction(x: np.meshgrid,y: np.meshgrid) -> np.array:
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    #Noise term
    #term5 = np.random.normal(0, 0.1, len(x))
    return term1 + term2 + term3 + term4 #+ term5


# Design matrix
def create_design_matrix(x: np.array, y: np.array, p: int) -> np.array:
    X = np.zeros((len(x), int(1/2 * p* (p+3))))
    col_index = 0
    for i in range(p):
        for j in range(p-i):
            if i+j != 0:
                X[:, col_index] = x**i + y**j
                print(f"X[:, {col_index}], x**{i} + y**{j}")
                col_index += 1
    return X