import functions as f
def plot_beta_ols_degree(X, y, z, degree):
    # Looping through each degree
    for degree in degrees:
        # Creating design matrix
        X = f.create_design_matrix(x, y, degree)
        print(X.shape)
        
        # Calculating OLS beta
        #OLSbeta = np.linalg.inv(X.T @ X + reg_term*np.eye(X.shape[1])) @ X.T @ z
        OLSbeta = f.OLS_beta()

        
        print(degree, OLSbeta)
        # Calculating ztilde
        ztilde = X @ OLSbeta
        
        # Calculating MSE and R2
        mse = np.mean((z - ztilde)**2)
        r2 = 1 - np.sum((z - ztilde)**2) / np.sum((z - np.mean(z))**2)
        
        # Appending beta values and scores
        beta_values.append(OLSbeta)
        mse_scores.append(mse)
        r2_scores.append(r2)