import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline



######################################################


# ================================
#           chunk_o1
# ================================

def chunk_o1():
    np.random.seed(123)
    x_1 = np.random.normal(0, 1, 100)
    f = 1 + 2*x_1 - 2*(x_1**2) + 3*(x_1**3)
    y = f + np.random.normal(0, 8, 100)
    
    models = {}
    degrees = [1, 2, 3, 20]
    colors = ['pink', 'blue', 'green', 'red']
    labels = ['ols1', 'ols2', 'ols3', 'ols4']
    
    for i, degree in enumerate(degrees):
        pipe = Pipeline([('poly', PolynomialFeatures(degree)), ('linear', LinearRegression())])
        models[f'ols{i+1}'] = pipe.fit(x_1.reshape(-1, 1), y)
    
    order = np.argsort(x_1)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_1, y, color='darkgrey', alpha=0.7)
    
    for i, (model_name, model) in enumerate(models.items()):
        predictions = model.predict(x_1.reshape(-1, 1))
        plt.plot(x_1[order], predictions[order], color=colors[i], linewidth=1.5, label=labels[i])
    
    plt.legend(loc='lower right')
    plt.xlabel('x_1')
    plt.ylabel('y')
    plt.show()
    
    return models, x_1, y



#########################################################




# ================================
#           chunk_o2
# ================================

def chunk_o2(models, x_1, y):
    mspe_values = []
    model_names = ['ols1', 'ols2', 'ols3', 'ols4']
    
    for model_name in model_names:
        predictions = models[model_name].predict(x_1.reshape(-1, 1))
        mspe = np.mean((predictions - y)**2)
        mspe_values.append(mspe)
    
    mspe_df = pd.DataFrame(mspe_values, 
                          index=model_names, 
                          columns=["In-sample MSPE's"])
    
    return mspe_df




##############################################################


# ================================
#           chunk_o3
# ================================

def chunk_o3():
    def xfunc(n):
        np.random.seed(123)
        return np.random.normal(0, 1, n)
    
    def simmse(M, n, sigma, poldeg):
        x_1 = xfunc(n)
        
        # Containers
        MSPE = np.zeros(M)
        yhat = np.zeros((M, n))
        olscoef = np.zeros((M, poldeg + 1))
        ymat = np.zeros((M, n))
        
        # Loop for samples
        for i in range(M):
            f = 1 + 2*x_1 - 2*(x_1**2)  # DGM
            y = f + np.random.normal(0, sigma, n)
            
            # Estimator using raw polynomials
            poly_features = PolynomialFeatures(degree=poldeg, include_bias=True)
            X_poly = poly_features.fit_transform(x_1.reshape(-1, 1))
            ols = LinearRegression(fit_intercept=False)
            ols.fit(X_poly, y)
            
            olscoef[i, :] = ols.coef_
            yhat[i, :] = ols.predict(X_poly)
            MSPE[i] = np.mean((y - yhat[i, :])**2)
            ymat[i, :] = y
            
        return [MSPE, yhat, sigma, olscoef, f, ymat]
    
    # Running different models with different polynomial degrees
    output1 = simmse(2000, 100, 7, 1)
    output2 = simmse(2000, 100, 7, 2)  # True model
    output3 = simmse(2000, 100, 7, 5)
    output4 = simmse(2000, 100, 7, 20)
    
    # Table
    tab = np.zeros((4, 5))
    row_names = ["ols1", "ols2", "ols3", "ols4"]
    col_names = ["bias^2", "var(yhat)", "MSE", "var(eps)", "In-sample MSPE"]
    
    f = output1[4]
    outputs = [output1, output2, output3, output4]
    
    # Var(yhat)
    for i, output in enumerate(outputs):
        tab[i, 1] = np.mean([np.mean((col - np.mean(col))**2) for col in output[1].T])
    
    # Bias^2
    for i, output in enumerate(outputs):
        tab[i, 0] = np.mean((np.mean(output[1], axis=0) - f)**2)
    
    # MSE
    for i, output in enumerate(outputs):
        fmat = np.tile(f, (output[5].shape[0], 1))
        tab[i, 2] = np.mean(np.mean((fmat - output[1])**2, axis=0))
    
    # MSPE
    for i, output in enumerate(outputs):
        tab[i, 4] = np.mean(np.mean((output[5] - output[1])**2, axis=0))
    
    # Irreducible error - var(eps)
    for i, output in enumerate(outputs):
        tab[i, 3] = np.mean([np.mean((col - np.mean(col))**2) for col in output[5].T])
    
    # Create DataFrame
    tab_df = pd.DataFrame(np.round(tab, 4), 
                         index=row_names, 
                         columns=col_names)
    
    return tab_df, outputs


######################################################################


# ================================
#           chunk_o4
# ================================

def chunk_o4(tab_df, outputs):
    # New Table
    tabb = np.zeros((4, 3))
    row_names = ["ols1", "ols2", "ols3", "ols4"]
    col_names = ["Cov(yi, yhat)", "True MSPE", "TrueMSPE-Cov"]
    
    # COV - 2*mean(diag(cov(yhat, y)))
    for i, output in enumerate(outputs):
        cov_matrix = np.cov(output[1], output[5])
        n_obs = output[1].shape[1]
        # Extract diagonal elements corresponding to covariance between yhat and y
        diag_covs = [cov_matrix[j, j + n_obs] for j in range(n_obs)]
        tabb[i, 0] = 2 * np.mean(diag_covs)
    
    # True MSPE = MSE + var(eps)
    tab_values = tab_df.values
    tabb[:, 1] = tab_values[:, 2] + tab_values[:, 3]  # MSE + var(eps)
    
    # True MSPE - Cov
    tabb[:, 2] = tabb[:, 1] - tabb[:, 0]
    
    # Create new table DataFrame
    tabb_df = pd.DataFrame(np.round(tabb, 4), 
                          index=row_names, 
                          columns=col_names)
    
    # Combine tables
    combined_df = pd.concat([tab_df, tabb_df], axis=1)
    
    return combined_df