import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.utils import resample


# ============================================================
#                  chunk_ridge_lasso_comparison
# ============================================================

def chunk_ridge_lasso_comparison():
    # Load and prepare data
    hitters = fetch_openml(name='boston_corrected', version=1, as_frame=True, parser='auto')
    df = hitters.frame.dropna(subset=['Salary'])
    X = pd.get_dummies(df.drop('Salary', axis=1), drop_first=True)
    y = df['Salary']
    
    # Train-test split
    np.random.seed(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    
    # Ridge regression
    np.random.seed(1)
    ridge_model = RidgeCV(cv=10).fit(X_train, y_train)
    yhat_ridge = ridge_model.predict(X_test)
    mse_r = np.mean((yhat_ridge - y_test)**2)
    
    # Lasso regression
    np.random.seed(1)
    lasso_model = LassoCV(cv=10).fit(X_train, y_train)
    yhat_lasso = lasso_model.predict(X_test)
    mse_l = np.mean((yhat_lasso - y_test)**2)
    
    print(f"Ridge MSE: {mse_r}")
    print(f"Lasso MSE: {mse_l}")
    
    return mse_r, mse_l







############################################################################################################



# ============================================================
#                  chunk_ridge_lasso_grid
# ============================================================

def chunk_ridge_lasso_grid(X, y):
    grid = np.logspace(10, -2, 100)
    
    np.random.seed(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    
    # Ridge regression
    np.random.seed(1)
    cv_ridge = RidgeCV(alphas=grid, cv=10).fit(X_train, y_train)
    bestlam_R = cv_ridge.alpha_
    ridge_mod = Ridge(alpha=bestlam_R).fit(X_train, y_train)
    yhat_R = ridge_mod.predict(X_test)
    mse_R = np.mean((yhat_R - y_test)**2)
    
    # Lasso regression
    np.random.seed(1)
    cv_lasso = LassoCV(alphas=grid, cv=10).fit(X_train, y_train)
    bestlam_L = cv_lasso.alpha_
    lasso_mod = Lasso(alpha=bestlam_L).fit(X_train, y_train)
    yhat_L = lasso_mod.predict(X_test)
    mse_L = np.mean((yhat_L - y_test)**2)
    
    print(f"Ridge MSE: {mse_R}")
    print(f"Lasso MSE: {mse_L}")
    
    return mse_R, mse_L








############################################################################################################


# ============================================================
#                  chunk_lasso_bootstrap
# ============================================================

def chunk_lasso_bootstrap(df):
    grid = np.logspace(10, -2, 100)
    MMSPE = np.zeros(len(grid))
    
    for i, alpha in enumerate(grid):
        MSPE = []
        for j in range(100):
            np.random.seed(j)
            train = resample(df, replace=True, random_state=j)
            test = df[~df.index.isin(train.index)]
            
            xtrain = pd.get_dummies(train.drop('Salary', axis=1), drop_first=True)
            ytrain = train['Salary']
            xtest = pd.get_dummies(test.drop('Salary', axis=1), drop_first=True)
            ytest = test['Salary']
            
            model = Lasso(alpha=alpha, max_iter=1000).fit(xtrain, ytrain)
            yhat = model.predict(xtest)
            MSPE.append(np.mean((yhat - ytest)**2))
        
        MMSPE[i] = np.mean(MSPE)
    
    min_mmspe = np.min(MMSPE)
    best_alpha = grid[np.argmin(MMSPE)]
    
    print(f"Min MMSPE: {min_mmspe}")
    print(f"Best lambda: {best_alpha}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(grid), MMSPE, 'o-', color='red', linewidth=3)
    plt.xlabel('log(lambda)')
    plt.ylabel('MMSPE')
    plt.show()
    
    return min_mmspe, best_alpha, MMSPE








############################################################################################################



# ============================================================
#                    chunk_lasso_coef
# ============================================================

def chunk_lasso_coef(grid, MMSPE, X, y, feature_names=None):
    best_alpha = grid[np.argmin(MMSPE)]
    model = Lasso(alpha=best_alpha, max_iter=1000).fit(X, y)
    
    coefficients = model.coef_
    intercept = model.intercept_
    nonzero_mask = coefficients != 0
    
    if feature_names is not None:
        nonzero_coef = pd.Series(coefficients[nonzero_mask], 
                               index=np.array(feature_names)[nonzero_mask])
    else:
        nonzero_coef = coefficients[nonzero_mask]
    
    print(f"Intercept: {intercept}")
    print(f"Non-zero coefficients:\n{nonzero_coef}")
    
    return intercept, nonzero_coef