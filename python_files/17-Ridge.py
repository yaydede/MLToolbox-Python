import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split


# ============================================================
#                       chunk_ridge1
# ============================================================

def chunk_ridge1():
    hitters = fetch_openml(name='boston_corrected', version=1, as_frame=True, parser='auto')
    df = hitters.frame.dropna(subset=['Salary'])
    return df








################################################################################################################




# ============================================================
#                       chunk_ridge2
# ============================================================

def chunk_ridge2(df):
    X = pd.get_dummies(df.drop('Salary', axis=1), drop_first=True)
    y = df['Salary']
    return X, y








################################################################################################################



# ============================================================
#                       chunk_ridge3
# ============================================================

def chunk_ridge3(X, y):
    grid = np.logspace(10, -2, 100)
    model = Ridge(alpha=grid[0])
    models = [Ridge(alpha=alpha).fit(X, y) for alpha in grid]
    return models, grid








################################################################################################################


# ============================================================
#                       chunk_ridge4
# ============================================================

def chunk_ridge4(models, grid, X):
    coef_matrix = np.column_stack([model.coef_ for model in models])
    print(f"Coefficient matrix shape: {coef_matrix.shape}")
    print(f"Lambda values [20, 80]: {grid[[19, 79]]}")
    print(f"Coefficients for lambda [20, 80]:\n{coef_matrix[:, [19, 79]]}")
    return coef_matrix








################################################################################################################

# ============================================================
#                       chunk_ridge5
# ============================================================

def chunk_ridge5(X, y, s=50):
    model = Ridge(alpha=s).fit(X, y)
    coefficients = np.concatenate([[model.intercept_], model.coef_])
    return coefficients









################################################################################################################


# ============================================================
#                       chunk_ridge6
# ============================================================

def chunk_ridge6(df):
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
            
            model = Ridge(alpha=alpha).fit(xtrain, ytrain)
            yhat = model.predict(xtest)
            MSPE.append(np.mean((yhat - ytest)**2))
        
        MMSPE[i] = np.mean(MSPE)
    
    min_mmspe = np.min(MMSPE)
    best_alpha = grid[np.argmin(MMSPE)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(grid), MMSPE, 'o-', color='red', linewidth=3)
    plt.xlabel('log(lambda)')
    plt.ylabel('MMSPE')
    plt.show()
    
    return min_mmspe, best_alpha, MMSPE








################################################################################################################



# ============================================================
#                       chunk_ridge7
# ============================================================

def chunk_ridge7(grid, MMSPE, X, y):
    lambda_opt = grid[np.argmin(MMSPE)]
    model = Ridge(alpha=lambda_opt).fit(X, y)
    coeff = np.concatenate([[model.intercept_], model.coef_])
    print(f"Coefficients:\n{coeff}")
    return coeff, lambda_opt









################################################################################################################


# ============================================================
#                       chunk_ridge8
# ============================================================

def chunk_ridge8(df):
    MSPE = []
    
    for j in range(100):
        np.random.seed(j)
        train = resample(df, replace=True, random_state=j)
        test = df[~df.index.isin(train.index)]
        
        X_train = pd.get_dummies(train.drop('Salary', axis=1), drop_first=True)
        y_train = train['Salary']
        X_test = pd.get_dummies(test.drop('Salary', axis=1), drop_first=True)
        y_test = test['Salary']
        
        model = LinearRegression().fit(X_train, y_train)
        yhat = model.predict(X_test)
        MSPE.append(np.mean((yhat - y_test)**2))
    
    mean_mspe = np.mean(MSPE)
    print(f"Mean MSPE: {mean_mspe}")
    return mean_mspe, model








################################################################################################################



# ============================================================
#                       chunk_ridge9
# ============================================================

def chunk_ridge9(X, y):
    bestlam = []
    mse = []
    grid = np.logspace(10, -2, 100)
    
    for i in range(100):
        np.random.seed(i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
        
        # Finding optimal lambda via cross-validation
        cv_model = RidgeCV(alphas=grid, cv=10).fit(X_train, y_train)
        bestlam.append(cv_model.alpha_)
        
        # Predicting with optimal lambda
        ridge_mod = Ridge(alpha=cv_model.alpha_).fit(X_train, y_train)
        yhat = ridge_mod.predict(X_test)
        mse.append(np.mean((yhat - y_test)**2))
    
    print(f"Mean best lambda: {np.mean(bestlam)}")
    print(f"Mean MSE: {np.mean(mse)}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(bestlam, color='blue')
    ax1.set_title('Best Lambda')
    ax2.plot(mse, color='pink')
    ax2.set_title('MSE')
    plt.show()
    
    return np.mean(bestlam), np.mean(mse)









################################################################################################################



# ============================================================
#                      chunk_ridge10
# ============================================================

def chunk_ridge10(X, y):
    mse = []
    alphas = np.logspace(-6, 6, 100)
    
    for i in range(100):
        np.random.seed(i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
        
        cv_model = RidgeCV(alphas=alphas, cv=10).fit(X_train, y_train)
        yhat = cv_model.predict(X_test)
        mse.append(np.mean((yhat - y_test)**2))
    
    mean_mse = np.mean(mse)
    print(f"Mean MSE: {mean_mse}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(mse, color='pink')
    plt.title('MSE')
    plt.show()
    
    return mean_mse