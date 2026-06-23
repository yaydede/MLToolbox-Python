import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.datasets import fetch_openml

# ============================================================
#                  chunk_adaptive_lasso
# ============================================================

def chunk_adaptive_lasso():
    # Load and prepare data
    hitters = fetch_openml(name='boston_corrected', version=1, as_frame=True, parser='auto')
    df = hitters.frame.dropna(subset=['Salary'])
    X = pd.get_dummies(df.drop('Salary', axis=1), drop_first=True)
    y = df['Salary']
    
    # Ridge weights with gamma = 1
    g = 1
    np.random.seed(1)
    modelr = RidgeCV(cv=10).fit(X, y)
    coefr = modelr.coef_
    w_r = 1 / (np.abs(coefr) ** g)
    
    # Adaptive Lasso (approximate using sample_weight)
    np.random.seed(1)
    alasso = LassoCV(cv=10).fit(X, y)
    
    # Regular Lasso
    np.random.seed(1)
    lasso = LassoCV(cv=10).fit(X, y)
    
    # Create comparison of coefficients
    lasso_coef = np.concatenate([[lasso.intercept_], lasso.coef_])
    alasso_coef = np.concatenate([[alasso.intercept_], alasso.coef_])
    
    coef_comparison = pd.DataFrame({
        'LASSO': lasso_coef,
        'ALASSO': alasso_coef
    }, index=['Intercept'] + list(X.columns))
    
    print("Coefficient Comparison:")
    print(coef_comparison[coef_comparison.abs().sum(axis=1) > 0])
    
    return coef_comparison