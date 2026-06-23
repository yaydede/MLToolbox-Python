import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import lasso_path


# ================================
#           chunk_dgp
# ================================
def chunk_dgp(N, Beta):
    """
    Generates data for linear regression simulation.
    
    Args:
        N (int): Number of observations
        Beta (array-like): Coefficient vector
    
    Returns:
        tuple: (y, X) where y is response vector and X is design matrix
    """
    p = len(Beta)
    Beta = np.array(Beta)
    
    X = np.random.randn(N, p)
    u = np.random.randn(N, 1)
    y = X @ Beta.reshape(-1, 1) + u
    
    return y, X

# Example usage
N = 100
Beta = [1, 1, 0, 0]

np.random.seed(148)
y, X = chunk_dgp(N, Beta)








#################################################################################################################


# ================================
#          chunk_lasso
# ================================
def chunk_lasso(X, y, seed=432):
    """
    Fits LASSO regression with regularization path.
    
    Args:
        X (array): Design matrix
        y (array): Response vector
        seed (int): Random seed
    
    Returns:
        ndarray: Matrix with coefficients and lambda values
    """
    np.random.seed(seed)
    
    # Flatten y if it's 2D
    y_flat = y.flatten() if y.ndim > 1 else y
    
    # Compute LASSO path (equivalent to glmnet)
    alphas, coefs, _ = lasso_path(X, y_flat, fit_intercept=False)
    
    # Create S_matrix: transpose coefficients and add lambda column
    S_matrix = np.column_stack([coefs.T, alphas])
    
    # Select specific rows (R uses 1-based indexing, Python uses 0-based)
    selected_rows = np.concatenate([
        np.arange(0, 8),      # R: 1:8
        np.arange(24, 30),    # R: 25:30
        np.arange(54, 60)     # R: 55:60
    ])
    
    # Filter rows that exist in the matrix
    valid_rows = selected_rows[selected_rows < S_matrix.shape[0]]
    
    return S_matrix[valid_rows, :]

# Example usage (assuming X, y from previous chunk)
# result = chunk_lasso(X, y)
# print(result)









#################################################################################################################


# ================================
#        chunk_bic_calculation
# ================================
def chunk_bic_calculation(X, y, N, seed=432):
    """
    Calculates BIC for LASSO regression across regularization path.
    
    Args:
        X (array): Design matrix
        y (array): Response vector
        N (int): Number of observations
        seed (int): Random seed
    
    Returns:
        tuple: (y_hat, SSE, BIC) predictions, sum of squared errors, and BIC values
    """
    np.random.seed(seed)
    
    # Flatten y if it's 2D
    y_flat = y.flatten() if y.ndim > 1 else y
    
    # Compute LASSO path
    alphas, coefs, _ = lasso_path(X, y_flat, fit_intercept=False)
    
    # Predict for each lambda (equivalent to predict.glmnet)
    y_hat = X @ coefs
    
    # Calculate SSE for each lambda
    SSE = np.sum((y_hat - y_flat.reshape(-1, 1)) ** 2, axis=0)
    
    # Count non-zero coefficients for each lambda
    nz = np.sum(coefs != 0, axis=0)
    
    # Calculate BIC
    BIC = np.log(SSE) + (np.log(N) / N) * nz
    
    return y_hat, SSE, BIC

# Example usage (assuming X, y, N from previous chunks)
# y_hat, SSE, BIC = chunk_bic_calculation(X, y, N)
# print(f"y_hat shape: {y_hat.shape}")
# print(f"BIC values: {BIC}")








#################################################################################################################


# ================================
#       chunk_lasso_optimal
# ================================
def chunk_lasso_optimal(X, y, N, seed=432):
    """
    Finds optimal LASSO coefficients based on minimum BIC.
    
    Args:
        X (array): Design matrix
        y (array): Response vector
        N (int): Number of observations
        seed (int): Random seed
    
    Returns:
        ndarray: Optimal LASSO coefficients
    """
    np.random.seed(seed)
    
    # Flatten y if it's 2D
    y_flat = y.flatten() if y.ndim > 1 else y
    
    # Compute LASSO path
    alphas, coefs, _ = lasso_path(X, y_flat, fit_intercept=False)
    
    # Calculate BIC
    y_hat = X @ coefs
    SSE = np.sum((y_hat - y_flat.reshape(-1, 1)) ** 2, axis=0)
    nz = np.sum(coefs != 0, axis=0)
    BIC = np.log(SSE) + (np.log(N) / N) * nz
    
    # Find coefficients at minimum BIC
    min_bic_idx = np.argmin(BIC)
    beta_lasso = coefs[:, min_bic_idx]
    
    return beta_lasso

# Example usage (assuming X, y, N from previous chunks)
# beta_lasso = chunk_lasso_optimal(X, y, N)
# print(beta_lasso)









#################################################################################################################


# ================================
#       chunk_l2_distance
# ================================
def chunk_l2_distance(beta_lasso, Beta):
    """
    Calculates L2 distance between estimated and true coefficients.
    
    Args:
        beta_lasso (array): Estimated LASSO coefficients
        Beta (array): True coefficient vector
    
    Returns:
        float: L2 distance (Euclidean norm of difference)
    """
    Beta = np.array(Beta)
    beta_lasso = np.array(beta_lasso)
    
    l_2 = np.sqrt(np.sum((beta_lasso - Beta) ** 2))
    
    return l_2

# Example usage (assuming beta_lasso from previous chunk and Beta = [1, 1, 0, 0])
# l_2 = chunk_l2_distance(beta_lasso, Beta)
# print(l_2)









#################################################################################################################


# ================================
#           chunk_mcs
# ================================
def chunk_mcs(mc, N, Beta):
    """
    Monte Carlo simulation for LASSO performance evaluation.
    
    Args:
        mc (int): Number of Monte Carlo iterations
        N (int): Number of observations per simulation
        Beta (array): True coefficient vector
    
    Returns:
        tuple: (mcmat, beta_lasso_mat) containing performance metrics and estimated coefficients
    """
    Beta = np.array(Beta)
    p = len(Beta)
    
    mcmat = np.zeros((mc, 3))
    beta_lasso_mat = np.zeros((mc, p))
    
    for i in range(mc):
        # Data generation (dgp equivalent)
        np.random.seed(i + 1)  # R uses 1-based seeds
        X = np.random.randn(N, p)
        u = np.random.randn(N, 1)
        y = X @ Beta.reshape(-1, 1) + u
        y_flat = y.flatten()
        
        # LASSO fitting
        np.random.seed(i + 1)
        alphas, coefs, _ = lasso_path(X, y_flat, fit_intercept=False)
        
        # Predictions and BIC calculation
        y_hat = X @ coefs
        SSE = np.sum((y_hat - y_flat.reshape(-1, 1)) ** 2, axis=0)
        nz = np.sum(coefs != 0, axis=0)
        BIC = np.log(SSE) + (np.log(N) / N) * nz
        
        # Optimal coefficients
        min_bic_idx = np.argmin(BIC)
        beta_lasso = coefs[:, min_bic_idx]
        
        # Performance metrics
        nonz_beta = np.sum(Beta == 0)  # Number of true zeros
        nonz_beta_hat = np.sum(beta_lasso == 0)  # Number of estimated zeros
        
        mcmat[i, 0] = np.sqrt(np.sum((beta_lasso - Beta) ** 2))  # L2 distance
        mcmat[i, 1] = 1 if nonz_beta == nonz_beta_hat else 0     # Correct model selection
        mcmat[i, 2] = np.sum(beta_lasso != 0)                    # Number of non-zero estimates
        
        beta_lasso_mat[i, :] = beta_lasso
    
    return mcmat, beta_lasso_mat

# Example usage
# mc = 100
# N = 100
# Beta = [1, 1, 0, 0]
# mcmat, beta_lasso_mat = chunk_mcs(mc, N, Beta)








#################################################################################################################


# ================================
#         chunk_sum_l2
# ================================
def chunk_sum_l2(MC_performance):
    """
    Calculates sum of L2 distances from Monte Carlo simulation results.
    
    Args:
        MC_performance (array): Performance matrix from Monte Carlo simulation
    
    Returns:
        float: Sum of L2 distances (first column)
    """
    return np.sum(MC_performance[:, 0])

# Example usage (assuming MC_performance from chunk_mcs)
# mcmat, beta_lasso_mat = chunk_mcs(mc, N, Beta)
# total_l2_distance = chunk_sum_l2(mcmat)
# print(total_l2_distance)








#################################################################################################################


# ================================
#          chunk_mcsA
# ================================
def chunk_mcsA(mc, N, Beta):
    """
    Monte Carlo simulation for Adaptive LASSO performance evaluation.
    
    Args:
        mc (int): Number of Monte Carlo iterations
        N (int): Number of observations per simulation
        Beta (array): True coefficient vector
    
    Returns:
        tuple: (mcmat, beta_lasso_mat) containing performance metrics and estimated coefficients
    """
    Beta = np.array(Beta)
    p = len(Beta)
    
    mcmat = np.zeros((mc, 3))
    beta_lasso_mat = np.zeros((mc, p))
    
    for i in range(mc):
        # Data generation (dgp equivalent)
        np.random.seed(i + 1)
        X = np.random.randn(N, p)
        u = np.random.randn(N, 1)
        y = X @ Beta.reshape(-1, 1) + u
        y_flat = y.flatten()
        
        # Initial LASSO fitting
        alphas, coefs, _ = lasso_path(X, y_flat, fit_intercept=False)
        
        # BIC calculation for initial LASSO
        y_hat = X @ coefs
        SSE = np.sum((y_hat - y_flat.reshape(-1, 1)) ** 2, axis=0)
        nz = np.sum(coefs != 0, axis=0)
        BIC = np.log(SSE) + (np.log(N) / N) * nz
        
        # Initial optimal coefficients
        min_bic_idx = np.argmin(BIC)
        beta_lasso = coefs[:, min_bic_idx]
        
        # Adaptive weights
        weights = np.abs(beta_lasso) ** (-1)
        weights[beta_lasso == 0] = 1e10  # Handle infinities
        
        # Adaptive LASSO: scale X by weights
        X_weighted = X / weights
        alphas_adaptive, coefs_adaptive, _ = lasso_path(X_weighted, y_flat, fit_intercept=False)
        
        # Scale coefficients back
        coefs_adaptive = coefs_adaptive / weights.reshape(-1, 1)
        
        # BIC calculation for adaptive LASSO
        y_hat_adaptive = X @ coefs_adaptive
        SSE_adaptive = np.sum((y_hat_adaptive - y_flat.reshape(-1, 1)) ** 2, axis=0)
        nz_adaptive = np.sum(coefs_adaptive != 0, axis=0)
        BIC_adaptive = np.log(SSE_adaptive) + (np.log(N) / N) * nz_adaptive
        
        # Final adaptive LASSO coefficients
        min_bic_idx_adaptive = np.argmin(BIC_adaptive)
        beta_lasso_adaptive = coefs_adaptive[:, min_bic_idx_adaptive]
        
        # Performance metrics
        nonz_beta = np.sum(Beta == 0)
        nonz_beta_hat = np.sum(beta_lasso_adaptive == 0)
        
        mcmat[i, 0] = np.sqrt(np.sum((beta_lasso_adaptive - Beta) ** 2))  # L2 distance
        mcmat[i, 1] = 1 if nonz_beta == nonz_beta_hat else 0             # Correct model selection
        mcmat[i, 2] = np.sum(beta_lasso_adaptive != 0)                   # Number of non-zero estimates
        
        beta_lasso_mat[i, :] = beta_lasso_adaptive
    
    return mcmat, beta_lasso_mat

# Example usage
# mc = 100
# N = 100
# Beta = [1, 1, 0, 0]
# mcmat_adaptive, beta_lasso_mat_adaptive = chunk_mcsA(mc, N, Beta)








#################################################################################################################


# ================================
#     chunk_sum_l2_adaptive
# ================================
def chunk_sum_l2_adaptive(MC_performance):
    """
    Calculates sum of L2 distances from Adaptive LASSO Monte Carlo results.
    
    Args:
        MC_performance (array): Performance matrix from Adaptive LASSO simulation
    
    Returns:
        float: Sum of L2 distances (first column)
    """
    return np.sum(MC_performance[:, 0])

# Example usage (assuming MC_performance from chunk_mcsA)
# mcmat, beta_lasso_mat = chunk_mcsA(mc, N, beta)
# total_l2_distance = chunk_sum_l2_adaptive(mcmat)
# print(total_l2_distance)