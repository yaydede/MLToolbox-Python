import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy import interpolate
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import mode
from sklearn.datasets import fetch_openml
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd


# ====================================
#           chunk_split
# ====================================

def chunk_split(image_path="png/split.png", width_scale=0.7, height_scale=0.7, method="matplotlib"):
    """
    Display an image with specified scaling, similar to knitr::include_graphics
    
    Args:
        image_path: Path to the image file
        width_scale: Width scaling factor (0.7 = 70%)
        height_scale: Height scaling factor (0.7 = 70%)
        method: Display method - "matplotlib" or "ipython"
    """
    if method == "matplotlib":
        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(img.shape[1] * width_scale / 100, 
                                       img.shape[0] * height_scale / 100))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    elif method == "ipython":
        display(Image(image_path, width=f"{int(width_scale*100)}%", 
                     height=f"{int(height_scale*100)}%"))
    
    return image_path



################################################################################




# ====================================
#           chunk_hy1
# ====================================

def chunk_hy1(n=10000, k=2, seed=2, verbose=False):
    """
    Create synthetic dataset and split into training and validation sets
    
    Args:
        n: Number of observations
        k: Number of folds (default 2)
        seed: Random seed
        verbose: Print dataset info if True
    
    Returns:
        tuple: (train_df, val_df) - training and validation DataFrames
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create synthetic dataset with DGM
    x = np.random.normal(3, 6, n)
    y = 2 + 13*x + np.random.normal(0, 1, n)
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Shuffle the data
    data = data.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)
    
    # Calculate slice size
    nslice = n // k
    
    # Split into train and validation sets
    train = data.iloc[:nslice].copy()
    val = data.iloc[nslice:].copy()
    
    if verbose:
        print(f"Training set shape: {train.shape}")
        print(f"Validation set shape: {val.shape}")
    
    return train, val






###################################################################################



# ====================================
#           chunk_hy2
# ====================================

def chunk_hy2(train, val, spans=[0.02, 0.1, 1.0]):
    """
    Fit LOESS models with different spans and predict on validation set
    
    Args:
        train: Training DataFrame with 'x' and 'y' columns
        val: Validation DataFrame with 'x' and 'y' columns
        spans: List of span parameters for LOESS smoothing
    
    Returns:
        dict: Dictionary with fitted models and predictions
    """
    results = {}
    
    for i, span in enumerate(spans):
        # Sort training data by x for LOESS-like behavior
        train_sorted = train.sort_values('x')
        
        # Calculate window size based on span
        window_size = max(3, int(len(train_sorted) * span))
        
        # Create locally weighted regression approximation
        x_train = train_sorted['x'].values
        y_train = train_sorted['y'].values
        
        # Use scipy's interpolation with smoothing
        # UnivariateSpline provides similar smoothing to LOESS
        smoothing_factor = len(train_sorted) * (1 - span) ** 3
        spline = interpolate.UnivariateSpline(x_train, y_train, 
                                            s=smoothing_factor, k=2)
        
        # Predict on validation set
        predictions = spline(val['x'].values)
        
        results[f'model_{i}'] = spline
        results[f'predictions_{i}'] = predictions
    
    return results






###############################################################################






# ====================================
#           chunk_hy3
# ====================================

def chunk_hy3(val, predictions_dict, spans=[0.02, 0.1, 1.0], verbose=True):
    """
    Calculate RMSPE for LOESS predictions and optionally display results
    
    Args:
        val: Validation DataFrame with 'y' column (actual values)
        predictions_dict: Dictionary with prediction arrays from chunk_hy2
        spans: List of span values used in models
        verbose: Print results if True
    
    Returns:
        list: RMSPE values for each span
    """
    rmspe_values = []
    results = []
    
    for i, span in enumerate(spans):
        pred_key = f'predictions_{i}'
        predictions = predictions_dict[pred_key]
        
        # Calculate RMSPE
        rmspe = np.sqrt(np.mean((val['y'].values - predictions) ** 2))
        rmspe_values.append(rmspe)
        
        result_str = f"With span = {span} rmspe is {rmspe:.6f}"
        results.append(result_str)
    
    if verbose:
        for result in results:
            print(result)
    
    return rmspe_values






######################################################################################





# ====================================
#           chunk_hy4
# ====================================

def chunk_hy4(image_path="png/grid3.png", width_scale=0.7, height_scale=0.7, method="matplotlib"):
    """
    Display an image with specified scaling, similar to knitr::include_graphics
    
    Args:
        image_path: Path to the image file
        width_scale: Width scaling factor (0.7 = 70%)
        height_scale: Height scaling factor (0.7 = 70%)
        method: Display method - "matplotlib" or "ipython"
    """
    if method == "matplotlib":
        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(img.shape[1] * width_scale / 100, 
                                       img.shape[0] * height_scale / 100))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    elif method == "ipython":
        display(Image(image_path, width=f"{int(width_scale*100)}%", 
                     height=f"{int(height_scale*100)}%"))
    
    return image_path







######################################################################################





# ====================================
#           chunk_hy5
# ====================================

def chunk_hy5(n=10000, k=10, span=0.1, seed=1, verbose=False):
    """
    Perform k-fold cross-validation with LOESS regression on simulated sine wave data
    
    Args:
        n: Number of observations
        k: Number of folds for cross-validation
        span: Span parameter for LOESS smoothing
        seed: Random seed
        verbose: Print fold progress if True
    
    Returns:
        tuple: (rmspe_values, mean_rmspe) - individual RMSPE values and their mean
    """
    # Set random seed
    np.random.seed(seed)
    
    # Simulate data
    x = np.sort(np.random.uniform(0, 2*np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)  # std = 1/4 = 0.25
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Shuffle indices
    mysample = np.random.permutation(n)
    nvalidate = round(n / k)
    
    rmspe_values = []
    
    # K-fold cross-validation loop
    for i in range(k):
        if verbose:
            print(f"K-fold loop: {i+1}")
        
        # Determine validation indices
        if i < k - 1:
            start_idx = i * nvalidate
            end_idx = (i + 1) * nvalidate
        else:
            start_idx = i * nvalidate
            end_idx = n
        
        ind_val = mysample[start_idx:end_idx]
        ind_train = np.setdiff1d(np.arange(n), ind_val)
        
        # Split data
        data_val = data.iloc[ind_val]
        data_train = data.iloc[ind_train]
        
        # Sort training data for interpolation
        train_sorted = data_train.sort_values('x')
        x_train = train_sorted['x'].values
        y_train = train_sorted['y'].values
        
        # Fit LOESS-like model using UnivariateSpline
        smoothing_factor = len(train_sorted) * (1 - span) ** 3
        model = interpolate.UnivariateSpline(x_train, y_train, 
                                           s=smoothing_factor, k=2)
        
        # Predict and calculate RMSPE
        predictions = model(data_val['x'].values)
        rmspe = np.sqrt(np.mean((data_val['y'].values - predictions) ** 2))
        rmspe_values.append(rmspe)
    
    mean_rmspe = np.mean(rmspe_values)
    
    if verbose:
        print(f"RMSPE values: {rmspe_values}")
        print(f"Mean RMSPE: {mean_rmspe:.6f}")
    
    return rmspe_values, mean_rmspe








#####################################################################################





# ====================================
#           chunk_hy6
# ====================================

def chunk_hy6(n=1000, k=10, seed=1, verbose=False):
    """
    Grid search with k-fold CV to find optimal span and degree parameters for LOESS
    
    Args:
        n: Number of observations
        k: Number of folds for cross-validation
        seed: Random seed
        verbose: Print progress if True
    
    Returns:
        pandas.DataFrame: Optimal parameters (span, degree) for each fold
    """
    # Set random seed
    np.random.seed(seed)
    
    # Simulate data
    x = np.sort(np.random.uniform(0, 2*np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Create parameter grid
    span_values = np.arange(0.01, 1.01, 0.02)  # 0.01 to 1.0 by 0.02
    degree_values = [1, 2]
    grid = pd.DataFrame(list(product(span_values, degree_values)), 
                       columns=['span', 'degree'])
    
    if verbose:
        print(f"Grid shape: {grid.shape}")
        print(grid.head())
    
    # Shuffle indices for CV
    ind = np.random.permutation(n)
    nval = round(n / k)
    
    optimal_indices = []
    
    # K-fold cross-validation
    for i in range(k):
        if verbose:
            print(f"Processing fold {i+1}/{k}")
        
        # Determine validation indices
        if i < k - 1:
            start_idx = i * nval
            end_idx = (i + 1) * nval
        else:
            start_idx = i * nval
            end_idx = n
        
        ind_val = ind[start_idx:end_idx]
        ind_train = np.setdiff1d(np.arange(n), ind_val)
        
        # Split data
        data_val = data.iloc[ind_val]
        data_train = data.iloc[ind_train]
        
        # Sort training data
        train_sorted = data_train.sort_values('x')
        x_train = train_sorted['x'].values
        y_train = train_sorted['y'].values
        
        rmspe_values = []
        
        # Grid search
        for _, row in grid.iterrows():
            span, degree = row['span'], int(row['degree'])
            
            # Fit model with current parameters
            smoothing_factor = len(train_sorted) * (1 - span) ** 3
            model = interpolate.UnivariateSpline(x_train, y_train, 
                                               s=smoothing_factor, k=min(degree, 3))
            
            # Predict and calculate RMSPE
            predictions = model(data_val['x'].values)
            rmspe = np.sqrt(np.mean((data_val['y'].values - predictions) ** 2))
            rmspe_values.append(rmspe)
        
        # Find optimal parameters for this fold
        best_idx = np.argmin(rmspe_values)
        optimal_indices.append(best_idx)
    
    # Create results dataframe
    opgrid = grid.iloc[optimal_indices].reset_index(drop=True)
    opgrid.index = range(1, k+1)
    
    if verbose:
        print("\nOptimal parameters for each fold:")
        print(opgrid)
    
    return opgrid






#########################################################################################################





# ====================================
#           chunk_hy7
# ====================================

def chunk_hy7(image_path="png/grid.png", figsize_scale=0.7):
    """Display image with specified scaling, equivalent to R knitr::include_graphics"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8*figsize_scale, 6*figsize_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()






###############################################################################################




# ====================================
#           chunk_hy8
# ====================================

def chunk_hy8():
    """LOESS cross-validation with hyperparameter tuning using polynomial regression approximation"""
    # Generate same data
    np.random.seed(1)
    n = 1000
    x = np.sort(np.random.uniform(0, 2*np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Grid (span approximated as alpha for Ridge, degree as polynomial degree)
    spans = np.arange(0.01, 1.01, 0.02)
    degrees = [1, 2]
    grid = list(itertools.product(spans, degrees))
    
    # Train-test split
    np.random.seed(321)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=321)
    
    # k-CV setup
    np.random.seed(3)
    k = 10
    indices = np.random.permutation(len(trainset))
    fold_size = len(trainset) // k
    
    # CV loop
    opt_indices = []
    
    for i in range(k):
        if i < k - 1:
            val_idx = indices[i * fold_size:(i + 1) * fold_size]
        else:
            val_idx = indices[i * fold_size:]
            
        train_idx = np.setdiff1d(indices, val_idx)
        
        data_train = trainset.iloc[train_idx]
        data_val = trainset.iloc[val_idx]
        
        rmspe_scores = []
        
        for s, (span, degree) in enumerate(grid):
            # Approximate LOESS with polynomial Ridge regression
            poly_ridge = Pipeline([
                ('poly', PolynomialFeatures(degree=int(degree))),
                ('ridge', Ridge(alpha=1/span))  # Inverse relationship
            ])
            
            poly_ridge.fit(data_train[['x']], data_train['y'])
            fit = poly_ridge.predict(data_val[['x']])
            rmspe_scores.append(np.sqrt(mean_squared_error(data_val['y'], fit)))
        
        opt_indices.append(np.argmin(rmspe_scores))
    
    # Extract optimal hyperparameters
    opt_params = [grid[i] for i in opt_indices]
    opt_spans = [param[0] for param in opt_params]
    opt_degrees = [param[1] for param in opt_params]
    
    opt_degree = mode(opt_degrees)[0][0]
    opt_span = np.mean(opt_spans)
    
    # Final model evaluation on test set
    final_model = Pipeline([
        ('poly', PolynomialFeatures(degree=int(opt_degree))),
        ('ridge', Ridge(alpha=1/opt_span))
    ])
    
    final_model.fit(trainset[['x']], trainset['y'])
    test_pred = final_model.predict(testset[['x']])
    rmspe_test = np.sqrt(mean_squared_error(testset['y'], test_pred))
    
    return {
        'optimal_degree': opt_degree,
        'optimal_span': opt_span,
        'test_rmspe': rmspe_test
    }







###############################################################################################################





# ====================================
#           chunk_hy9
# ====================================

def chunk_hy9(t=100):
    """Repeated LOESS cross-validation with hyperparameter tuning over multiple train-test splits"""
    # Generate same data
    np.random.seed(1)
    n = 1000
    x = np.sort(np.random.uniform(0, 2*np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Grid
    spans = np.arange(0.01, 1.01, 0.02)
    degrees = [1, 2]
    grid = list(itertools.product(spans, degrees))
    
    rmspe_test = []
    
    for l in range(t):
        # Train-test split
        np.random.seed(10 + l)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=10+l)
        
        # k-CV setup
        np.random.seed(100 + l)
        k = 10
        indices = np.random.permutation(len(trainset))
        fold_size = len(trainset) // k
        
        # CV loop
        opt_indices = []
        
        for i in range(k):
            if i < k - 1:
                val_idx = indices[i * fold_size:(i + 1) * fold_size]
            else:
                val_idx = indices[i * fold_size:]
                
            train_idx = np.setdiff1d(indices, val_idx)
            
            data_train = trainset.iloc[train_idx]
            data_val = trainset.iloc[val_idx]
            
            rmspe_scores = []
            
            for s, (span, degree) in enumerate(grid):
                poly_ridge = Pipeline([
                    ('poly', PolynomialFeatures(degree=int(degree))),
                    ('ridge', Ridge(alpha=1/span))
                ])
                
                poly_ridge.fit(data_train[['x']], data_train['y'])
                fit = poly_ridge.predict(data_val[['x']])
                rmspe_scores.append(np.sqrt(mean_squared_error(data_val['y'], fit)))
            
            opt_indices.append(np.argmin(rmspe_scores))
        
        # Extract optimal hyperparameters
        opt_params = [grid[i] for i in opt_indices]
        opt_spans = [param[0] for param in opt_params]
        opt_degrees = [param[1] for param in opt_params]
        
        opt_degree = mode(opt_degrees)[0][0]
        opt_span = np.mean(opt_spans)
        
        # Final model evaluation on test set
        final_model = Pipeline([
            ('poly', PolynomialFeatures(degree=int(opt_degree))),
            ('ridge', Ridge(alpha=1/opt_span))
        ])
        
        final_model.fit(trainset[['x']], trainset['y'])
        test_pred = final_model.predict(testset[['x']])
        rmspe_test.append(np.sqrt(mean_squared_error(testset['y'], test_pred)))
    
    return np.array(rmspe_test)





####################################################################################################




# ====================================
#           chunk_hy10
# ====================================

def chunk_hy10(rmspe_test):
    """Plot RMSPE test results with mean line and calculate statistics"""
    plt.figure(figsize=(10, 6))
    plt.plot(rmspe_test, 'ro', markersize=4)
    plt.axhline(y=np.mean(rmspe_test), color='green', linewidth=3, label=f'Mean: {np.mean(rmspe_test):.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSPE Test')
    plt.title('RMSPE Test Scores Across Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'mean': np.mean(rmspe_test),
        'variance': np.var(rmspe_test, ddof=1)
    }




##################################################################################




# ====================================
#           chunk_hy11
# ====================================

def chunk_hy11(n=1000):
    """Calculate bootstrap sample statistics - training set size and out-of-bag percentage"""
    np.random.seed(42)  # For reproducibility
    x_train = np.unique(np.random.choice(n, size=n, replace=True))
    
    train_pct = len(x_train) / n
    oob_pct = 1 - train_pct
    
    return {
        'train_percentage': train_pct,
        'oob_percentage': oob_pct,
        'unique_samples': len(x_train)
    }





###############################################################################################################




# ====================================
#           chunk_hy12
# ====================================

def chunk_hy12(t=100):
    """LOESS hyperparameter tuning using out-of-bag validation instead of k-fold CV"""
    # Generate same data
    np.random.seed(1)
    n = 1000
    x = np.sort(np.random.uniform(0, 2*np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Grid
    spans = np.arange(0.01, 1.01, 0.02)
    degrees = [1, 2]
    grid = list(itertools.product(spans, degrees))
    
    rmspe_test = []
    
    for l in range(t):
        # Train-test split
        np.random.seed(10 + l)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=10+l)
        trainset = trainset.reset_index(drop=True)
        
        # OOB loops (10 bootstrap iterations)
        opt_indices = []
        
        for i in range(10):
            # Bootstrap sample with replacement
            boot_indices = np.unique(np.random.choice(len(trainset), size=len(trainset), replace=True))
            oob_indices = np.setdiff1d(np.arange(len(trainset)), boot_indices)
            
            data_train = trainset.iloc[boot_indices]
            data_val = trainset.iloc[oob_indices]  # Out-of-bag validation
            
            rmspe_scores = []
            
            for s, (span, degree) in enumerate(grid):
                poly_ridge = Pipeline([
                    ('poly', PolynomialFeatures(degree=int(degree))),
                    ('ridge', Ridge(alpha=1/span))
                ])
                
                poly_ridge.fit(data_train[['x']], data_train['y'])
                fit = poly_ridge.predict(data_val[['x']])
                rmspe_scores.append(np.sqrt(mean_squared_error(data_val['y'], fit)))
            
            opt_indices.append(np.argmin(rmspe_scores))
        
        # Extract optimal hyperparameters
        opt_params = [grid[i] for i in opt_indices]
        opt_spans = [param[0] for param in opt_params]
        opt_degrees = [param[1] for param in opt_params]
        
        opt_degree = mode(opt_degrees)[0][0]
        opt_span = np.mean(opt_spans)
        
        # Final model evaluation on test set
        final_model = Pipeline([
            ('poly', PolynomialFeatures(degree=int(opt_degree))),
            ('ridge', Ridge(alpha=1/opt_span))
        ])
        
        final_model.fit(trainset[['x']], trainset['y'])
        test_pred = final_model.predict(testset[['x']])
        rmspe_test.append(np.sqrt(mean_squared_error(testset['y'], test_pred)))
    
    return np.array(rmspe_test)




###########################################################################################################




# ====================================
#           chunk_hy13
# ====================================

def chunk_hy13(rmspe_test):
    """Plot RMSPE test results from OOB validation with mean line and calculate statistics"""
    plt.figure(figsize=(10, 6))
    plt.plot(rmspe_test, 'ro', markersize=4)
    plt.axhline(y=np.mean(rmspe_test), color='green', linewidth=3, label=f'Mean: {np.mean(rmspe_test):.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSPE Test')
    plt.title('RMSPE Test Scores - OOB Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'mean': np.mean(rmspe_test),
        'variance': np.var(rmspe_test, ddof=1)
    }






###################################################################################################################




# ====================================
#           chunk_hy14
# ====================================

def chunk_hy14(t=100):
    """LOESS hyperparameter tuning using averaged OOB validation across grid parameters"""
    # Generate same data
    np.random.seed(1)
    n = 1000
    x = np.sort(np.random.uniform(0, 2*np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Grid
    spans = np.arange(0.01, 1.01, 0.02)
    degrees = [1, 2]
    grid = list(itertools.product(spans, degrees))
    
    rmspe_test = []
    
    for l in range(t):
        # Train-test split
        np.random.seed(10 + l)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=10+l)
        trainset = trainset.reset_index(drop=True)
        
        opt_scores = []
        
        # Grid loop - evaluate each parameter combination
        for s, (span, degree) in enumerate(grid):
            rmspe_scores = []
            
            # OOB loops - 10 bootstrap iterations for current parameter set
            for i in range(10):
                np.random.seed(i + 100)
                boot_indices = np.unique(np.random.choice(len(trainset), size=len(trainset), replace=True))
                oob_indices = np.setdiff1d(np.arange(len(trainset)), boot_indices)
                
                data_train = trainset.iloc[boot_indices]
                data_val = trainset.iloc[oob_indices]  # Out-of-bag validation
                
                poly_ridge = Pipeline([
                    ('poly', PolynomialFeatures(degree=int(degree))),
                    ('ridge', Ridge(alpha=1/span))
                ])
                
                poly_ridge.fit(data_train[['x']], data_train['y'])
                fit = poly_ridge.predict(data_val[['x']])
                rmspe_scores.append(np.sqrt(mean_squared_error(data_val['y'], fit)))
            
            opt_scores.append(np.mean(rmspe_scores))  # Average RMSPE for this parameter set
        
        # Select best parameter combination
        best_param_idx = np.argmin(opt_scores)
        opt_span, opt_degree = grid[best_param_idx]
        
        # Final model evaluation on test set
        final_model = Pipeline([
            ('poly', PolynomialFeatures(degree=int(opt_degree))),
            ('ridge', Ridge(alpha=1/opt_span))
        ])
        
        final_model.fit(trainset[['x']], trainset['y'])
        test_pred = final_model.predict(testset[['x']])
        rmspe_test.append(np.sqrt(mean_squared_error(testset['y'], test_pred)))
    
    return np.array(rmspe_test)





###########################################################################################################






# ====================================
#           chunk_hy15
# ====================================

def chunk_hy15(rmspe_test):
    """Plot RMSPE test results from averaged OOB validation with mean line and calculate statistics"""
    plt.figure(figsize=(10, 6))
    plt.plot(rmspe_test, 'ro', markersize=4)
    plt.axhline(y=np.mean(rmspe_test), color='green', linewidth=3, label=f'Mean: {np.mean(rmspe_test):.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSPE Test')
    plt.title('RMSPE Test Scores - Averaged OOB Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'mean': np.mean(rmspe_test),
        'variance': np.var(rmspe_test, ddof=1)
    }






############################################################################################################





# ====================================
#           chunk_hy16
# ====================================

def chunk_hy16():
    """Load EuStockMarkets-like data, add day index, and plot FTSE time series"""
    # Create synthetic EuStockMarkets-like data since it's not directly available
    np.random.seed(42)
    n_days = 1860  # Approximate length of EuStockMarkets dataset
    
    # Generate synthetic stock data with realistic patterns
    days = np.arange(1, n_days + 1)
    
    # DAX (German stock index)
    dax_trend = 1600 + 0.1 * days + np.cumsum(np.random.normal(0, 10, n_days))
    
    # SMI (Swiss stock index)  
    smi_trend = 1700 + 0.15 * days + np.cumsum(np.random.normal(0, 12, n_days))
    
    # CAC (French stock index)
    cac_trend = 1950 + 0.12 * days + np.cumsum(np.random.normal(0, 15, n_days))
    
    # FTSE (British stock index)
    ftse_trend = 2400 + 0.2 * days + np.cumsum(np.random.normal(0, 20, n_days))
    
    data = pd.DataFrame({
        'DAX': np.maximum(dax_trend, 0),
        'SMI': np.maximum(smi_trend, 0), 
        'CAC': np.maximum(cac_trend, 0),
        'FTSE': np.maximum(ftse_trend, 0),
        'day_index': days
    })
    
    print(data.head())
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['day_index'], data['FTSE'], color='orange', linewidth=1)
    plt.xlabel('Day Index')
    plt.ylabel('FTSE')
    plt.title('FTSE Stock Index Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return data






######################################################################################################################




# ====================================
#           chunk_hy17
# ====================================

def chunk_hy17(data=None):
    """Decompose FTSE time series and plot components"""
    if data is None:
        # Generate synthetic EuStockMarkets-like data
        np.random.seed(42)
        n_days = 1860
        days = np.arange(1, n_days + 1)
        ftse_trend = 2400 + 0.2 * days + np.cumsum(np.random.normal(0, 20, n_days))
        data = pd.DataFrame({
            'FTSE': np.maximum(ftse_trend, 0),
            'day_index': days
        })
    
    # Create time series with proper index
    ts = pd.Series(data['FTSE'].values, index=pd.date_range('1991-01-01', periods=len(data), freq='D'))
    
    # Perform seasonal decomposition (using additive model, period=252 for trading days)
    decomposition = seasonal_decompose(ts, model='additive', period=252)
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Original series
    decomposition.observed.plot(ax=axes[0], color='red', title='Original')
    axes[0].set_ylabel('Observed')
    
    # Trend component
    decomposition.trend.plot(ax=axes[1], color='red', title='Trend')
    axes[1].set_ylabel('Trend')
    
    # Seasonal component
    decomposition.seasonal.plot(ax=axes[2], color='red', title='Seasonal')
    axes[2].set_ylabel('Seasonal')
    
    # Residual component
    decomposition.resid.plot(ax=axes[3], color='red', title='Residuals')
    axes[3].set_ylabel('Residuals')
    axes[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    return decomposition






####################################################################################################






# ====================================
#           chunk_hy18
# ====================================

def chunk_hy18(data):
    """Plot FTSE with linear regression and LOESS smoothing at different span values"""
    plt.figure(figsize=(12, 8))
    
    # Original FTSE data
    plt.plot(data['day_index'], data['FTSE'], color='red', linewidth=2, label='FTSE', alpha=0.7)
    
    # Linear regression trend
    X = data[['day_index']]
    y = data['FTSE']
    lr = LinearRegression()
    lr.fit(X, y)
    linear_pred = lr.predict(X)
    plt.plot(data['day_index'], linear_pred, color='green', linewidth=1, label='Linear Trend')
    
    # LOESS approximations with different spans using polynomial regression
    # Span 0.01 - very tight fit (high degree polynomial with low regularization)
    poly_tight = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ])
    poly_tight.fit(X, y)
    tight_pred = poly_tight.predict(X)
    plt.plot(data['day_index'], tight_pred, color='grey', linewidth=2, label='Tight Smooth (span≈0.01)')
    
    # Span 0.1 - medium fit (moderate smoothing)
    # Use rolling window for medium smoothing
    window_size = max(1, int(0.1 * len(data)))
    medium_pred = data['FTSE'].rolling(window=window_size, center=True).mean()
    plt.plot(data['day_index'], medium_pred, color='blue', linewidth=2, label='Medium Smooth (span≈0.1)')
    
    # Span 0.9 - very smooth (almost linear)
    # Use large rolling window for heavy smoothing
    window_size_large = max(1, int(0.9 * len(data)))
    smooth_pred = data['FTSE'].rolling(window=window_size_large, center=True).mean()
    plt.plot(data['day_index'], smooth_pred, color='yellow', linewidth=2, label='Heavy Smooth (span≈0.9)')
    
    plt.xlabel('Day Index')
    plt.ylabel('FTSE')
    plt.title('FTSE with Different Smoothing Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()






############################################################################################################





# ====================================
#           chunk_hy19
# ====================================

def chunk_hy19(image_path="png/TScv.png", figsize_scale=0.7):
    """Display time series cross-validation image with specified scaling"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8*figsize_scale, 6*figsize_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()






#############################################################################################################





# ====================================
#           chunk_hy20
# ====================================

def chunk_hy20(data):
    """Time series rolling cross-validation with LOESS hyperparameter tuning"""
    spans = np.arange(0.05, 1.05, 0.05)
    h = 10
    opt_indices = []
    
    # Calculate fold sizes for rolling CV
    n_valid = len(data) // h
    cuts = [0] + [n_valid * j for j in range(1, h + 1)]
    cuts[-1] = len(data)  # Ensure last cut includes all data
    
    # Rolling CV loop
    for i in range(h):
        if i < h - 1:
            train_data = data.iloc[cuts[0]:cuts[i + 1]]
            if i + 2 < len(cuts):
                valid_data = data.iloc[cuts[i + 1]:cuts[i + 2]]
            else:
                continue  # Skip if no validation data
        else:
            train_data = data.iloc[cuts[0]:cuts[i]]
            continue  # No validation for last fold
        
        if len(valid_data) == 0:
            continue
            
        rmspe_scores = []
        
        for s, span in enumerate(spans):
            # Approximate LOESS with polynomial Ridge regression
            poly_ridge = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge(alpha=1/span))
            ])
            
            X_train = train_data[['day_index']]
            y_train = train_data['FTSE']
            X_valid = valid_data[['day_index']]
            y_valid = valid_data['FTSE']
            
            poly_ridge.fit(X_train, y_train)
            predictions = poly_ridge.predict(X_valid)
            rmspe_scores.append(np.sqrt(mean_squared_error(y_valid, predictions)))
        
        opt_indices.append(np.argmin(rmspe_scores))
    
    # Calculate optimal span
    opt_spans = [spans[idx] for idx in opt_indices if idx < len(spans)]
    opt_span = np.mean(opt_spans)
    
    # Final model with optimal span
    final_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('ridge', Ridge(alpha=1/opt_span))
    ])
    
    final_model.fit(data[['day_index']], data['FTSE'])
    smooth_pred = final_model.predict(data[['day_index']])
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(data['day_index'], data['FTSE'], color='gray', linewidth=1, label='FTSE Original')
    plt.plot(data['day_index'], smooth_pred, color='red', linewidth=2, label=f'LOESS Smooth (span={opt_span:.3f})')
    
    plt.xlabel('Day Index')
    plt.ylabel('FTSE')
    plt.title('FTSE with Optimal LOESS Smoothing from Rolling CV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_span': opt_span,
        'optimal_indices': opt_indices
    }








#######################################################################################################################





# ====================================
#           chunk_hy21
# ====================================

def chunk_hy21(image_path="png/grid4.png", figsize_scale=0.7):
    """Display grid4 image with specified scaling"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8*figsize_scale, 6*figsize_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()