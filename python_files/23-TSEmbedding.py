import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# ====================================
#      chunk_toronto_data
# ====================================

def chunk_toronto_data(data_dict=None, data_path=None):
    """
    Load and prepare Toronto time series data with mobility and cases.
    
    Parameters:
    -----------
    data_dict : dict, optional
        Dictionary containing 'mob' and 'cases' data arrays
    data_path : str, optional  
        Path to data file (for future implementation)
        
    Returns:
    --------
    pd.DataFrame : Time series dataframe with datetime index
    """
    
    # Create date range from 2020-03-01 to 2020-11-21
    day = pd.date_range(start='2020-03-01', end='2020-11-21', freq='D')
    
    # Load or use provided data
    if data_dict is not None:
        mob_data = data_dict['mob']
        cases_data = data_dict['cases']
    else:
        # Placeholder for data loading - adapt based on your data format
        print("Note: Provide data_dict parameter with 'mob' and 'cases' arrays")
        return None
    
    # Create DataFrame (equivalent to tibble)
    tdata = pd.DataFrame({
        'Day': day,
        'mob': mob_data,
        'cases': cases_data
    })
    
    # Convert to time series with Day as index (equivalent to as_tsibble)
    toronto = tdata.set_index('Day')
    
    # Display the dataframe
    print(f"Toronto Time Series Data ({toronto.shape[0]} observations)")
    print(f"Date range: {toronto.index.min()} to {toronto.index.max()}")
    print("\nFirst 10 rows:")
    print(toronto.head(10))
    print("\nLast 10 rows:")
    print(toronto.tail(10))
    print(f"\nData types:\n{toronto.dtypes}")
    print(f"\nSummary statistics:\n{toronto.describe()}")
    
    return toronto







##################################################################################################


# ====================================
#      chunk_var_stationary
# ====================================

def chunk_var_stationary(toronto, max_lags=10):
    """
    Make time series stationary by differencing and fit VAR model with BIC selection.
    
    Parameters:
    -----------
    toronto : pd.DataFrame
        Time series dataframe with 'cases' and 'mob' columns
    max_lags : int
        Maximum number of lags to consider for VAR model
        
    Returns:
    --------
    dict : VAR model results and transformed data
    """
    
    # Create differenced series for stationarity
    trdf = toronto.copy()
    trdf['diffcases'] = trdf['cases'].diff()
    trdf['diffmob'] = trdf['mob'].diff()
    
    # Remove first row with NaN values (equivalent to trdf[-1, ])
    trdf_clean = trdf.dropna()
    
    # Prepare data for VAR model (select differenced variables)
    var_data = trdf_clean[['diffcases', 'diffmob']]
    
    # Fit VAR model with BIC selection
    var_model = VAR(var_data)
    
    # Select optimal lag order using BIC
    lag_order = var_model.select_order(maxlags=max_lags)
    optimal_lags = lag_order.bic
    
    print(f"BIC lag selection results:")
    print(f"Selected lag order: {optimal_lags}")
    
    # Fit VAR model with selected lag order
    fitted_var = var_model.fit(optimal_lags, ic='bic')
    
    # Display model summary (equivalent to glance() and report())
    print(f"\nVAR Model Summary:")
    print(f"Number of observations: {fitted_var.nobs}")
    print(f"Number of variables: {fitted_var.neqs}")
    print(f"Lag order: {fitted_var.k_ar}")
    print(f"Log Likelihood: {fitted_var.llf:.4f}")
    print(f"AIC: {fitted_var.aic:.4f}")
    print(f"BIC: {fitted_var.bic:.4f}")
    print(f"HQIC: {fitted_var.hqic:.4f}")
    
    # Print coefficient summary
    print(f"\nModel Coefficients:")
    print(fitted_var.summary())
    
    # Test for serial correlation in residuals
    try:
        serial_test = fitted_var.test_serial_correlation(lags=optimal_lags)
        print(f"\nPortmanteau Test for Serial Correlation:")
        print(f"Test Statistic: {serial_test.test_statistic:.4f}")
        print(f"P-value: {serial_test.pvalue:.4f}")
        print(f"Critical Value (5%): {serial_test.critical_value:.4f}")
    except:
        pass
    
    # Test for normality of residuals
    try:
        normality_test = fitted_var.test_normality()
        print(f"\nJarque-Bera Test for Normality:")
        print(f"Test Statistic: {normality_test.test_statistic:.4f}")  
        print(f"P-value: {normality_test.pvalue:.4f}")
    except:
        pass
    
    return {
        'var_model': fitted_var,
        'transformed_data': trdf_clean,
        'optimal_lags': optimal_lags,
        'lag_selection': lag_order,
        'residuals': fitted_var.resid,
        'fitted_values': fitted_var.fittedvalues,
        'model_summary': {
            'nobs': fitted_var.nobs,
            'neqs': fitted_var.neqs,
            'k_ar': fitted_var.k_ar,
            'llf': fitted_var.llf,
            'aic': fitted_var.aic,
            'bic': fitted_var.bic,
            'hqic': fitted_var.hqic
        }
    }








##################################################################################################



# ====================================
#      chunk_var_forecast
# ====================================

def chunk_var_forecast(fitted_var, trdf_clean, h=14, start_display=200, plot=True):
    """
    Generate VAR model forecast and create visualization.
    
    Parameters:
    -----------
    fitted_var : statsmodels VAR results
        Fitted VAR model from chunk_var_stationary
    trdf_clean : pd.DataFrame
        Transformed data with differenced series
    h : int
        Forecast horizon (number of periods ahead)
    start_display : int
        Starting observation for display (equivalent to [-c(1:200), ])
    plot : bool
        Whether to create forecast plot
        
    Returns:
    --------
    dict : Forecast results and plot data
    """
    
    # Generate forecast
    forecast_result = fitted_var.forecast(fitted_var.endog, steps=h)
    
    # Create forecast index (dates)
    last_date = trdf_clean.index[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=h,
        freq='D'
    )
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(
        forecast_result,
        index=forecast_index,
        columns=['diffcases_forecast', 'diffmob_forecast']
    )
    
    # Display subset of data (equivalent to trdf[-c(1:200), ])
    display_data = trdf_clean.iloc[start_display:].copy()
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot diffcases
        ax1.plot(display_data.index, display_data['diffcases'], 
                'b-', label='Actual diffcases', alpha=0.8)
        ax1.plot(forecast_df.index, forecast_df['diffcases_forecast'], 
                'r--', label='Forecast diffcases', linewidth=2)
        ax1.axvline(x=trdf_clean.index[-1], color='gray', linestyle=':', 
                   alpha=0.7, label='Forecast Start')
        ax1.set_title('VAR Forecast - Differenced Cases')
        ax1.set_ylabel('diffcases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot diffmob
        ax2.plot(display_data.index, display_data['diffmob'], 
                'g-', label='Actual diffmob', alpha=0.8)
        ax2.plot(forecast_df.index, forecast_df['diffmob_forecast'], 
                'orange', linestyle='--', label='Forecast diffmob', linewidth=2)
        ax2.axvline(x=trdf_clean.index[-1], color='gray', linestyle=':', 
                   alpha=0.7, label='Forecast Start')
        ax2.set_title('VAR Forecast - Differenced Mobility')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('diffmob')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Calculate forecast confidence intervals (if available)
    try:
        forecast_ci = fitted_var.forecast_interval(fitted_var.endog, steps=h, alpha=0.05)
        
        # Create confidence interval DataFrames
        lower_ci = pd.DataFrame(
            forecast_ci[0],
            index=forecast_index,
            columns=['diffcases_lower', 'diffmob_lower']
        )
        upper_ci = pd.DataFrame(
            forecast_ci[1], 
            index=forecast_index,
            columns=['diffcases_upper', 'diffmob_upper']
        )
        
        forecast_with_ci = pd.concat([forecast_df, lower_ci, upper_ci], axis=1)
        
    except:
        forecast_with_ci = forecast_df
    
    print(f"VAR Forecast Summary:")
    print(f"Forecast horizon: {h} periods")
    print(f"Forecast start date: {forecast_index[0]}")
    print(f"Forecast end date: {forecast_index[-1]}")
    print(f"\nForecast values:")
    print(forecast_df.round(4))
    
    return {
        'forecast': forecast_df,
        'forecast_with_ci': forecast_with_ci,
        'display_data': display_data,
        'forecast_index': forecast_index,
        'fitted_model': fitted_var
    }








##################################################################################################



# ====================================
#      chunk_embed_example
# ====================================

def chunk_embed_example(Y=None, embed_dim=3):
    """
    Create embedded time series matrix (equivalent to R's embed function).
    
    Parameters:
    -----------
    Y : array-like, optional
        Time series data (defaults to 1:10)
    embed_dim : int
        Embedding dimension (number of lags + current value)
        
    Returns:
    --------
    pd.DataFrame : Embedded time series matrix
    """
    
    def embed_series(series, dimension):
        """
        Embed a time series into a matrix of lagged values.
        Equivalent to R's embed() function.
        """
        series = np.array(series)
        n = len(series)
        
        if dimension > n:
            raise ValueError(f"Embedding dimension ({dimension}) cannot exceed series length ({n})")
        
        # Create embedded matrix
        embedded = np.zeros((n - dimension + 1, dimension))
        
        for i in range(dimension):
            embedded[:, i] = series[dimension - 1 - i:n - i]
            
        return embedded
    
    # Default data if none provided
    if Y is None:
        Y = np.arange(1, 11)  # 1:10 in R
    
    # Create embedded matrix
    embedded_matrix = embed_series(Y, embed_dim)
    
    # Create column names (Y(t), Y(t-1), Y(t-2), ...)
    col_names = ['Y(t)'] + [f'Y(t-{i})' for i in range(1, embed_dim)]
    
    # Convert to DataFrame
    embedded_df = pd.DataFrame(embedded_matrix, columns=col_names)
    
    print("Original series:", Y)
    print(f"\nEmbedded matrix (dimension {embed_dim}):")
    print(embedded_df)
    
    return embedded_df








##################################################################################################




# ====================================
#      chunk_stationary_ar
# ====================================

def chunk_stationary_ar(n=10000, rho=0.85, seed=345, plot=True):
    """
    Generate stationary AR(1) process and create diagnostic plots.
    
    Parameters:
    -----------
    n : int
        Number of observations
    rho : float
        AR coefficient (< 1 for stationarity)
    seed : int
        Random seed for reproducibility
    plot : bool
        Whether to create diagnostic plots
        
    Returns:
    --------
    dict : Generated AR process and lagged values
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize series
    y = np.zeros(n)
    y[0] = 0  # Starting value
    
    # Generate random errors
    eps = np.random.normal(0, 1, n)
    
    # Generate AR(1) process: y[t] = rho * y[t-1] + eps[t]
    for j in range(n - 1):
        y[j + 1] = y[j] * rho + eps[j]
    
    # Create lagged series for scatter plot
    ylagged = y[1:n]    # y[2:n] in R (y(t))
    y_prev = y[0:(n-1)] # y[1:(n-1)] in R (y(t-1))
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot: y(t) vs y(t-1)
        ax1.scatter(y_prev, ylagged, color='lightpink', alpha=0.6, s=1)
        ax1.set_xlabel('y(t-1)')
        ax1.set_ylabel('y')
        ax1.set_title(f'AR(1) Process: ρ = {rho}')
        ax1.grid(True, alpha=0.3)
        
        # Time series plot (first 500 observations)
        ax2.plot(y[:500], color='red', linewidth=1)
        ax2.set_xlabel('t')
        ax2.set_ylabel('y')
        ax2.set_title('Time Series Plot (First 500 obs)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print(f"AR(1) Process Generated:")
    print(f"n = {n}, ρ = {rho}")
    print(f"Process is {'stationary' if abs(rho) < 1 else 'non-stationary'}")
    print(f"Sample mean: {np.mean(y):.4f}")
    print(f"Sample std: {np.std(y):.4f}")
    print(f"Sample autocorr at lag 1: {np.corrcoef(y_prev, ylagged)[0,1]:.4f}")
    
    return {
        'y': y,
        'ylagged': ylagged,
        'y_prev': y_prev,
        'rho': rho,
        'n': n,
        'eps': eps
    }








##################################################################################################


# ====================================
#       chunk_embed_ar
# ====================================

def chunk_embed_ar(y, embed_dim=2):
    """
    Embed AR time series and display head of original and embedded series.
    
    Parameters:
    -----------
    y : array-like
        Time series data
    embed_dim : int
        Embedding dimension (default 2 for AR(1))
        
    Returns:
    --------
    pd.DataFrame : Embedded time series matrix
    """
    
    def embed_series(series, dimension):
        """
        Embed a time series into a matrix of lagged values.
        Equivalent to R's embed() function.
        """
        series = np.array(series)
        n = len(series)
        
        if dimension > n:
            raise ValueError(f"Embedding dimension ({dimension}) cannot exceed series length ({n})")
        
        # Create embedded matrix
        embedded = np.zeros((n - dimension + 1, dimension))
        
        for i in range(dimension):
            embedded[:, i] = series[dimension - 1 - i:n - i]
            
        return embedded
    
    # Display head of original series
    print("head(y):")
    print(pd.Series(y[:6]).to_string())
    
    # Create embedded matrix
    y_embedded = embed_series(y, embed_dim)
    
    # Create column names matching R output
    if embed_dim == 2:
        col_names = ["yt", "yt_1"]
    else:
        col_names = ["yt"] + [f"yt_{i}" for i in range(1, embed_dim)]
    
    # Convert to DataFrame
    y_em = pd.DataFrame(y_embedded, columns=col_names)
    
    # Display head of embedded matrix
    print(f"\nhead(y_em):")
    print(y_em.head(6).to_string())
    
    return y_em








##################################################################################################


# ====================================
#      chunk_ar_regression
# ====================================

def chunk_ar_regression(y_em):
    """
    Fit AR(1) regression without intercept (equivalent to lm(yt ~ yt_1 - 1)).
    
    Parameters:
    -----------
    y_em : pd.DataFrame
        Embedded time series matrix with 'yt' and 'yt_1' columns
        
    Returns:
    --------
    dict : Regression results and fitted model
    """
    
    # Ensure it's a DataFrame
    if isinstance(y_em, np.ndarray):
        y_em = pd.DataFrame(y_em, columns=['yt', 'yt_1'])
    
    # Extract variables for regression
    X = y_em[['yt_1']].values  # Predictor (lagged values)
    y = y_em['yt'].values      # Response (current values)
    
    # Fit linear regression without intercept (fit_intercept=False)
    ar1 = LinearRegression(fit_intercept=False)
    ar1.fit(X, y)
    
    # Calculate additional statistics
    y_pred = ar1.predict(X)
    residuals = y - y_pred
    
    # Calculate R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate standard error of coefficient
    n = len(y)
    mse = ss_res / (n - 1)  # No intercept, so n-1 degrees of freedom
    se_coef = np.sqrt(mse / np.sum(X.flatten() ** 2))
    
    # Calculate t-statistic and approximate p-value
    t_stat = ar1.coef_[0] / se_coef
    
    # Display results in R-style format
    print("Call:")
    print("lm(formula = yt ~ yt_1 - 1, data = y_em)")
    print()
    print("Coefficients:")
    print(f"{'yt_1':>6}")
    print(f"{ar1.coef_[0]:6.4f}")
    print()
    
    # Additional detailed output
    print("Detailed Results:")
    print(f"Coefficient (ρ): {ar1.coef_[0]:.6f}")
    print(f"Standard Error:  {se_coef:.6f}")
    print(f"t-value:        {t_stat:.4f}")
    print(f"R-squared:      {r_squared:.6f}")
    print(f"Residual SE:    {np.sqrt(mse):.6f} on {n-1} degrees of freedom")
    print(f"Number of obs:  {n}")
    
    return {
        'model': ar1,
        'coefficient': ar1.coef_[0],
        'standard_error': se_coef,
        't_statistic': t_stat,
        'r_squared': r_squared,
        'residuals': residuals,
        'fitted_values': y_pred,
        'mse': mse,
        'n_obs': n
    }







##################################################################################################



# ====================================
#      chunk_ar_shuffle
# ====================================

def chunk_ar_shuffle(y_em, seed=None):
    """
    Shuffle embedded AR data and fit regression to demonstrate spurious correlation.
    
    Parameters:
    -----------
    y_em : pd.DataFrame
        Embedded time series matrix with 'yt' and 'yt_1' columns
    seed : int, optional
        Random seed for reproducible shuffling
        
    Returns:
    --------
    dict : Shuffled data and regression results
    """
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle the rows (equivalent to R's sample with replace=FALSE)
    n_rows = len(y_em)
    ind = np.random.choice(n_rows, size=n_rows, replace=False)
    y_em_sh = y_em.iloc[ind].copy()
    
    # Reset index after shuffling
    y_em_sh = y_em_sh.reset_index(drop=True)
    
    # Extract variables for regression
    X = y_em_sh[['yt_1']].values  # Predictor (shuffled lagged values)
    y = y_em_sh['yt'].values      # Response (shuffled current values)
    
    # Fit linear regression without intercept
    ar1 = LinearRegression(fit_intercept=False)
    ar1.fit(X, y)
    
    # Calculate additional statistics
    y_pred = ar1.predict(X)
    residuals = y - y_pred
    
    # Calculate R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate standard error of coefficient
    n = len(y)
    mse = ss_res / (n - 1)
    se_coef = np.sqrt(mse / np.sum(X.flatten() ** 2))
    
    # Calculate t-statistic
    t_stat = ar1.coef_[0] / se_coef
    
    # Display results in R-style format
    print("# Shuffled Data Analysis")
    print("Call:")
    print("lm(formula = yt ~ yt_1 - 1, data = y_em_sh)")
    print()
    print("Coefficients:")
    print(f"{'yt_1':>6}")
    print(f"{ar1.coef_[0]:6.4f}")
    print()
    
    # Additional detailed output
    print("Detailed Results (Shuffled):")
    print(f"Coefficient (ρ): {ar1.coef_[0]:.6f}")
    print(f"Standard Error:  {se_coef:.6f}")
    print(f"t-value:        {t_stat:.4f}")
    print(f"R-squared:      {r_squared:.6f}")
    print(f"Residual SE:    {np.sqrt(mse):.6f} on {n-1} degrees of freedom")
    print(f"Number of obs:  {n}")
    print()
    print("Note: Shuffling breaks the temporal relationship,")
    print("showing coefficient should be close to 0 for independent data.")
    
    return {
        'shuffled_data': y_em_sh,
        'shuffle_indices': ind,
        'model': ar1,
        'coefficient': ar1.coef_[0],
        'standard_error': se_coef,
        't_statistic': t_stat,
        'r_squared': r_squared,
        'residuals': residuals,
        'fitted_values': y_pred,
        'mse': mse,
        'n_obs': n
    }







##################################################################################################



# ====================================
#   chunk_multivariate_embed
# ====================================

def chunk_multivariate_embed(Y=None, X=None, embed_dim=3):
    """
    Create multivariate embedded time series matrix (equivalent to R's embed on matrix).
    
    Parameters:
    -----------
    Y : array-like, optional
        First time series (defaults to 1:10)
    X : array-like, optional  
        Second time series (defaults to 21:30)
    embed_dim : int
        Embedding dimension
        
    Returns:
    --------
    pd.DataFrame : Multivariate embedded time series matrix
    """
    
    def embed_multivariate(data_matrix, dimension):
        """
        Embed multivariate time series into matrix of lagged values.
        Equivalent to R's embed() function applied to a matrix.
        """
        data_matrix = np.array(data_matrix)
        n_obs, n_vars = data_matrix.shape
        
        if dimension > n_obs:
            raise ValueError(f"Embedding dimension ({dimension}) cannot exceed number of observations ({n_obs})")
        
        # Create embedded matrix
        n_embedded_obs = n_obs - dimension + 1
        n_embedded_cols = n_vars * dimension
        
        embedded = np.zeros((n_embedded_obs, n_embedded_cols))
        
        for lag in range(dimension):
            for var in range(n_vars):
                col_idx = lag * n_vars + var
                start_idx = dimension - 1 - lag
                end_idx = n_obs - lag
                embedded[:, col_idx] = data_matrix[start_idx:end_idx, var]
        
        return embedded
    
    # Create default data if not provided
    if Y is None:
        Y = np.arange(1, 11)  # 1:10 in R
    if X is None:
        X = np.arange(21, 31)  # 21:30 in R
    
    # Create matrix (equivalent to cbind in R)
    tsdf = np.column_stack([Y, X])
    
    print("Original matrix (tsdf):")
    tsdf_df = pd.DataFrame(tsdf, columns=['Y', 'X'])
    print(tsdf_df.to_string())
    
    # Create embedded matrix
    embedded_matrix = embed_multivariate(tsdf, embed_dim)
    
    # Create column names matching R output exactly
    col_names = []
    var_names = ['y', 'x']
    
    for lag in range(embed_dim):
        for var_name in var_names:
            if lag == 0:
                col_names.append(f"{var_name}(t)")
            else:
                col_names.append(f"{var_name}(t-{lag})")
    
    # Convert to DataFrame
    first = pd.DataFrame(embedded_matrix, columns=col_names)
    
    print(f"\nEmbedded matrix (dimension {embed_dim}):")
    print("head(first):")
    print(first.head(6).to_string())
    
    return first







##################################################################################################


# ====================================
#   chunk_multivariate_lags
# ====================================

def chunk_multivariate_lags(first):
    """
    Create different lag structures from embedded multivariate data.
    
    Parameters:
    -----------
    first : pd.DataFrame
        Embedded multivariate time series from chunk_multivariate_embed
        
    Returns:
    --------
    dict : Dictionary containing first, second, and third lag structures
    """
    
    n = len(first)
    
    # Second structure: Remove first row, remove last column from remaining rows
    # Equivalent to: cbind(first[-1,1], first[-n,-1])
    second_y = first.iloc[1:, 0]  # first[-1,1] - y(t) from row 2 onwards
    second_x = first.iloc[:-1, 1:].values  # first[-n,-1] - all X cols except last row
    
    # Combine and create DataFrame
    second_data = np.column_stack([second_y.values, second_x])
    second = pd.DataFrame(second_data, columns=["y(t)", "x(t-1)", "y(t-2)", "x(t-2)", "y(t-3)", "x(t-3)"])
    
    # Third structure: Remove first two rows, remove last two rows from remaining columns
    # Equivalent to: cbind(first[-c(1:2),1], first[-c(n,n-1),-1])
    third_y = first.iloc[2:, 0]  # first[-c(1:2),1] - y(t) from row 3 onwards
    third_x = first.iloc[:-2, 1:].values  # first[-c(n,n-1),-1] - all X cols except last 2 rows
    
    # Combine and create DataFrame
    third_data = np.column_stack([third_y.values, third_x])
    third = pd.DataFrame(third_data, columns=["y(t)", "x(t-2)", "y(t-3)", "x(t-3)", "y(t-4)", "x(t-4)"])
    
    # Display results
    print("head(first):")
    print(first.head(6).to_string())
    print()
    
    print("head(second):")
    print(second.head(6).to_string())
    print()
    
    print("head(third):")
    print(third.head(6).to_string())
    print()
    
    return {
        'first': first,
        'second': second,
        'third': third
    }








##################################################################################################



# ====================================
#    chunk_remove_colnames
# ====================================

def chunk_remove_colnames(first):
    """
    Remove column names from DataFrame (equivalent to colnames(first) <- NULL).
    
    Parameters:
    -----------
    first : pd.DataFrame
        DataFrame with column names to remove
        
    Returns:
    --------
    pd.DataFrame : DataFrame without column names (numeric column indices)
    """
    
    # Create a copy to avoid modifying the original
    first_no_names = first.copy()
    
    # Remove column names by setting them to None (equivalent to colnames(first) <- NULL)
    first_no_names.columns = range(len(first_no_names.columns))
    
    return first_no_names







##################################################################################################


# ================================================
#              chunk_row_shift
# ================================================

def chunk_row_shift(first):
    """
    Combines first column (excluding first row) with remaining columns (excluding last row).
    Equivalent to R: cbind(first[-1,1], first[-nrow(first),-1])
    """
    if isinstance(first, pd.DataFrame):
        col1 = first.iloc[1:, 0].reset_index(drop=True)
        remaining_cols = first.iloc[:-1, 1:].reset_index(drop=True)
        return pd.concat([col1, remaining_cols], axis=1)
    elif isinstance(first, np.ndarray):
        col1 = first[1:, 0:1]
        remaining_cols = first[:-1, 1:]
        return np.hstack([col1, remaining_cols])
    else:
        raise TypeError("Input must be pandas DataFrame or numpy array")








##################################################################################################



# ================================================
#              chunk_prepare_data
# ================================================

def chunk_prepare_data(trdf):
    """
    Prepares data by creating DataFrame with dcases and dmob columns,
    removing incomplete cases, and converting to matrix.
    Equivalent to R data preparation script.
    """
    df = pd.DataFrame({
        'dcases': trdf['diffcases'],
        'dmob': trdf['diffmob']
    })
    
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.values
    
    print(f"First 6 rows:\n{df[:6]}")
    return df








##################################################################################################


# ================================================
#              chunk_forecast
# ================================================

def chunk_forecast(df, h=3, w=3):
    """
    Creates h-step ahead forecasts using embedded time series data.
    Equivalent to R forecasting script with embed() and iterative lm().
    """
    fh = []
    
    # Create embedded matrix (equivalent to embed(df, w))
    n = len(df)
    dt = np.array([df[i:i+w] for i in range(n-w+1)])
    
    y = dt[:, 0].copy()
    X = dt[:, 1:].copy()
    
    for i in range(h):
        model = LinearRegression()
        model.fit(X, y)
        fitted_values = model.predict(X)
        fh.append(fitted_values[-1])
        
        y = y[1:]
        X = X[:-1]
    
    print(f"Forecasts: {fh}")
    return np.array(fh)







##################################################################################################


# ================================================
#              chunk_plot_forecast
# ================================================

def chunk_plot_forecast(trdf, fh):
    """
    Plots original time series data and forecast values.
    Equivalent to R plot() and lines() functions.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    x_orig = np.arange(1, 267)
    plt.plot(x_orig, trdf['diffcases'], color='red', linewidth=1)
    
    # Plot forecast
    x_forecast = np.arange(267, 270)
    plt.plot(x_forecast, fh, color='blue', linewidth=1)
    
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Time Series with Forecast')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return plt.gcf()







##################################################################################################



# ================================================
#              chunk_grid_forecast
# ================================================

def chunk_grid_forecast(df, h=7, w_range=None):
    """
    Creates forecasts using a grid of window sizes with train/test split.
    Equivalent to R script with embed() and nested loops for window size testing.
    """
    if w_range is None:
        w_range = list(range(3, 15))
    
    # Split data
    train = df[:258]
    test = df[258:]
    
    fh = np.zeros((len(w_range), h))
    
    for s, w in enumerate(w_range):
        # Create embedded matrix
        n = len(train)
        dt = np.array([train[i:i+w] for i in range(n-w+1)])
        
        y = dt[:, 0].copy()
        X = dt[:, 1:].copy()
        
        for i in range(h):
            model = LinearRegression()
            model.fit(X, y)
            fitted_values = model.predict(X)
            fh[s, i] = fitted_values[-1]
            
            y = y[1:]
            X = X[:-1]
    
    # Create DataFrame with proper labels
    fh_df = pd.DataFrame(fh, index=w_range, columns=range(1, h+1))
    print(f"Forecast matrix shape: {fh_df.shape}")
    
    return fh_df








##################################################################################################


# ================================================
#              chunk_rmspe_evaluation
# ================================================

def chunk_rmspe_evaluation(fh, test):
    """
    Calculates RMSPE for each window size and finds the best performing one.
    Equivalent to R RMSPE calculation and which.min() function.
    """
    rmspe = []
    
    for i in range(len(fh)):
        mse = np.mean((fh.iloc[i, :] - test.flatten()) ** 2)
        rmspe.append(np.sqrt(mse))
    
    rmspe = np.array(rmspe)
    best_window_idx = np.argmin(rmspe)
    
    print(f"RMSPE values: {rmspe}")
    print(f"Best window size index: {best_window_idx}")
    
    return rmspe, best_window_idx