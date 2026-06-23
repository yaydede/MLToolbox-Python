import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# ====================================
#           chunk_ag1
# ====================================

def chunk_ag1(data_path=None, data_dict=None):
    """
    Load Toronto data, create time series, and apply Box-Cox transformation
    using Guerrero method for optimal lambda selection.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to data file (if loading from file)
    data_dict : dict, optional
        Dictionary containing 'mob' and 'cases' data
        
    Returns:
    --------
    pd.DataFrame : Time series dataframe with Box-Cox transformed cases
    """
    
    def guerrero_lambda(x, seasonal_periods=7):
        """Estimate optimal Box-Cox lambda using Guerrero method"""
        def guerrero_cv(lam):
            if lam == 0:
                transformed = np.log(x + 1e-8)
            else:
                transformed = (np.power(x + 1e-8, lam) - 1) / lam
            
            # Split into seasonal periods
            n_seasons = len(transformed) // seasonal_periods
            if n_seasons < 2:
                return np.inf
                
            seasonal_data = transformed[:n_seasons * seasonal_periods].reshape(n_seasons, seasonal_periods)
            seasonal_means = np.mean(seasonal_data, axis=1)
            seasonal_stds = np.std(seasonal_data, axis=1, ddof=1)
            
            # Guerrero criterion: minimize CV of seasonal standard deviations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = np.std(seasonal_stds, ddof=1) / np.mean(seasonal_stds) if np.mean(seasonal_stds) > 0 else np.inf
            
            return cv if not np.isnan(cv) else np.inf
        
        # Search for optimal lambda
        result = minimize_scalar(guerrero_cv, bounds=(-1, 2), method='bounded')
        return result.x if result.success else 0
    
    def box_cox_transform(x, lambda_val):
        """Apply Box-Cox transformation"""
        x = np.array(x)
        x = np.maximum(x, 1e-8)  # Ensure positive values
        
        if abs(lambda_val) < 1e-8:
            return np.log(x)
        else:
            return (np.power(x, lambda_val) - 1) / lambda_val
    
    # Create date range
    day = pd.date_range(start='2020-03-01', end='2020-11-21', freq='D')
    
    # Load or use provided data
    if data_dict is not None:
        mob_data = data_dict['mob']
        cases_data = data_dict['cases']
    else:
        # Placeholder for data loading - adapt based on your data format
        print("Note: data_path loading not implemented - provide data_dict parameter")
        return None
    
    # Create DataFrame
    tdata = pd.DataFrame({
        'Day': day,
        'mob': mob_data,
        'cases': cases_data
    })
    
    # Convert to time series with Day as index
    toronto = tdata.set_index('Day')
    
    # Calculate optimal lambda using Guerrero method
    lmbd = guerrero_lambda(toronto['cases'].values)
    
    # Apply Box-Cox transformation
    toronto['boxcases'] = box_cox_transform(toronto['cases'], lmbd)
    
    return toronto







######################################################################################################




# ====================================
#           chunk_ag2
# ====================================

def chunk_ag2(toronto, target_col='boxcases'):
    """
    Perform SARIMA grid search optimization using AICc and RMSE criteria.
    
    Parameters:
    -----------
    toronto : pd.DataFrame
        Time series data with datetime index
    target_col : str
        Name of target column to model
        
    Returns:
    --------
    dict : Best models and parameter combinations
    """
    
    # Define parameter ranges
    p_range = range(0, 4)
    q_range = range(0, 4) 
    P_range = range(0, 4)
    Q_range = range(0, 3)
    
    # Create parameter combinations
    combinations = list(product(p_range, q_range, P_range, Q_range))
    
    # Remove unstable combinations
    # Remove (0,0,0,0) and (0,0,*,*) patterns
    combinations = [combo for combo in combinations if not (combo[0] == 0 and combo[1] == 0)]
    combinations = combinations[1:]  # Remove first combination
    
    # Convert to DataFrame for easier handling
    comb_df = pd.DataFrame(combinations, columns=['p', 'q', 'P', 'Q'])
    
    aicc_values = []
    rmse_values = []
    
    for idx, (p, q, P, Q) in enumerate(combinations):
        try:
            # Fit SARIMA model with seasonal period of 7 (daily data)
            model = ARIMA(
                toronto[target_col], 
                order=(p, 1, q),
                seasonal_order=(P, 1, Q, 7),
                trend=None
            )
            
            fitted_model = model.fit()
            
            # Calculate AICc
            aicc = fitted_model.aicc
            
            # Calculate RMSE from residuals  
            residuals = fitted_model.resid
            rmse = np.sqrt(np.mean(residuals**2))
            
            aicc_values.append(aicc)
            rmse_values.append(rmse)
            
        except Exception:
            aicc_values.append(np.nan)
            rmse_values.append(np.nan)
    
    # Add results to combinations DataFrame
    comb_df['AICc'] = aicc_values
    comb_df['RMSE'] = rmse_values
    
    # Find best models
    valid_aicc = comb_df.dropna(subset=['AICc'])
    valid_rmse = comb_df.dropna(subset=['RMSE'])
    
    best_aicc_idx = valid_aicc['AICc'].idxmin()
    best_rmse_idx = valid_rmse['RMSE'].idxmin()
    
    best_aicc_params = comb_df.loc[best_aicc_idx, ['p', 'q', 'P', 'Q']].to_dict()
    best_rmse_params = comb_df.loc[best_rmse_idx, ['p', 'q', 'P', 'Q']].to_dict()
    
    best_aicc_params['AICc'] = comb_df.loc[best_aicc_idx, 'AICc']
    best_rmse_params['RMSE'] = comb_df.loc[best_rmse_idx, 'RMSE']
    
    print(f"Best AICc: {best_aicc_params}")
    print(f"Best RMSE: {best_rmse_params}")
    
    return {
        'combinations': comb_df,
        'best_aicc': best_aicc_params,
        'best_rmse': best_rmse_params,
        'aicc_values': aicc_values,
        'rmse_values': rmse_values
    }








######################################################################################################




# ====================================
#           chunk_ag3
# ====================================

def chunk_ag3(toronto, target_col='boxcases', train_end='2020-11-14', h=7, plot=True):
    """
    Perform out-of-sample SARIMA grid search using training data, then forecast
    and evaluate on test data. Plot best model forecast.
    
    Parameters:
    -----------
    toronto : pd.DataFrame
        Time series data with datetime index
    target_col : str
        Name of target column to model
    train_end : str
        End date for training data split
    h : int
        Forecast horizon
    plot : bool
        Whether to create forecast plot
        
    Returns:
    --------
    dict : Best model parameters, forecast, and results
    """
    
    # Define parameter ranges
    p_range = range(0, 4)
    q_range = range(0, 4)
    P_range = range(0, 4) 
    Q_range = range(0, 3)
    
    # Create parameter combinations
    combinations = list(product(p_range, q_range, P_range, Q_range))
    
    # Remove unstable combinations
    combinations = [combo for combo in combinations if not (combo[0] == 0 and combo[1] == 0)]
    combinations = combinations[1:]  # Remove first combination
    
    # Convert to DataFrame
    comb_df = pd.DataFrame(combinations, columns=['p', 'q', 'P', 'Q'])
    
    # Split data
    train = toronto[toronto.index <= train_end].copy()
    test = toronto[toronto.index > train_end].copy()
    
    rmse_values = []
    
    for idx, (p, q, P, Q) in enumerate(combinations):
        try:
            # Fit SARIMA model on training data
            model = ARIMA(
                train[target_col],
                order=(p, 1, q),
                seasonal_order=(P, 1, Q, 7),
                trend=None
            )
            
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=h)
            
            # Calculate RMSE against actual test values
            test_values = test[target_col].iloc[:h].values
            forecast_values = forecast.values[:len(test_values)]
            
            rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
            rmse_values.append(rmse)
            
        except Exception:
            rmse_values.append(np.nan)
    
    # Add RMSE to combinations DataFrame
    comb_df['RMSE'] = rmse_values
    
    # Find best model
    valid_rmse = comb_df.dropna(subset=['RMSE'])
    best_rmse_idx = valid_rmse['RMSE'].idxmin()
    best_params = comb_df.loc[best_rmse_idx, ['p', 'q', 'P', 'Q']].to_dict()
    best_rmse = comb_df.loc[best_rmse_idx, 'RMSE']
    
    print(f"Best model: p={best_params['p']}, q={best_params['q']}, P={best_params['P']}, Q={best_params['Q']}, RMSE={best_rmse:.4f}")
    
    # Fit and forecast with best model
    best_model = ARIMA(
        train[target_col],
        order=(best_params['p'], 1, best_params['q']),
        seasonal_order=(best_params['P'], 1, best_params['Q'], 7),
        trend=None
    ).fit()
    
    best_forecast = best_model.forecast(steps=h)
    
    # Create forecast dates
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=h,
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'forecast': best_forecast.values
    }, index=forecast_index)
    
    # Plot results
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(toronto.index, toronto[target_col], 'k-', label='Actual', alpha=0.8)
        plt.plot(train.index, train[target_col], 'b-', label='Training Data', alpha=0.6)
        plt.plot(forecast_df.index, forecast_df['forecast'], 'r--', 
                label=f'SARIMA({best_params["p"]},1,{best_params["q"]})×({best_params["P"]},1,{best_params["Q"]})₇', 
                linewidth=2)
        plt.axvline(x=pd.to_datetime(train_end), color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
        plt.legend()
        plt.title('SARIMA Forecast - Best Model')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.tight_layout()
        plt.show()
    
    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'forecast': forecast_df,
        'combinations': comb_df,
        'fitted_model': best_model,
        'train_data': train,
        'test_data': test
    }