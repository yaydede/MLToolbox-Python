import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import timedelta
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.deterministic import DeterministicProcess


# ============================================
#              chunk_ar1
# ============================================

def chunk_ar1(data_dict):
    """
    Convert R tsibble processing to Python DataFrame.
    
    Args:
        data_dict: Dictionary containing 'mob' and 'cases' data
    
    Returns:
        pandas.DataFrame: Time series data with Day, mob, and cases columns
    """
    # Generate date range
    start_date = datetime(2020, 3, 1)
    end_date = datetime(2020, 11, 21)
    day_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create DataFrame (equivalent to tibble + as_tsibble)
    toronto = pd.DataFrame({
        'Day': day_range,
        'mob': data_dict['mob'],
        'cases': data_dict['cases']
    })
    
    # Set Day as index for time series functionality
    toronto.set_index('Day', inplace=True)
    
    return toronto







################################################################################################


# ============================================
#              chunk_ar2
# ============================================

def chunk_ar2(toronto_df):
    """
    Create side-by-side time series plots for mobility and COVID-19 cases.
    
    Args:
        toronto_df: DataFrame with Day index and mob, cases columns
    
    Returns:
        matplotlib.figure.Figure: Figure object with subplots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot COVID-19 cases (left subplot - equivalent to 'b')
    ax1.plot(toronto_df.index, toronto_df['cases'], color='red', linewidth=1.5)
    ax1.set_title('Covid-19 Cases\nToronto 2020')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Cases')
    ax1.grid(True, alpha=0.3)
    
    # Plot mobility index (right subplot - equivalent to 'a')
    ax2.plot(toronto_df.index, toronto_df['mob'], color='blue', linewidth=1.5)
    ax2.set_title('Mobility Index\nToronto 2020')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Index')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig










################################################################################################


# ============================================
#              chunk_ar3
# ============================================

def chunk_ar3(toronto_df):
    """
    Apply Guerrero method to find optimal Box-Cox lambda and plot transformed cases.
    
    Args:
        toronto_df: DataFrame with Day index and cases column
    
    Returns:
        tuple: (lambda_guerrero, matplotlib.figure.Figure)
    """
    cases = toronto_df['cases'].values
    
    # Guerrero method for optimal lambda
    def guerrero_objective(lmbda):
        if lmbda == 0:
            transformed = np.log(cases)
        else:
            transformed = (np.power(cases, lmbda) - 1) / lmbda
        
        # Split into periods and calculate coefficient of variation
        n_periods = 12  # seasonal periods
        period_len = len(cases) // n_periods
        cv_sum = 0
        
        for i in range(n_periods):
            start_idx = i * period_len
            end_idx = (i + 1) * period_len if i < n_periods - 1 else len(cases)
            period_data = transformed[start_idx:end_idx]
            if len(period_data) > 1:
                cv_sum += np.std(period_data) / np.abs(np.mean(period_data))
        
        return cv_sum / n_periods
    
    # Find optimal lambda
    result = minimize_scalar(guerrero_objective, bounds=(-1, 2), method='bounded')
    lmbd = result.x
    
    # Apply Box-Cox transformation
    if lmbd == 0:
        transformed_cases = np.log(cases)
    else:
        transformed_cases = (np.power(cases, lmbd) - 1) / lmbd
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(toronto_df.index, transformed_cases, color='red', linewidth=1.5)
    ax.set_title(f'Cases - Transformed with λ = {lmbd:.2f}', fontsize=14)
    ax.set_ylabel('')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return lmbd, fig








################################################################################################



# ============================================
#              chunk_ar4
# ============================================

def chunk_ar4(toronto_df):
    """
    Perform unit root tests: number of differences needed and KPSS tests.
    
    Args:
        toronto_df: DataFrame with cases column
    
    Returns:
        dict: Test results including ndiffs, KPSS level, and KPSS first difference
    """
    cases = toronto_df['cases'].dropna()
    
    # Number of differences needed (similar to unitroot_ndiffs)
    def unitroot_ndiffs(series, max_d=2):
        """Determine number of differences needed for stationarity"""
        for d in range(max_d + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()
            
            # Use ADF test for stationarity
            adf_stat, adf_pvalue = adfuller(test_series)[:2]
            if adf_pvalue <= 0.05:  # Stationary at 5% level
                return d
        return max_d
    
    ndiffs = unitroot_ndiffs(cases)
    
    # KPSS test on level
    kpss_level_stat, kpss_level_pval = kpss(cases, regression='c')[:2]
    
    # First difference and KPSS test on difference
    diff_cases = cases.diff().dropna()
    kpss_diff_stat, kpss_diff_pval = kpss(diff_cases, regression='c')[:2]
    
    results = {
        'ndiffs': ndiffs,
        'kpss_level_stat': kpss_level_stat,
        'kpss_level_pval': kpss_level_pval,
        'kpss_diff_stat': kpss_diff_stat,
        'kpss_diff_pval': kpss_diff_pval
    }
    
    print(f"Number of differences needed: {ndiffs}")
    print(f"KPSS test (level) - Statistic: {kpss_level_stat:.4f}, p-value: {kpss_level_pval:.4f}")
    print(f"KPSS test (first diff) - Statistic: {kpss_diff_stat:.4f}, p-value: {kpss_diff_pval:.4f}")
    
    return results







################################################################################################

# ============================================
#              chunk_ar5
# ============================================

def chunk_ar5(toronto_df, lmbd):
    """
    Create 2x2 grid of ACF plots for different transformations of cases data.
    
    Args:
        toronto_df: DataFrame with cases column
        lmbd: Box-Cox lambda parameter from chunk_ar3
    
    Returns:
        matplotlib.figure.Figure: Figure with 2x2 ACF subplots
    """
    cases = toronto_df['cases'].dropna()
    
    # Box-Cox transformation
    if lmbd == 0:
        bc_cases = np.log(cases)
    else:
        bc_cases = (np.power(cases, lmbd) - 1) / lmbd
    
    # Prepare data series
    series_data = {
        'level': cases,
        'fdiff': cases.diff().dropna(),
        'diffbc': pd.Series(bc_cases).diff().dropna(),
        'ddiff': pd.Series(bc_cases).diff().diff().dropna()
    }
    
    titles = [
        'Covid-19 Cases',
        'First-difference', 
        'First-difference Box-Cox',
        'Double-difference Box-Cox'
    ]
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot ACF for each series
    for i, (key, series) in enumerate(series_data.items()):
        plot_acf(series, ax=axes[i], lags=20, alpha=0.05, 
                title=f'ACF - {titles[i]}', auto_ylims=True)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig







################################################################################################


# ============================================
#              chunk_ar6
# ============================================

def chunk_ar6(toronto_df, lmbd):
    """
    Determine number of seasonal differences needed for stationarity.
    
    Args:
        toronto_df: DataFrame with cases column
        lmbd: Box-Cox lambda parameter
    
    Returns:
        dict: Number of seasonal differences for original and Box-Cox transformed data
    """
    cases = toronto_df['cases'].dropna()
    
    # Box-Cox transformation
    if lmbd == 0:
        bc_cases = np.log(cases)
    else:
        bc_cases = (np.power(cases, lmbd) - 1) / lmbd
    
    def unitroot_nsdiffs(series, seasonal_period=7, max_D=1):
        """
        Determine number of seasonal differences needed for stationarity.
        Uses KPSS test on seasonally differenced data.
        """
        if len(series) < seasonal_period * 2:
            return 0
            
        for D in range(max_D + 1):
            if D == 0:
                test_series = series
            else:
                # Apply seasonal differencing
                test_series = series.diff(seasonal_period * D).dropna()
            
            if len(test_series) < seasonal_period:
                return D
                
            try:
                # KPSS test for stationarity (null: stationary)
                kpss_stat, kpss_pval = kpss(test_series, regression='c')[:2]
                if kpss_pval > 0.05:  # Fail to reject null (stationary)
                    return D
            except:
                return D
                
        return max_D
    
    # Test original cases
    nsdiffs_original = unitroot_nsdiffs(cases)
    
    # Test Box-Cox transformed cases
    bc_series = pd.Series(bc_cases, index=toronto_df.index[:len(bc_cases)])
    nsdiffs_boxcox = unitroot_nsdiffs(bc_series)
    
    results = {
        'nsdiffs_original': nsdiffs_original,
        'nsdiffs_boxcox': nsdiffs_boxcox
    }
    
    print(f"Seasonal differences needed (original): {nsdiffs_original}")
    print(f"Seasonal differences needed (Box-Cox): {nsdiffs_boxcox}")
    
    return results







################################################################################################


# ============================================
#              chunk_ar7
# ============================================

def chunk_ar7(toronto_df, lmbd):
    """
    Create time series display with seasonal and first differencing.
    Shows time series plot, ACF, and PACF in a 3-panel layout.
    
    Args:
        toronto_df: DataFrame with cases column
        lmbd: Box-Cox lambda parameter
    
    Returns:
        matplotlib.figure.Figure: Figure with 3 subplots
    """
    cases = toronto_df['cases'].dropna()
    
    # Box-Cox transformation
    if lmbd == 0:
        bc_cases = np.log(cases)
    else:
        bc_cases = (np.power(cases, lmbd) - 1) / lmbd
    
    # Apply seasonal differencing (lag=7) then first differencing
    bc_series = pd.Series(bc_cases, index=toronto_df.index[:len(bc_cases)])
    seasonal_diff = bc_series.diff(7).dropna()  # Seasonal differencing
    final_series = seasonal_diff.diff().dropna()  # First differencing
    
    # Create 3-panel plot layout
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time series plot
    axes[0].plot(final_series.index, final_series, color='black', linewidth=1)
    axes[0].set_title('Seasonal & first differenced')
    axes[0].set_ylabel('')
    axes[0].grid(True, alpha=0.3)
    
    # ACF plot
    plot_acf(final_series, ax=axes[1], lags=36, alpha=0.05, 
             title='ACF', auto_ylims=True)
    axes[1].grid(True, alpha=0.3)
    
    # PACF plot  
    plot_pacf(final_series, ax=axes[2], lags=36, alpha=0.05,
              title='PACF', auto_ylims=True)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig








################################################################################################


# ============================================
#              chunk_ar8
# ============================================

def chunk_ar8(toronto_df):
    """
    Extract STL decomposition features from time series data.
    
    Args:
        toronto_df: DataFrame with cases column
    
    Returns:
        pandas.DataFrame: Transposed DataFrame with STL features
    """
    cases = toronto_df['cases'].dropna()
    
    # STL decomposition (seasonal_period=7 for daily data)
    stl = STL(cases, seasonal=7, trend=None, robust=True)
    result = stl.fit()
    
    # Extract components
    seasonal = result.seasonal
    trend = result.trend
    remainder = result.resid
    
    # Calculate STL features (equivalent to R's feat_stl)
    features = {}
    
    # Trend strength
    detrended = cases - trend
    features['trend_strength'] = max(0, 1 - np.var(remainder) / np.var(detrended))
    
    # Seasonal strength  
    deseasoned = cases - seasonal
    features['seasonal_strength_7'] = max(0, 1 - np.var(remainder) / np.var(deseasoned))
    
    # Seasonal peak and trough
    seasonal_avg = seasonal.groupby(seasonal.index.dayofweek).mean()
    features['seasonal_peak_7'] = seasonal_avg.idxmax()
    features['seasonal_trough_7'] = seasonal_avg.idxmin()
    
    # Spikiness (volatility of remainder)
    features['spikiness'] = np.var(remainder)
    
    # Linearity (strength of linear trend)
    time_index = np.arange(len(trend))
    if len(trend) > 1:
        linear_trend = np.polyfit(time_index, trend.fillna(method='linear'), 1)[0]
        features['linearity'] = abs(linear_trend)
    else:
        features['linearity'] = 0
    
    # Curvature (strength of quadratic trend)
    if len(trend) > 2:
        quad_coef = np.polyfit(time_index, trend.fillna(method='linear'), 2)[0]
        features['curvature'] = abs(quad_coef)
    else:
        features['curvature'] = 0
    
    # Create DataFrame and transpose (equivalent to t(t[1:2]))
    df = pd.DataFrame([features])
    result_df = df.iloc[:, :2].T  # First 2 columns transposed
    result_df.columns = ['Value']
    
    print(result_df)
    return result_df









################################################################################################



# ============================================
#              chunk_ar9
# ============================================

def chunk_ar9(toronto_df, lmbd):
    """
    Add Box-Cox transformed cases column and create time series display 
    with first differencing showing time series, ACF, and PACF plots.
    
    Args:
        toronto_df: DataFrame with cases column
        lmbd: Box-Cox lambda parameter
    
    Returns:
        tuple: (updated DataFrame, matplotlib.figure.Figure)
    """
    # Create copy to avoid modifying original
    toronto = toronto_df.copy()
    cases = toronto['cases'].dropna()
    
    # Box-Cox transformation
    if lmbd == 0:
        boxcases = np.log(cases)
    else:
        boxcases = (np.power(cases, lmbd) - 1) / lmbd
    
    # Add boxcases column to DataFrame
    toronto['boxcases'] = np.nan
    toronto.loc[:len(boxcases)-1, 'boxcases'] = boxcases
    
    # First difference of Box-Cox transformed data
    diff_boxcases = pd.Series(boxcases).diff().dropna()
    
    # Create 3-panel plot layout
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time series plot of differenced data
    axes[0].plot(diff_boxcases.index, diff_boxcases, color='black', linewidth=1)
    axes[0].set_title('First difference of Box-Cox transformed cases')
    axes[0].set_ylabel('')
    axes[0].grid(True, alpha=0.3)
    
    # ACF plot
    plot_acf(diff_boxcases, ax=axes[1], lags=20, alpha=0.05, 
             title='ACF', auto_ylims=True)
    axes[1].grid(True, alpha=0.3)
    
    # PACF plot
    plot_pacf(diff_boxcases, ax=axes[2], lags=20, alpha=0.05,
              title='PACF', auto_ylims=True)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return toronto, fig









################################################################################################



# ============================================
#              chunk_ar10
# ============================================

def chunk_ar10(toronto_df, lmbd):
    """
    Fit multiple ARIMA models and compare their performance.
    
    Args:
        toronto_df: DataFrame with boxcases column or cases column
        lmbd: Box-Cox lambda parameter
    
    Returns:
        dict: Model results and comparison DataFrame
    """
    # Get Box-Cox transformed data
    if 'boxcases' in toronto_df.columns:
        boxcases = toronto_df['boxcases'].dropna()
    else:
        cases = toronto_df['cases'].dropna()
        if lmbd == 0:
            boxcases = np.log(cases)
        else:
            boxcases = (np.power(cases, lmbd) - 1) / lmbd
    
    # Fit ARIMA models
    models = {}
    
    # AR2: ARIMA(2,1,0)x(3,1,0)
    try:
        ar2_model = ARIMA(boxcases, order=(2, 1, 0), seasonal_order=(3, 1, 0, 7))
        models['AR2'] = ar2_model.fit()
    except:
        models['AR2'] = None
    
    # MA3: ARIMA(0,1,2)x(0,1,3)  
    try:
        ma3_model = ARIMA(boxcases, order=(0, 1, 2), seasonal_order=(0, 1, 3, 7))
        models['MA3'] = ma3_model.fit()
    except:
        models['MA3'] = None
    
    # Auto ARIMA (simplified grid search)
    try:
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        # Grid search over reasonable parameter ranges
        for p in range(0, 4):
            for q in range(0, 4):
                for P in range(0, 3):
                    for Q in range(0, 3):
                        try:
                            temp_model = ARIMA(boxcases, order=(p, 1, q), 
                                             seasonal_order=(P, 1, Q, 7))
                            temp_fit = temp_model.fit()
                            if temp_fit.aic < best_aic:
                                best_aic = temp_fit.aic
                                best_model = temp_fit
                                best_order = (p, 1, q, P, 1, Q)
                        except:
                            continue
        
        models['auto'] = best_model
    except:
        models['auto'] = None
    
    # Create comparison DataFrame
    comparison_data = []
    for name, model in models.items():
        if model is not None:
            comparison_data.append({
                '.model': name,
                'AIC': model.aic,
                'AICc': model.aic + (2 * model.params.size * (model.params.size + 1)) / 
                       (len(boxcases) - model.params.size - 1),
                'BIC': model.bic
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('AICc')
    
    # Print model summaries (transposed format)
    print("Model Summaries (transposed):")
    for name, model in models.items():
        if model is not None:
            print(f"\n{name}:")
            print(f"  Order: {model.model.order}")
            print(f"  Seasonal Order: {model.model.seasonal_order}")
            print(f"  AIC: {model.aic:.2f}")
    
    print("\nModel Comparison (sorted by AICc):")
    print(comparison_df.to_string(index=False))
    
    # Report on MA3 model if available
    if models['MA3'] is not None:
        print(f"\nMA3 Model Report:")
        print(f"Order: ARIMA{models['MA3'].model.order}x{models['MA3'].model.seasonal_order}")
        print(f"AIC: {models['MA3'].aic:.2f}")
        print(f"BIC: {models['MA3'].bic:.2f}")
        print(f"Log Likelihood: {models['MA3'].llf:.2f}")
    
    return {
        'models': models,
        'comparison': comparison_df,
        'best_model': comparison_df.iloc[0]['.model'] if not comparison_df.empty else None
    }














################################################################################################


# ============================================
#              chunk_ar11
# ============================================

def chunk_ar11(models_dict):
    """
    Perform Ljung-Box test on model residuals for diagnostic checking.
    
    Args:
        models_dict: Dictionary containing fitted ARIMA models from chunk_ar10
    
    Returns:
        pandas.DataFrame: Ljung-Box test results for all models
    """
    results_list = []
    
    # Model order for degrees of freedom calculation
    model_dof = {
        'auto': None,  # Will be calculated from model
        'MA3': 5,      # ARIMA(0,1,2)x(0,1,3) has 5 parameters
        'AR2': 5       # ARIMA(2,1,0)x(3,1,0) has 5 parameters
    }
    
    for model_name in ['auto', 'MA3', 'AR2']:
        if model_name in models_dict and models_dict[model_name] is not None:
            model = models_dict[model_name]
            
            # Get residuals (equivalent to .innov in R)
            residuals = model.resid
            
            # Calculate degrees of freedom
            if model_name == 'auto':
                dof = len(model.params)
            else:
                dof = model_dof[model_name]
            
            # Ljung-Box test (lag=24, adjusted for dof)
            try:
                lb_result = acorr_ljungbox(residuals, lags=24, return_df=True, auto_lag=False)
                
                # Get the test statistic and p-value for lag 24
                lb_stat = lb_result['lb_stat'].iloc[-1]  # Last lag (24)
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                
                results_list.append({
                    '.model': model_name,
                    'lb_stat': lb_stat,
                    'lb_pvalue': lb_pvalue,
                    'dof': dof,
                    'lag': 24
                })
                
            except Exception as e:
                results_list.append({
                    '.model': model_name,
                    'lb_stat': np.nan,
                    'lb_pvalue': np.nan,
                    'dof': dof,
                    'lag': 24
                })
    
    # Create DataFrame (equivalent to rbind in R)
    results_df = pd.DataFrame(results_list)
    
    print("Ljung-Box Test Results (lag=24):")
    print("="*50)
    for _, row in results_df.iterrows():
        print(f"Model: {row['.model']}")
        print(f"  Ljung-Box Statistic: {row['lb_stat']:.4f}")
        print(f"  p-value: {row['lb_pvalue']:.4f}")
        print(f"  Degrees of Freedom: {row['dof']}")
        print(f"  Interpretation: {'Residuals appear random' if row['lb_pvalue'] > 0.05 else 'Evidence of autocorrelation'}")
        print()
    
    return results_df









################################################################################################

# ============================================
#              chunk_ar12
# ============================================

def chunk_ar12(models_dict):
    """
    Create residual diagnostic plots for MA3 model (equivalent to gg_tsresiduals).
    Shows residuals time series, ACF of residuals, and histogram of residuals.
    
    Args:
        models_dict: Dictionary containing fitted ARIMA models
    
    Returns:
        matplotlib.figure.Figure: Figure with 3 diagnostic plots
    """
    if 'MA3' not in models_dict or models_dict['MA3'] is None:
        print("MA3 model not available")
        return None
    
    model = models_dict['MA3']
    residuals = model.resid
    
    # Create 3-panel plot layout
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Time series plot of residuals
    axes[0].plot(residuals.index, residuals, color='black', linewidth=1)
    axes[0].set_title('Residuals')
    axes[0].set_ylabel('Residuals')
    axes[0].axhline(y=0, color='blue', linestyle='--', alpha=0.7)
    axes[0].grid(True, alpha=0.3)
    
    # 2. ACF plot of residuals (lag=36)
    plot_acf(residuals.dropna(), ax=axes[1], lags=36, alpha=0.05, 
             title='ACF of Residuals', auto_ylims=True)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Histogram of residuals with normal curve overlay
    axes[2].hist(residuals.dropna(), bins=20, density=True, alpha=0.7, 
                color='lightblue', edgecolor='black')
    
    # Overlay normal distribution curve
    mu, sigma = stats.norm.fit(residuals.dropna())
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
    
    axes[2].set_title('Histogram of Residuals')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print some diagnostic statistics
    print("MA3 Model Residual Diagnostics:")
    print("="*40)
    print(f"Mean of residuals: {residuals.mean():.6f}")
    print(f"Std of residuals: {residuals.std():.6f}")
    print(f"Skewness: {stats.skew(residuals.dropna()):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals.dropna()):.4f}")
    
    return fig











################################################################################################


# ============================================
#              chunk_ar13
# ============================================

def chunk_ar13(models_dict, toronto_df, h=7):
    """
    Generate forecasts from fitted ARIMA models.
    
    Args:
        models_dict: Dictionary containing fitted ARIMA models
        toronto_df: Original DataFrame to get the last date
        h: Forecast horizon (number of periods)
    
    Returns:
        dict: Forecast results for all models
    """
    # Get the last date from the original data
    last_date = toronto_df.index[-1] if hasattr(toronto_df, 'index') else toronto_df.iloc[-1, 0]
    
    # Create forecast dates
    if isinstance(last_date, pd.Timestamp):
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=h, freq='D')
    else:
        # If not datetime, create a simple range
        forecast_dates = range(len(toronto_df), len(toronto_df) + h)
    
    forecast_results = {}
    
    for model_name, model in models_dict.items():
        if model is not None:
            try:
                # Generate forecast
                forecast = model.forecast(steps=h)
                forecast_conf_int = model.get_forecast(steps=h).conf_int()
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    '.model': [model_name] * h,
                    'Day': forecast_dates,
                    '.mean': forecast.values,
                    '.lower': forecast_conf_int.iloc[:, 0].values,
                    '.upper': forecast_conf_int.iloc[:, 1].values
                })
                
                forecast_results[model_name] = forecast_df
                
            except Exception as e:
                print(f"Failed to generate forecast for {model_name}: {str(e)}")
                forecast_results[model_name] = None
    
    # Combine all forecasts into single DataFrame (like R's forecast output)
    if forecast_results:
        combined_forecasts = pd.concat([df for df in forecast_results.values() if df is not None], 
                                     ignore_index=True)
    else:
        combined_forecasts = pd.DataFrame()
    
    print("Forecast Results (h=7):")
    print("="*50)
    
    for model_name, df in forecast_results.items():
        if df is not None:
            print(f"\n{model_name} Model Forecasts:")
            print(df[['Day', '.mean', '.lower', '.upper']].to_string(index=False, float_format='%.4f'))
        else:
            print(f"\n{model_name}: No forecast available")
    
    return {
        'forecasts': forecast_results,
        'combined': combined_forecasts,
        'horizon': h
    }









################################################################################################


# ============================================
#              chunk_ar14
# ============================================

def chunk_ar14(forecast_results, toronto_df, lmbd):
    """
    Create forecast plots showing historical data and predictions.
    
    Args:
        forecast_results: Dictionary from chunk_ar13 containing forecasts
        toronto_df: DataFrame with historical data
        lmbd: Box-Cox lambda parameter
    
    Returns:
        tuple: (combined_figure, individual_figures_dict)
    """
    # Get Box-Cox transformed historical data
    if 'boxcases' in toronto_df.columns:
        historical_data = toronto_df['boxcases'].dropna()
    else:
        cases = toronto_df['cases'].dropna()
        if lmbd == 0:
            historical_data = np.log(cases)
        else:
            historical_data = (np.power(cases, lmbd) - 1) / lmbd
    
    # Create combined plot (all models)
    fig_combined, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical_data.index, historical_data, color='black', linewidth=1.5, label='Historical')
    
    # Plot forecasts for each model
    colors = {'auto': 'blue', 'MA3': 'red', 'AR2': 'green'}
    
    for model_name in ['auto', 'MA3', 'AR2']:
        if model_name in forecast_results['forecasts'] and forecast_results['forecasts'][model_name] is not None:
            fc_data = forecast_results['forecasts'][model_name]
            color = colors.get(model_name, 'gray')
            
            # Plot point forecast
            ax.plot(fc_data['Day'], fc_data['.mean'], color=color, linewidth=2, 
                   label=f'{model_name} forecast')
            
            # Plot confidence intervals (optional - remove level=NULL equivalent)
            ax.fill_between(fc_data['Day'], fc_data['.lower'], fc_data['.upper'], 
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Transformed Cases with Box-Cox')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('COVID-19 Forecasting - All Models')
    
    # Create individual plots for auto and MA3 models
    individual_figs = {}
    
    for model_name, title_suffix in [('auto', 'Auto'), ('MA3', 'MA3')]:
        if model_name in forecast_results['forecasts'] and forecast_results['forecasts'][model_name] is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            ax.plot(historical_data.index, historical_data, color='black', 
                   linewidth=1.5, label='Historical')
            
            # Plot forecast
            fc_data = forecast_results['forecasts'][model_name]
            color = colors[model_name]
            
            ax.plot(fc_data['Day'], fc_data['.mean'], color=color, linewidth=2, 
                   label='Forecast')
            ax.fill_between(fc_data['Day'], fc_data['.lower'], fc_data['.upper'], 
                           color=color, alpha=0.3, label='Confidence Interval')
            
            ax.set_xlabel('Days')
            ax.set_ylabel('Box-Cox Transformed Cases')
            ax.set_title(f'COVID-19 Forecasting - {title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            individual_figs[model_name] = fig
    
    # Create side-by-side plot (equivalent to grid.arrange)
    if 'auto' in individual_figs and 'MA3' in individual_figs:
        fig_side_by_side, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Auto model plot
        if 'auto' in forecast_results['forecasts'] and forecast_results['forecasts']['auto'] is not None:
            ax1.plot(historical_data.index, historical_data, color='black', linewidth=1.5)
            fc_auto = forecast_results['forecasts']['auto']
            ax1.plot(fc_auto['Day'], fc_auto['.mean'], color='blue', linewidth=2)
            ax1.fill_between(fc_auto['Day'], fc_auto['.lower'], fc_auto['.upper'], 
                            color='blue', alpha=0.3)
            ax1.set_title('COVID-19 Forecasting - Auto')
            ax1.set_ylabel('Box-Cox Transformed Cases')
            ax1.grid(True, alpha=0.3)
        
        # MA3 model plot  
        if 'MA3' in forecast_results['forecasts'] and forecast_results['forecasts']['MA3'] is not None:
            ax2.plot(historical_data.index, historical_data, color='black', linewidth=1.5)
            fc_ma3 = forecast_results['forecasts']['MA3']
            ax2.plot(fc_ma3['Day'], fc_ma3['.mean'], color='red', linewidth=2)
            ax2.fill_between(fc_ma3['Day'], fc_ma3['.lower'], fc_ma3['.upper'], 
                            color='red', alpha=0.3)
            ax2.set_title('COVID-19 Forecasting - MA3')
            ax2.set_ylabel('Box-Cox Transformed Cases')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig_combined, fig_side_by_side, individual_figs
    
    return fig_combined, None, individual_figs










################################################################################################




# ============================================
#              chunk_ar15
# ============================================

def chunk_ar15(toronto_df, lmbd):
    """
    Create training dataset and fit multiple ARIMA models with ensemble.
    
    Args:
        toronto_df: DataFrame with boxcases column
        lmbd: Box-Cox lambda parameter
    
    Returns:
        dict: Training data and fitted models including ensemble
    """
    # Filter training data up to 2020-11-14 (equivalent to filter_index)
    if hasattr(toronto_df.index, 'date'):
        # If index is datetime
        train_data = toronto_df[toronto_df.index <= pd.Timestamp('2020-11-14')]
    else:
        # If index is not datetime, use iloc for approximate filtering
        # Assuming daily data starting from 2020-03-01, 2020-11-14 would be around index 258
        cutoff_date = pd.Timestamp('2020-11-14')
        start_date = pd.Timestamp('2020-03-01')
        days_diff = (cutoff_date - start_date).days
        train_data = toronto_df.iloc[:days_diff + 1]
    
    # Get Box-Cox transformed training data
    if 'boxcases' in train_data.columns:
        train_boxcases = train_data['boxcases'].dropna()
    else:
        cases = train_data['cases'].dropna()
        if lmbd == 0:
            train_boxcases = np.log(cases)
        else:
            train_boxcases = (np.power(cases, lmbd) - 1) / lmbd
    
    # Fit ARIMA models on training data
    models = {}
    
    # AR2: ARIMA(2,1,0)x(3,1,0)
    try:
        ar2_model = ARIMA(train_boxcases, order=(2, 1, 0), seasonal_order=(3, 1, 0, 7))
        models['AR2'] = ar2_model.fit()
    except:
        models['AR2'] = None
        print("AR2 model failed to fit")
    
    # MA3: ARIMA(0,1,2)x(0,1,3)
    try:
        ma3_model = ARIMA(train_boxcases, order=(0, 1, 2), seasonal_order=(0, 1, 3, 7))
        models['MA3'] = ma3_model.fit()
    except:
        models['MA3'] = None
        print("MA3 model failed to fit")
    
    # Auto ARIMA (simplified grid search on training data)
    try:
        best_aic = float('inf')
        best_model = None
        
        for p in range(0, 4):
            for q in range(0, 4):
                for P in range(0, 3):
                    for Q in range(0, 3):
                        try:
                            temp_model = ARIMA(train_boxcases, order=(p, 1, q), 
                                             seasonal_order=(P, 1, Q, 7))
                            temp_fit = temp_model.fit()
                            if temp_fit.aic < best_aic:
                                best_aic = temp_fit.aic
                                best_model = temp_fit
                        except:
                            continue
        
        models['auto'] = best_model
    except:
        models['auto'] = None
        print("Auto model failed to fit")
    
    # Create ensemble/mixed model forecasts function
    def create_ensemble_forecast(models_dict, h=7):
        """Create ensemble forecast by averaging individual model forecasts"""
        forecasts = []
        valid_models = []
        
        for name, model in models_dict.items():
            if model is not None and name != 'mixed':
                try:
                    forecast = model.forecast(steps=h)
                    forecasts.append(forecast.values)
                    valid_models.append(name)
                except:
                    continue
        
        if forecasts:
            # Average the forecasts
            ensemble_forecast = np.mean(forecasts, axis=0)
            return ensemble_forecast, valid_models
        else:
            return None, []
    
    # Store ensemble function in models dict
    models['mixed'] = create_ensemble_forecast
    
    print("Training Models Fitted:")
    print("="*30)
    for name, model in models.items():
        if name != 'mixed':
            if model is not None:
                print(f"{name}: Successfully fitted")
                print(f"  AIC: {model.aic:.2f}")
            else:
                print(f"{name}: Failed to fit")
    
    if models['mixed']:
        print("mixed: Ensemble model ready (average of fitted models)")
    
    return {
        'train_data': train_data,
        'train_boxcases': train_boxcases,
        'models': models,
        'cutoff_date': '2020-11-14'
    }










################################################################################################


# ============================================
#              chunk_ar16
# ============================================

def chunk_ar16(training_results, toronto_df, lmbd, h=7):
    """
    Generate forecasts from training models, create plots, and calculate accuracy.
    
    Args:
        training_results: Dictionary from chunk_ar15 with training models
        toronto_df: Full DataFrame with test data
        lmbd: Box-Cox lambda parameter
        h: Forecast horizon
    
    Returns:
        dict: Forecasts, accuracy metrics, and plots
    """
    models = training_results['models']
    train_data = training_results['train_data']
    
    # Get test data (data after training cutoff)
    test_start_date = pd.Timestamp('2020-11-15')
    if hasattr(toronto_df.index, 'date'):
        test_data = toronto_df[toronto_df.index >= test_start_date]
    else:
        # Use iloc for approximate filtering
        cutoff_date = pd.Timestamp('2020-11-14')
        start_date = pd.Timestamp('2020-03-01')
        days_diff = (cutoff_date - start_date).days
        test_data = toronto_df.iloc[days_diff + 1:]
    
    # Get actual test values (Box-Cox transformed)
    if 'boxcases' in test_data.columns:
        actual_test = test_data['boxcases'].iloc[:h].dropna()
    else:
        test_cases = test_data['cases'].iloc[:h].dropna()
        if lmbd == 0:
            actual_test = np.log(test_cases)
        else:
            actual_test = (np.power(test_cases, lmbd) - 1) / lmbd
    
    # Generate forecasts from each model
    forecasts = {}
    forecast_dates = pd.date_range(start=test_start_date, periods=h, freq='D')
    
    # Individual model forecasts
    for model_name in ['AR2', 'MA3', 'auto']:
        if models[model_name] is not None:
            try:
                fc = models[model_name].forecast(steps=h)
                forecasts[model_name] = {
                    'forecast': fc.values,
                    'dates': forecast_dates
                }
            except:
                forecasts[model_name] = None
    
    # Ensemble forecast
    if models['mixed']:
        try:
            ensemble_fc, used_models = models['mixed'](models, h)
            if ensemble_fc is not None:
                forecasts['mixed'] = {
                    'forecast': ensemble_fc,
                    'dates': forecast_dates
                }
                print(f"Ensemble uses models: {used_models}")
        except:
            forecasts['mixed'] = None
    
    # Create forecast plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get full Box-Cox transformed historical data for plotting
    if 'boxcases' in toronto_df.columns:
        full_historical = toronto_df['boxcases'].dropna()
    else:
        cases = toronto_df['cases'].dropna()
        if lmbd == 0:
            full_historical = np.log(cases)
        else:
            full_historical = (np.power(cases, lmbd) - 1) / lmbd
    
    # Plot historical data
    ax.plot(full_historical.index, full_historical, color='black', linewidth=1.5, label='Actual')
    
    # Add vertical line at training cutoff
    cutoff_line = pd.Timestamp('2020-11-14')
    ax.axvline(x=cutoff_line, color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
    
    # Plot forecasts
    colors = {'AR2': 'green', 'MA3': 'red', 'auto': 'blue', 'mixed': 'purple'}
    
    for model_name, fc_data in forecasts.items():
        if fc_data is not None:
            ax.plot(fc_data['dates'], fc_data['forecast'], 
                   color=colors[model_name], linewidth=2, marker='o',
                   label=f'{model_name} forecast')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Box-Cox Transformed Cases')
    ax.set_title('COVID-19 Forecasting - Training Models (h=7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate accuracy metrics
    accuracy_results = []
    
    for model_name, fc_data in forecasts.items():
        if fc_data is not None and len(actual_test) > 0:
            forecast_values = fc_data['forecast'][:len(actual_test)]
            
            # Calculate accuracy metrics
            errors = actual_test.values - forecast_values
            me = np.mean(errors)  # Mean Error
            mae = np.mean(np.abs(errors))  # Mean Absolute Error
            mse = np.mean(errors**2)  # Mean Squared Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            mape = np.mean(np.abs(errors / actual_test.values)) * 100  # Mean Absolute Percentage Error
            
            accuracy_results.append({
                '.model': model_name,
                'ME': me,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            })
    
    accuracy_df = pd.DataFrame(accuracy_results)
    
    print("Forecast Accuracy (7-step ahead):")
    print("="*50)
    if not accuracy_df.empty:
        print(accuracy_df.to_string(index=False, float_format='%.4f'))
    else:
        print("No accuracy metrics available")
    
    return {
        'forecasts': forecasts,
        'accuracy': accuracy_df,
        'actual_test': actual_test,
        'figure': fig
    }







################################################################################################




# ====================================
#           chunk_ar17
# ====================================

def chunk_ar17(train, toronto, boxcases_col='boxcases', h=7, plot=True):
    """
    Fit mean and linear trend+seasonal models, forecast h periods ahead,
    and calculate accuracy metrics.
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data with datetime index
    toronto : pd.DataFrame  
        Test data for accuracy calculation
    boxcases_col : str
        Column name for the target variable
    h : int
        Forecast horizon
    plot : bool
        Whether to create forecast plot
        
    Returns:
    --------
    dict : Dictionary containing forecasts and accuracy metrics
    """
    
    # Mean model forecast
    mean_forecast = np.full(h, train[boxcases_col].mean())
    
    # Linear trend + seasonal model
    dp = DeterministicProcess(
        index=train.index,
        trend_order=1,
        seasonal=True,
        period=12 if len(train) > 24 else 4  # Assume monthly or quarterly
    )
    
    X_train = dp.in_sample()
    lm_model = LinearRegression().fit(X_train, train[boxcases_col])
    
    # Generate forecast dates
    forecast_index = pd.date_range(
        start=train.index[-1] + pd.Timedelta(days=1),
        periods=h,
        freq=train.index.freq or 'D'
    )
    
    dp_forecast = DeterministicProcess(
        index=forecast_index,
        trend_order=1,
        seasonal=True,
        period=12 if len(train) > 24 else 4
    )
    
    X_forecast = dp_forecast.in_sample()
    lm_forecast = lm_model.predict(X_forecast)
    
    # Create forecast dataframe
    forecasts = pd.DataFrame({
        'mean': mean_forecast,
        'lm': lm_forecast
    }, index=forecast_index)
    
    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(toronto.index, toronto[boxcases_col], 'k-', label='Actual', linewidth=2)
        plt.plot(forecasts.index, forecasts['mean'], 'b--', label='Mean', alpha=0.8)
        plt.plot(forecasts.index, forecasts['lm'], 'r--', label='Linear + Seasonal', alpha=0.8)
        plt.legend()
        plt.title('Forecast Comparison')
        plt.tight_layout()
        plt.show()
    
    # Calculate accuracy metrics
    test_values = toronto[boxcases_col].iloc[:h].values
    
    accuracy_results = {}
    for model_name in ['mean', 'lm']:
        pred_values = forecasts[model_name].values[:len(test_values)]
        accuracy_results[model_name] = {
            'MAE': mean_absolute_error(test_values, pred_values),
            'RMSE': np.sqrt(mean_squared_error(test_values, pred_values)),
            'MAPE': np.mean(np.abs((test_values - pred_values) / test_values)) * 100
        }
    
    return {
        'forecasts': forecasts,
        'accuracy': accuracy_results,
        'models': {'mean_model': mean_forecast, 'lm_model': lm_model}
    }