import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt



# ================================================
#              chunk_tsr1
# ================================================

def chunk_tsr1(data_path="~/Dropbox/ToolShed_draft/toronto2.rds"):
    """
    Loads RDS data and creates time series DataFrame with date index.
    Equivalent to R tsibble creation with fpp3 libraries.
    """
    # Note: For RDS loading, you'll need pyreadr: pip install pyreadr
    try:
        import pyreadr
        result = pyreadr.read_r(data_path)
        data = result[None]  # Default key for single object RDS files
    except ImportError:
        print("pyreadr not available. Please provide data as DataFrame.")
        return None
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return None
    
    # Create date range
    start_date = datetime(2020, 3, 1)
    end_date = datetime(2020, 11, 21)
    day = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create time series DataFrame
    toronto2 = pd.DataFrame({
        'Day': day,
        **data.iloc[:, 1:].to_dict('series')  # All columns except first
    })
    toronto2 = toronto2.set_index('Day')
    
    print(f"Toronto2 time series shape: {toronto2.shape}")
    return toronto2








##############################################################################################################


# ================================================
#              chunk_tsr2
# ================================================

def chunk_tsr2(toronto2):
    """
    Creates differenced variables and filters/reorders columns.
    Equivalent to R mutate() with difference() and column manipulation.
    """
    df = toronto2.copy()
    
    # Create differenced variables (equivalent to difference() in R)
    df['dcases'] = df['cases'].diff()
    df['dmob'] = df['mob'].diff()
    df['ddelay'] = df['delay'].diff()
    df['dmale'] = df['male'].diff()
    df['dtemp'] = df['temp'].diff()
    df['dhum'] = df['hum'].diff()
    
    # Remove level variables (columns 2:5, 7, 8 in R indexing)
    # R columns: cases(2), mob(3), delay(4), male(5), temp(7), hum(8)
    cols_to_remove = ['cases', 'mob', 'delay', 'male', 'temp', 'hum']
    dft = df.drop(columns=cols_to_remove)
    
    # Remove first row (due to differencing) and reorder columns
    dft = dft.iloc[1:].copy()
    
    # Reorder columns: keeping first col, then dcases, dmob, ddelay, dmale, dtemp, dhum
    desired_order = [col for col in dft.columns if col not in ['dcases', 'dmob', 'ddelay', 'dmale', 'dtemp', 'dhum']] + \
                   ['dcases', 'dmob', 'ddelay', 'dmale', 'dtemp', 'dhum']
    dft = dft[desired_order]
    
    print(f"Transformed data shape: {dft.shape}")
    return dft








##############################################################################################################


# ================================================
#              chunk_tsr3
# ================================================

def chunk_tsr3(dft, h=7, w_range=None):
    """
    Creates Random Forest forecasts using a grid of window sizes.
    Equivalent to R script with randomForest and embed() functions.
    """
    if w_range is None:
        w_range = list(range(3, 22))
    
    fh = np.zeros((len(w_range), h))
    
    for s, w in enumerate(w_range):
        # Create embedded matrix from second column (dcases)
        series = dft.iloc[:, 1].values
        n = len(series)
        dt = np.array([series[i:i+w] for i in range(n-w+1)])
        
        # Split train/test
        test_ind = len(dt) - h
        train = dt[:test_ind]
        test = dt[test_ind:]
        
        y = train[:, 0].copy()
        X = train[:, 1:].copy()
        
        for i in range(h):
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            fh[s, :] = rf.predict(test[:, 1:])
            
            y = y[1:]
            X = X[:-1]
    
    # Create DataFrame with proper labels
    fh_df = pd.DataFrame(fh, index=w_range, columns=range(1, h+1))
    print(f"Random Forest forecast matrix shape: {fh_df.shape}")
    
    return fh_df








##############################################################################################################


# ================================================
#              chunk_tsr4
# ================================================

def chunk_tsr4(fh, test):
    """
    Calculates RMSPE for Random Forest forecasts and finds best window size.
    Equivalent to R RMSPE calculation and which.min() function.
    """
    actual = test[:, 0]  # First column of test data
    rmspe = []
    
    for i in range(len(fh)):
        mse = np.mean((fh.iloc[i, :] - actual) ** 2)
        rmspe.append(np.sqrt(mse))
    
    rmspe = np.array(rmspe)
    best_window_idx = np.argmin(rmspe)
    
    print(f"RMSPE values: {rmspe}")
    print(f"Best window size index: {best_window_idx}")
    
    return rmspe, best_window_idx








##############################################################################################################


# ================================================
#              chunk_tsr5
# ================================================

def chunk_tsr5(actual, fh):
    """
    Plots actual vs forecast values for multiple window sizes.
    Equivalent to R plot() and lines() with legend.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot actual values
    plt.plot(actual, color='red', linewidth=3, label='Actual', linestyle='-')
    
    # Plot forecasts for different window sizes
    plt.plot(fh.iloc[0, :], color='blue', linewidth=1, label='3-day', linestyle='-')
    plt.plot(fh.iloc[1, :], color='green', linewidth=1, label='4-day', linestyle='-')
    plt.plot(fh.iloc[4, :], color='orange', linewidth=1, label='7-day', linestyle='-')
    plt.plot(fh.iloc[11, :], color='black', linewidth=1, label='14-day', linestyle='-')
    
    plt.ylim(-80, 50)
    plt.xlabel('Last 7 days')
    plt.ylabel('Actual (red) vs. Forecasts')
    plt.title('7-Day Forecasts')
    plt.legend(title='Lags', loc='lower right', fontsize=9, 
              frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()









##############################################################################################################


# ================================================
#              chunk_tsr6
# ================================================

def chunk_tsr6(dft, h=7, w_range=None):
    """
    Creates Random Forest forecasts using all variables except first column.
    Equivalent to R script with randomForest and embed() on dft[, -1].
    """
    if w_range is None:
        w_range = list(range(3, 15))
    
    fh = np.zeros((len(w_range), h))
    
    for s, w in enumerate(w_range):
        # Create embedded matrix from all columns except first
        data_matrix = dft.iloc[:, 1:].values  # Exclude first column
        n = len(data_matrix)
        
        # Embed multivariate data
        dt = []
        for i in range(n - w + 1):
            dt.append(data_matrix[i:i+w].flatten())
        dt = np.array(dt)
        
        # Split train/test
        test_ind = len(dt) - h
        train = dt[:test_ind]
        test = dt[test_ind:]
        
        y = train[:, 0].copy()  # First element is target
        X = train[:, 1:].copy()  # Rest are features
        
        for i in range(h):
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            fh[s, :] = rf.predict(test[:, 1:])
            
            y = y[1:]
            X = X[:-1]
    
    # Create DataFrame with proper labels
    fh_df = pd.DataFrame(fh, index=w_range, columns=range(1, h+1))
    print(f"Multivariate RF forecast matrix shape: {fh_df.shape}")
    
    return fh_df








##############################################################################################################



# ================================================
#              chunk_tsr7
# ================================================

def chunk_tsr7(actual, fh):
    """
    Plots actual vs multivariate forecast values for multiple window sizes.
    Equivalent to R plot() and lines() with legend for multivariate forecasts.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot actual values
    plt.plot(actual, color='red', linewidth=3, label='Actual', linestyle='-')
    
    # Plot forecasts for different window sizes
    plt.plot(fh.iloc[0, :], color='blue', linewidth=1, label='3-day', linestyle='-')
    plt.plot(fh.iloc[2, :], color='green', linewidth=1, label='5-day', linestyle='-')
    plt.plot(fh.iloc[4, :], color='orange', linewidth=1, label='7-day', linestyle='-')
    plt.plot(fh.iloc[11, :], color='black', linewidth=1, label='14-day', linestyle='-')
    
    plt.ylim(-80, 50)
    plt.xlabel('Last 7 days')
    plt.ylabel('Actual (red) vs. Forecasts')
    plt.title('7-Day Forecasts')
    plt.legend(title='Lags', loc='lower right', fontsize=9, 
              frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()








##############################################################################################################



# ================================================
#              chunk_tsr8
# ================================================

def chunk_tsr8(dft, l_range=None, ws=150):
    """
    Rolling window Random Forest forecasting with different embedding lags.
    Equivalent to R script with rolling windows and randomForest.
    """
    if l_range is None:
        l_range = list(range(3, 11))
    
    rmspe = []
    all_fh = []
    all_y = []
    
    for s, l in enumerate(l_range):
        # Create embedded matrix from all columns except first
        data_matrix = dft.iloc[:, 1:].values
        n = len(data_matrix)
        
        # Embed multivariate data
        dt = []
        for i in range(n - l + 1):
            dt.append(data_matrix[i:i+l].flatten())
        dt = np.array(dt)
        
        nwin = len(dt) - ws  # Number of rolling windows
        fh = []
        y = []
        
        for i in range(nwin):
            # Rolling window: moves one day forward each iteration
            train = dt[i:ws+i]
            test = dt[ws+i]
            
            # Set seed for reproducibility
            np.random.seed(i + s)
            rf = RandomForestRegressor(n_estimators=100, random_state=i+s)
            rf.fit(train[:, 1:], train[:, 0])
            
            pred = rf.predict(test[1:].reshape(1, -1))[0]
            fh.append(pred)
            y.append(test[0])  # Actual value
        
        all_y.append(y)
        all_fh.append(fh)
        
        err = np.array(y) - np.array(fh)
        rmspe.append(np.sqrt(np.mean(err ** 2)))
    
    rmspe = np.array(rmspe)
    bst = np.argmin(rmspe)
    
    print(f"RMSPE values: {rmspe}")
    print(f"Best lag: {l_range[bst]}")
    
    return rmspe, bst, all_fh, all_y, l_range








##############################################################################################################



# ================================================
#              chunk_tsr9
# ================================================

def chunk_tsr9(all_y, all_fh, bst):
    """
    Plots actual vs predicted values for best performing lag configuration.
    Creates two subplots: full series and zoomed last 50 days view.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: Full series
    ax1.plot(all_y[bst], color='red', linewidth=1, label='Actual')
    ax1.plot(all_fh[bst], color='blue', linewidth=1, label='Predicted')
    ax1.set_ylabel('Actual (red) vs Predicted (Blue)')
    ax1.set_xlabel('Days')
    ax1.set_title('1-Day-Ahead')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right subplot: Last 50 days (days 60-110)
    y_subset = all_y[bst][60:110]
    fh_subset = all_fh[bst][60:110]
    
    ax2.plot(y_subset, color='red', marker='o', linewidth=1, markersize=3, label='Actual')
    ax2.plot(fh_subset, color='blue', linewidth=1, label='Predicted')
    ax2.set_ylabel('Actual (red) vs Predicted (Blue)')
    ax2.set_xlabel('Days')
    ax2.set_title('Last 50 Days')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig








##############################################################################################################



# ================================================
#              chunk_tsr10
# ================================================

def chunk_tsr10(seed=321, n=10):
    """
    Demonstrates differencing and reconstruction using cumsum.
    Equivalent to R diff() and cumsum() operations.
    """
    np.random.seed(seed)
    y = np.random.normal(size=n)
    z = np.diff(y)  # First differences
    back = np.cumsum(np.concatenate([[y[0]], z]))  # Reconstruct using cumsum
    
    result = pd.DataFrame({
        'y': y,
        'back': back
    })
    
    print("Original vs Reconstructed:")
    print(result)
    
    return result







##############################################################################################################


# ================================================
#              chunk_tsr11
# ================================================

def chunk_tsr11(df, all_fh, bst, l_range, ws=150):
    """
    Back-transforms difference forecasts to original scale and plots results.
    Equivalent to R back-transformation using cumsum logic.
    """
    y = df['cases'].values
    l_bst = l_range[bst]  # Best lag value
    
    # Remove Y's until ws + l[bst] - 1 (day before first forecast)
    y_a_day_before = y[ws + l_bst - 1:]
    
    # Add predicted changes to observed values a day earlier
    back_forecast = y_a_day_before[:-1] + np.array(all_fh[bst])
    
    # Actual Y's in test set starting at ws + l[bst]
    ytest = y[ws + l_bst:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(ytest, color='blue', linewidth=1, label='Actual Y')
    plt.plot(back_forecast, color='red', linewidth=1, label='Forecast')
    plt.ylabel('Actual Y (Blue) vs Forecast (Red)')
    plt.xlabel('Days')
    plt.title('Back-transformed Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Forecast period: {len(back_forecast)} days")
    
    return back_forecast, ytest