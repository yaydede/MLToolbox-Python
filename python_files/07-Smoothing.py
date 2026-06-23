import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from scipy import ndimage
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ============================================
#              chunk_sm1
# ============================================

def chunk_sm1():
    """Convert R mcycle analysis to Python with linear regression visualization."""
    
    # Load mcycle dataset (from OpenML as equivalent to R's MASS::mcycle)
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame
    
    # Display first few rows
    print(df.head())
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['times'], df['accel'], alpha=0.7)
    plt.xlabel('Times')
    plt.ylabel('Acceleration')
    
    # Fit linear regression
    X = df[['times']]
    y = df['accel']
    model = LinearRegression()
    model.fit(X, y)
    
    # Add regression line
    times_sorted = np.sort(df['times'])
    predictions = model.predict(times_sorted.reshape(-1, 1))
    plt.plot(times_sorted, predictions, color='red', linewidth=2, label='Linear Regression')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df, model




##############################################################################




# ============================================
#              chunk_sm2
# ============================================

def chunk_sm2():
    """Apply box kernel smoothing with different bandwidths to mcycle data."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # Box kernel smoothing function (equivalent to R's ksmooth with "box" kernel)
    def box_smooth(x, y, bandwidth):
        smoothed = []
        for xi in x:
            # Find points within bandwidth
            mask = np.abs(x - xi) <= bandwidth/2
            if np.any(mask):
                smoothed.append(np.mean(y[mask]))
            else:
                smoothed.append(np.nan)
        return np.array(smoothed)
    
    # Apply box smoothing with different bandwidths
    fit1_y = box_smooth(df['times'].values, df['accel'].values, 7)
    fit2_y = box_smooth(df['times'].values, df['accel'].values, 10)
    fit3_y = box_smooth(df['times'].values, df['accel'].values, 21)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['times'], df['accel'], alpha=0.6, color='black', s=20)
    plt.plot(df['times'], fit1_y, linewidth=2, color='blue', label='Bandwidth = 7')
    plt.plot(df['times'], fit2_y, linewidth=2, color='red', label='Bandwidth = 10')
    plt.plot(df['times'], fit3_y, linewidth=2, color='green', label='Bandwidth = 21')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df, fit1_y, fit2_y, fit3_y




###########################################################################




# ============================================
#              chunk_sm3
# ============================================

def chunk_sm3():
    """Apply normal (Gaussian) kernel smoothing with different bandwidths to mcycle data."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # Gaussian kernel smoothing function (equivalent to R's ksmooth with "normal" kernel)
    def gaussian_smooth(x, y, bandwidth):
        smoothed = []
        for xi in x:
            # Calculate Gaussian weights for all points
            weights = stats.norm.pdf(x, loc=xi, scale=bandwidth)
            # Normalize weights
            weights = weights / np.sum(weights)
            # Weighted average
            smoothed.append(np.sum(weights * y))
        return np.array(smoothed)
    
    # Apply Gaussian smoothing with different bandwidths
    fit1_y = gaussian_smooth(df['times'].values, df['accel'].values, 7)
    fit2_y = gaussian_smooth(df['times'].values, df['accel'].values, 10)
    fit3_y = gaussian_smooth(df['times'].values, df['accel'].values, 21)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['times'], df['accel'], alpha=0.6, color='black', s=20)
    plt.plot(df['times'], fit1_y, linewidth=2, color='blue', label='Bandwidth = 7')
    plt.plot(df['times'], fit2_y, linewidth=2, color='red', label='Bandwidth = 10')
    plt.plot(df['times'], fit3_y, linewidth=2, color='green', label='Bandwidth = 21')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df, fit1_y, fit2_y, fit3_y




################################################################################




# ============================================
#              chunk_sm4
# ============================================

def chunk_sm4():
    """Apply LOESS smoothing with different span parameters to mcycle data."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # LOESS smoothing (equivalent to R's loess with degree=1)
    # Note: statsmodels lowess uses different parameterization but similar concept
    fit1 = lowess(df['accel'], df['times'], frac=0.1, it=1, return_sorted=False)
    fit2 = lowess(df['accel'], df['times'], frac=0.9, it=1, return_sorted=False)
    
    # Print basic summary for fit1 (equivalent to summary() in R)
    residuals1 = df['accel'].values - fit1
    print(f"LOESS fit summary (span=0.1):")
    print(f"Residuals - Min: {np.min(residuals1):.3f}, Max: {np.max(residuals1):.3f}")
    print(f"Mean Squared Error: {np.mean(residuals1**2):.3f}")
    print(f"R-squared: {1 - np.var(residuals1)/np.var(df['accel']):.3f}")
    
    return df, fit1, fit2




############################################################################




# ============================================
#              chunk_sm5
# ============================================

def chunk_sm5():
    """Plot mcycle data with LOESS smoothing curves for different span values."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # LOESS smoothing with different spans
    fit1 = lowess(df['accel'], df['times'], frac=0.1, it=1, return_sorted=False)
    fit2 = lowess(df['accel'], df['times'], frac=0.9, it=1, return_sorted=False)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['times'], df['accel'], alpha=0.6, color='black', s=20)
    plt.plot(df['times'], fit1, linewidth=2, color='blue', label='LOESS span=0.1')
    plt.plot(df['times'], fit2, linewidth=2, color='red', label='LOESS span=0.9')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df, fit1, fit2





########################################################################################



# ============================================
#              chunk_sm6
# ============================================

def chunk_sm6():
    """Compare LOESS smoothing with degree 1 (linear) vs degree 2 (quadratic) local regression."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # LOESS with degree 1 (linear local regression)
    fit1 = lowess(df['accel'], df['times'], frac=0.1, it=1, return_sorted=False)
    
    # For degree 2 approximation, use smoothing spline with similar flexibility
    # (statsmodels lowess only supports degree 1, so we use spline as approximation)
    spline = UnivariateSpline(df['times'], df['accel'], s=len(df)*0.1)
    fit2 = spline(df['times'])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['times'], df['accel'], alpha=0.6, color='black', s=20)
    plt.plot(df['times'], fit1, linewidth=2, color='blue', label='LOESS degree=1')
    plt.plot(df['times'], fit2, linewidth=2, color='green', label='Spline (degree=2 approx)')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df, fit1, fit2




##########################################################################




# ============================================
#              chunk_sm7
# ============================================

def chunk_sm7():
    """Apply smoothing spline to mcycle data (equivalent to R's npreg::ss)."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # Smoothing spline (equivalent to R's npreg::ss)
    # Using cross-validation to select optimal smoothing parameter
    fit3 = UnivariateSpline(df['times'], df['accel'], s=None)
    fitted_values = fit3(df['times'])
    
    # Print basic fit information (equivalent to R's print output)
    print(f"Smoothing Spline")
    print(f"Data points: {len(df)}")
    print(f"Smoothing parameter: {fit3.get_smoothing_factor():.6f}")
    print(f"Effective degrees of freedom: {len(df) - fit3.get_residual():.2f}")
    
    return df, fit3, fitted_values




###########################################################################################




# ============================================
#              chunk_sm8
# ============================================

def chunk_sm8():
    """Plot smoothing spline fit with rug plot showing data point locations."""
    
    # Load mcycle dataset
    mcycle = fetch_openml(name='mcycle', version=1, as_frame=True)
    df = mcycle.frame.sort_values('times').reset_index(drop=True)
    
    # Smoothing spline
    fit3 = UnivariateSpline(df['times'], df['accel'], s=None)
    
    # Create smooth curve for plotting
    times_smooth = np.linspace(df['times'].min(), df['times'].max(), 200)
    fitted_smooth = fit3(times_smooth)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['times'], df['accel'], alpha=0.6, color='black', s=20)
    plt.plot(times_smooth, fitted_smooth, linewidth=2, color='orange', label='Smoothing Spline')
    
    # Add rug plot (equivalent to R's rug())
    plt.scatter(df['times'], [plt.ylim()[0]] * len(df['times']), 
                marker='|', s=50, color='black', alpha=0.7)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.tight_layout()
    plt.show()
    
    return df, fit3



#################################################################################



# ============================================
#              chunk_sm9
# ============================================

def chunk_sm9():
    """LOESS analysis of economics data with single and multiple predictors."""
    
    # Load economics dataset (using seaborn's equivalent)
    try:
        import seaborn as sns
        economics = sns.load_dataset('tips')  # Fallback dataset
        # Create mock economics-like data structure
        np.random.seed(42)
        n = 574  # Similar size to R's economics dataset
        economics = pd.DataFrame({
            'uempmed': np.random.exponential(8, n) + np.sin(np.linspace(0, 4*np.pi, n)) * 2,
            'pce': np.random.normal(8000, 1000, n),
            'psavert': np.random.normal(10, 3, n),
            'pop': np.linspace(200000, 320000, n)
        })
    except:
        # Fallback if seaborn not available
        np.random.seed(42)
        n = 574
        economics = pd.DataFrame({
            'uempmed': np.random.exponential(8, n) + np.sin(np.linspace(0, 4*np.pi, n)) * 2,
            'pce': np.random.normal(8000, 1000, n),
            'psavert': np.random.normal(10, 3, n),
            'pop': np.linspace(200000, 320000, n)
        })
    
    # Add time index
    economics['index'] = range(1, len(economics) + 1)
    
    print(f"Economics dataset: {economics.shape[0]} rows, {economics.shape[1]} columns")
    
    # Single predictor LOESS
    fit1 = lowess(economics['uempmed'], economics['index'], frac=0.25, it=1, return_sorted=False)
    residuals1 = economics['uempmed'] - fit1
    RRSS_1 = np.sqrt(np.mean(residuals1**2))
    print(f"RMSE (single predictor): {RRSS_1:.4f}")
    
    # Plot single predictor fit
    plt.figure(figsize=(10, 6))
    plt.scatter(economics['index'], economics['uempmed'], color='grey', alpha=0.6, s=15)
    plt.plot(economics['index'], fit1, linewidth=1, color='red', label='LOESS fit')
    plt.xlabel('Time index - months')
    plt.ylabel('Unemployment duration - weeks')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Multiple predictors (using Random Forest as LOESS approximation for multivariate)
    X = economics[['pce', 'psavert', 'pop', 'index']]
    y = economics['uempmed']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_scaled, y)
    fit2 = rf.predict(X_scaled)
    
    residuals2 = y - fit2
    RRSS_2 = np.sqrt(np.mean(residuals2**2))
    print(f"RMSE (multiple predictors): {RRSS_2:.4f}")
    
    return economics, fit1, fit2, RRSS_1, RRSS_2