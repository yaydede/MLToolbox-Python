import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from scipy.interpolate import BSpline
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from sklearn.metrics import r2_score
from patsy import bs
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml
from pyearth import Earth
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from pygam import LinearGAM, s
from sklearn.preprocessing import StandardScaler
from pygam import GAM, s
from mpl_toolkits.mplot3d import Axes3D
from pygam import GAM, s, l
from pygam import GAM, l

#========================================
#              chunk_nb1
#========================================

def chunk_nb1():
    # Random integers from 1 to 100
    np.random.seed(123)
    data = np.random.randint(1, 101, size=100)
    
    # Create histogram with 10 bins
    counts, bins, patches = plt.hist(data, bins=10, color='lightblue', alpha=0.7)
    plt.title('Histogram', fontsize=12)
    plt.tick_params(axis='both', labelsize=9)
    plt.show()
    
    # Calculate density (proportion of total)
    density = counts / len(data)
    
    print(f"Counts: {counts}")
    print(f"Density: {density}")
    print(f"Sum of density: {density.sum()}")
    
    return {'counts': counts, 'density': density, 'bins': bins}



#######################################################################################



#========================================
#              chunk_bn2
#========================================

def chunk_bn2(data):
    # Scale the data
    scaled_data = data / (10 * np.mean(data))
    
    # Create histogram with 25 bins
    counts, bins, patches = plt.hist(scaled_data, bins=25, alpha=0.7, xlim=(0, 0.2))
    plt.tick_params(axis='both', labelsize=9)
    
    # Calculate bin midpoints and density
    mids = (bins[:-1] + bins[1:]) / 2
    density = counts / (len(data) * (bins[1] - bins[0]))
    
    # Add density line (Naive)
    plt.plot(mids, density, color='blue', linewidth=2, label='Naive')
    
    plt.show()
    
    return {'counts': counts, 'density': density, 'mids': mids, 'bins': bins}



######################################################################################


#========================================
#              chunk_bn3
#========================================

def chunk_bn3(file_path="fes73.rds"):
    # Load and normalize data (assuming numpy array saved as .npy since Python can't read .rds directly)
    # For .rds files, you'd need rpy2 or convert to another format
    try:
        X = np.load(file_path.replace('.rds', '.npy'))
    except:
        print(f"Could not load {file_path}. Using simulated data.")
        np.random.seed(42)
        X = np.random.exponential(2, 1000)
    
    X = X / np.mean(X)
    filtered_X = X[X < 3.5]
    
    # Create histogram
    plt.hist(filtered_X, bins=130, density=True, color='white', edgecolor='black', alpha=0.7)
    plt.tick_params(axis='both', labelsize=9)
    
    # Generate x values for smooth density curves
    x_range = np.linspace(X.min(), X.max(), 1000)
    
    # Gaussian KDE with different bandwidths (adjust equivalent)
    kde_narrow = gaussian_kde(X, bw_method=lambda x: gaussian_kde.scott_factor(x) / 4)
    kde_default = gaussian_kde(X, bw_method='scott')
    kde_wide = gaussian_kde(X, bw_method=lambda x: gaussian_kde.scott_factor(x) * 4)
    
    # Rectangular kernel (uniform) with narrow bandwidth
    rect_kde = KernelDensity(kernel='tophat', bandwidth=kde_narrow.factor * np.std(X))
    rect_kde.fit(X.reshape(-1, 1))
    rect_density = np.exp(rect_kde.score_samples(x_range.reshape(-1, 1)))
    
    # Plot density lines
    plt.plot(x_range, kde_narrow(x_range), color='red', linewidth=2, label='BW/4')
    plt.plot(x_range, kde_default(x_range), color='blue', linewidth=2, label='Default BW')
    plt.plot(x_range, kde_wide(x_range), color='green', linewidth=2, label='BW×4')
    plt.plot(x_range, rect_density, color='black', linewidth=2, label='Rectangular BW/4')
    
    plt.legend()
    plt.show()
    
    # Print details of wide bandwidth KDE
    wide_kde_details = f"Bandwidth: {kde_wide.factor * np.std(X):.6f}, N: {len(X)}"
    print(f"Wide KDE details: {wide_kde_details}")
    
    return {'data': X, 'filtered_data': filtered_X, 'kde_wide': kde_wide}



##################################################################


#========================================
#              chunk_bn4
#========================================

def chunk_bn4(X):
    # Create KDE with bandwidth/4 (narrow bandwidth)
    poo = gaussian_kde(X, bw_method=lambda x: gaussian_kde.scott_factor(x) / 4)
    
    # Create interpolation function equivalent to R's approxfun
    x_range = np.linspace(X.min(), X.max(), 1000)
    y_density = poo(x_range)
    dens = interp1d(x_range, y_density, kind='linear', bounds_error=False, fill_value=0)
    
    # Plot the density curve
    plt.plot(x_range, y_density, color='blue', linewidth=2)
    
    # Specific points to evaluate
    x_new = np.array([0.5, 1.5, 2.2])
    y_new = dens(x_new)
    
    # Add points to plot
    plt.scatter(x_new, y_new, color='red', s=50, zorder=5)
    plt.show()
    
    # Evaluate density at specific point
    density_at_point = dens(1.832)
    print(f"Density at 1.832: {density_at_point:.6f}")
    
    return {'kde': poo, 'interpolator': dens, 'x_range': x_range, 'y_density': y_density}




###########################################################

#========================================
#              chunk_bn5
#========================================

def chunk_bn5():
    # Simulating data
    n = 300
    np.random.seed(1)
    x = np.sort(np.random.uniform(0, 2 * np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    
    # Create grid for predictions
    t = np.linspace(x.min(), x.max(), 100)
    
    # Local regression approximations using weighted least squares
    def local_regression(x_data, y_data, t_pred, degree=1, span=0.5):
        predictions = []
        bandwidth = span * (x_data.max() - x_data.min())
        
        for t_point in t_pred:
            # Calculate weights (tricube kernel approximation)
            distances = np.abs(x_data - t_point)
            weights = np.where(distances <= bandwidth, 
                             (1 - (distances/bandwidth)**3)**3, 0)
            
            if np.sum(weights) > 0:
                # Weighted polynomial regression
                if degree == 0:
                    pred = np.average(y_data, weights=weights)
                else:
                    # Create polynomial features
                    X_local = np.vander(x_data - t_point, degree + 1, increasing=True)



####################################################################


# ========================================
#              chunk_bn6
# ========================================

def chunk_bn6(x, y):
    """
    Create LOESS-style smoothed fits with different spans and plot results.
    
    Args:
        x: array-like, x coordinates
        y: array-like, y coordinates
    """
    x, y = np.array(x), np.array(y)
    
    # Sort data for smooth curves
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
    
    # Create smoothed fits (using splines as LOESS approximation)
    # span 0.05: very tight fit (low smoothing)
    fit0 = UnivariateSpline(x_sorted, y_sorted, s=len(x)*0.01)(x_sorted)
    
    # span 0.75: moderate smoothing (default-like)
    fit1 = UnivariateSpline(x_sorted, y_sorted, s=len(x)*0.5)(x_sorted)
    
    # span 2.0: heavy smoothing
    fit2 = UnivariateSpline(x_sorted, y_sorted, s=len(x)*2.0)(x_sorted)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='gray', alpha=0.6)
    plt.plot(x_sorted, fit0, linewidth=2, color='green', label='span=0.05')
    plt.plot(x_sorted, fit1, linewidth=2, color='red', label='span=0.75')
    plt.plot(x_sorted, fit2, linewidth=2, color='blue', label='span=2.0')
    
    plt.tick_params(axis='both', labelsize=9)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return fit0, fit1, fit2
            



##############################################################################

# ========================================
#              chunk_bn7
# ========================================

def chunk_bn7(file_path="irates.dat"):
    """
    Fit piecewise linear regression model with breakpoint at 10.8.
    
    Args:
        file_path: str, path to data file with TB3MS and GS10 columns
    """
    # Read data
    data = pd.read_csv(file_path, delim_whitespace=True)
    y = data['GS10'].values
    x = data['TB3MS'].values
    
    # Create piecewise term
    xk = (x - 10.8) * (x > 10.8)
    
    # Fit linear regression
    X = np.column_stack([np.ones(len(x)), x, xk])  # intercept, x, xk
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    # Get fitted values
    y_fitted = model.predict(X)
    
    # Print regression summary
    r2 = r2_score(y, y_fitted)
    residuals = y - y_fitted
    mse = np.mean(residuals**2)
    
    print(f"R-squared: {r2:.4f}")
    print(f"Coefficients: Intercept={model.coef_[0]:.4f}, x={model.coef_[1]:.4f}, xk={model.coef_[2]:.4f}")
    
    # Plot results
    reorder = np.argsort(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x[reorder], y_fitted[reorder], color='red', linewidth=2, label='Fitted line')
    plt.scatter(x, y, color='grey', alpha=0.6, s=20)
    plt.axvline(x=10.8, linestyle='--', color='darkgreen', linewidth=2, label='Breakpoint')
    
    plt.xlabel('TB3MS')
    plt.ylabel('GS10')
    plt.tick_params(axis='both', labelsize=9)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, y_fitted
                                                        

##################################################################################
                                                        

#========================================
#              chunk_bn8
#========================================

def chunk_bn8(x, y):
    # Define 5 knots
    k = np.array([2.8, 4.8, 6.8, 8.8, 10.8])
    
    # Create spline basis matrix
    # Xk = (x - k) * (x > k) for each knot
    Xk = np.zeros((len(x), len(k)))
    for i, knot in enumerate(k):
        Xk[:, i] = np.maximum(0, x - knot)
    
    # Fit linear regression with spline terms
    X = np.column_stack([np.ones(len(x)), x, Xk])
    model = sm.OLS(y, X).fit()
    fitted_values = model.fittedvalues
    
    # Create ordered plot to avoid messy lines
    reorder = np.argsort(x)
    
    # Plot
    plt.scatter(x, y, color='gray', alpha=0.6, s=20)
    plt.plot(x[reorder], fitted_values[reorder], linewidth=2, color='red', label='Fitted spline')
    
    # Add vertical lines at knots
    for knot in k:
        plt.axvline(x=knot, linestyle='--', color='darkgreen', alpha=0.7)
    
    plt.ylim(2.5, 14.5)
    plt.xlabel('TB3MS')
    plt.ylabel('GS10')
    plt.tick_params(axis='both', labelsize=9)
    plt.legend()
    plt.show()
    
    print(f"R-squared: {model.rsquared:.4f}")
    
    return {'x': x, 'y': y, 'knots': k, 'Xk': Xk, 'fitted': fitted_values, 'model': model}


###########################################################################################

#========================================
#              chunk_bn9
#========================================

def chunk_bn9(x, y):
    # Equidistant knots
    nknots = 5
    k = np.linspace(x.min(), x.max(), nknots + 2)[1:-1]  # Interior knots only
    
    # Create B-spline basis (degree 3, cubic)
    degree = 3
    # Full knot vector: add boundary knots with multiplicity
    knots_full = np.concatenate([
        np.repeat(x.min(), degree),
        k,
        np.repeat(x.max(), degree)
    ])
    
    # Create B-spline basis matrix
    n_basis = len(knots_full) - degree - 1
    B = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        # Create basis function
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1
        spl = BSpline(knots_full, coeffs, degree)
        B[:, i] = spl(x)
    
    # Fit model with B-spline basis
    X = np.column_stack([np.ones(len(x)), B])
    model = sm.OLS(y, X).fit()
    
    print("B-spline Regression Summary:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.2f}")
    
    # Prediction on grid
    u = np.linspace(x.min(), x.max(), 100)
    B_pred = np.zeros((len(u), n_basis))
    
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1
        spl = BSpline(knots_full, coeffs, degree)
        B_pred[:, i] = spl(u)
    
    X_pred = np.column_stack([np.ones(len(u)), B_pred])
    pred_fit = X_pred @ model.params
    
    # Calculate standard errors for predictions
    pred_se = np.sqrt(np.diag(X_pred @ model.cov_params() @ X_pred.T))
    
    # Plot
    plt.scatter(x, y, color='gray', alpha=0.6, s=20)
    plt.plot(u, pred_fit, linewidth=2, color='red', label='B-spline fit')
    plt.plot(u, pred_fit + 1.96 * pred_se, linestyle='--', color='black', alpha=0.7, label='95% CI')
    plt.plot(u, pred_fit - 1.96 * pred_se, linestyle='--', color='black', alpha=0.7)
    
    # Add vertical lines at knots
    for knot in k:
        plt.axvline(x=knot, linestyle='--', color='darkgreen', alpha=0.7)
    
    plt.ylim(2.5, 14.5)
    plt.xlabel('TB3MS')
    plt.ylabel('GS10')
    plt.tick_params(axis='both', labelsize=9)
    plt.legend()
    plt.show()
    
    return {'x': x, 'y': y, 'knots': k, 'model': model, 'u': u, 'pred_fit': pred_fit, 'pred_se': pred_se}

##########################################################################



# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn10
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn10(x, y):
    """
    Fit B-spline regression model and create diagnostic plot with confidence intervals.
    """
    # Fit B-spline model with degree 3 and 8 degrees of freedom
    X_spline = bs(x, degree=3, df=8)
    model = LinearRegression().fit(X_spline, y)
    
    # Generate prediction points
    u = np.linspace(x.min(), x.max(), 100)
    X_pred = bs(u, degree=3, df=8)
    
    # Make predictions
    y_pred = model.predict(X_pred)
    
    # Calculate standard errors for confidence intervals
    # Simplified approach using residual standard error
    residuals = y - model.predict(X_spline)
    mse = np.mean(residuals**2)
    se = np.sqrt(mse * np.diag(X_pred @ np.linalg.pinv(X_spline.T @ X_spline) @ X_pred.T))
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='gray', alpha=0.7)
    plt.plot(u, y_pred, color='red', linewidth=2, label='Fitted curve')
    plt.plot(u, y_pred + 1.96 * se, '--', color='black', alpha=0.7, label='95% CI')
    plt.plot(u, y_pred - 1.96 * se, '--', color='black', alpha=0.7)
    
    # Add knot lines
    knots = np.percentile(x, np.linspace(0, 100, 8-3+1)[1:-1])  # Interior knots for df=8, degree=3
    for knot in knots:
        plt.axvline(x=knot, linestyle='--', color='darkgreen', alpha=0.7)
    
    plt.ylim(2.5, 14.5)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, y_pred, se


##################################################################



# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn11
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn11(x, y):
    """
    Compare B-spline regression models with different degrees of freedom.
    """
    # Fit three B-spline models with different df
    X_spline1 = bs(x, degree=3, df=6)    # quartiles
    X_spline2 = bs(x, degree=3, df=12)   # deciles  
    X_spline3 = bs(x, degree=3, df=102)  # percentile
    
    model1 = LinearRegression().fit(X_spline1, y)
    model2 = LinearRegression().fit(X_spline2, y)
    model3 = LinearRegression().fit(X_spline3, y)
    
    # Make predictions
    pred1 = model1.predict(X_spline1)
    pred2 = model2.predict(X_spline2)
    pred3 = model3.predict(X_spline3)
    
    # Sort for smooth lines
    reorder = np.argsort(x)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='gray', alpha=0.7)
    plt.plot(x[reorder], pred1[reorder], color='red', linewidth=2, label='df=6 (quartiles)')
    plt.plot(x[reorder], pred2[reorder], color='blue', linewidth=2, label='df=12 (deciles)')  
    plt.plot(x[reorder], pred3[reorder], color='green', linewidth=2, label='df=102 (percentile)')
    
    plt.ylim(2.5, 14.5)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return (model1, model2, model3), (pred1, pred2, pred3)


##########################################################################



# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn12
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn12():
    """
    Generate synthetic data and compare smoothing splines with different degrees of freedom.
    """
    # Set seed and generate data
    np.random.seed(1)
    n = 200
    x = np.random.uniform(0, 1, n)
    dgm = np.sin(12 * (x + 0.2)) / (x + 0.2)  # data generating mechanism
    y = dgm + np.random.normal(0, 1, n)
    
    # Create smoothing splines
    spline1 = UnivariateSpline(x, y, s=len(x)-20)  # approximately df=20
    spline2 = UnivariateSpline(x, y, s=len(x)-40)  # approximately df=40
    
    # Generate smooth prediction points
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_spline1 = spline1(x_smooth)
    y_spline2 = spline2(x_smooth)
    
    # Sort for plotting true DGM
    order = np.argsort(x)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='gray', alpha=0.7)
    plt.plot(x[order], dgm[order], color='black', linewidth=2, label='True DGM')
    plt.plot(x_smooth, y_spline1, color='red', linewidth=2, label='Smooth spline (df≈20)')
    plt.plot(x_smooth, y_spline2, color='blue', linewidth=2, label='Smooth spline (df≈40)')
    
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return x, y, dgm, spline1, spline2



#######################################################################




# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn13
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn13(x, y, dgm):
    """
    Plot data with true DGM and optimal smoothing spline using cross-validation.
    """
    # Create smoothing spline with cross-validation (default behavior)
    spline_cv = UnivariateSpline(x, y)  # Uses cross-validation to select smoothing parameter
    
    # Generate smooth prediction points
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_spline_cv = spline_cv(x_smooth)
    
    # Sort for plotting true DGM
    order = np.argsort(x)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='gray', alpha=0.7)
    plt.plot(x[order], dgm[order], color='black', linewidth=2, label='True DGM')
    plt.plot(x_smooth, y_spline_cv, color='red', linewidth=2, label='Smooth spline (CV)')
    
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return spline_cv





##########################################################################################



# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn14
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn14():
    """
    Fit MARS (Multivariate Adaptive Regression Splines) model to Longley dataset.
    """
    # Load Longley dataset (classic econometric dataset)
    try:
        # Try to get from sklearn/openml
        longley = fetch_openml(name='longley', version=1, as_frame=True, parser='auto')
        df = longley.frame
        # Rename target column to match R
        if 'class' in df.columns:
            df = df.rename(columns={'class': 'Employed'})
        elif 'target' in df.columns:
            df = df.rename(columns={'target': 'Employed'})
    except:
        # Fallback: create the classic Longley dataset manually
        data = {
            'GNP.deflator': [83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9],
            'GNP': [234.289, 259.426, 258.054, 284.599, 328.975, 346.999, 365.385, 363.112, 397.469, 419.180, 442.769, 444.546, 482.704, 502.601, 518.173, 554.894],
            'Unemployed': [235.6, 232.5, 368.2, 335.1, 209.9, 193.2, 187.0, 357.8, 290.4, 282.2, 293.6, 468.1, 381.3, 393.2, 480.6, 400.7],
            'Armed.Forces': [159.0, 145.6, 161.6, 165.0, 309.9, 359.4, 354.7, 335.0, 304.8, 285.7, 279.8, 263.7, 255.2, 251.4, 257.2, 282.7],
            'Population': [107.608, 108.632, 109.773, 110.929, 112.075, 113.270, 115.094, 116.219, 117.388, 118.734, 120.445, 121.950, 123.366, 125.368, 127.852, 130.081],
            'Year': [1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962],
            'Employed': [60.323, 61.122, 60.171, 61.187, 63.221, 63.639, 64.989, 63.761, 66.019, 67.857, 68.169, 66.513, 68.655, 69.564, 69.331, 70.551]
        }
        df = pd.DataFrame(data)
    
    print("Dataset Summary:")
    print(df.describe())
    
    # Prepare features and target
    X = df.drop('Employed', axis=1)
    y = df['Employed']
    
    # Fit MARS model
    model = Earth(max_degree=2, max_terms=10)
    model.fit(X, y)
    
    print(f"\nMODEL SUMMARY:")
    print(f"Number of terms: {len(model.coef_)}")
    print(f"R-squared: {model.score(X, y):.4f}")
    
    # Variable importance (approximate)
    feature_importance = np.abs(model.coef_[1:])  # exclude intercept
    importance_df = pd.DataFrame({
        'Variable': X.columns,
        'Importance': feature_importance[:len(X.columns)]
    }).sort_values('Importance', ascending=False)
    
    print(f"\nVariable Importance:")
    print(importance_df)
    
    # Create diagnostic plot
    predictions = model.predict(X)
    residuals = y - predictions
    
    plt.figure(figsize=(10, 6))
    
    # Residuals vs Fitted
    plt.subplot(1, 2, 1)
    plt.scatter(predictions, residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    
    # Actual vs Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y, predictions, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate MSE
    mse = mean_squared_error(y, predictions)
    print(f"\nIn-sample MSE: {mse:.6f}")
    
    return model, predictions, mse




##########################################################################




# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn15
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn15():
    """
    Fit MARS models with different degrees on Ames Housing dataset.
    """
    try:
        # Try to load Ames dataset (you may need to download it separately)
        from sklearn.datasets import fetch_openml
        ames = fetch_openml(name="house_prices", as_frame=True)
        df = ames.frame
        # Rename target to match expected column name
        if 'SalePrice' in df.columns:
            df = df.rename(columns={'SalePrice': 'Sale_Price'})
        elif 'target' in df.columns:
            df = df.rename(columns={'target': 'Sale_Price'})
    except:
        # Fallback: create synthetic Ames-like data
        print("Creating synthetic housing data...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'Lot_Area': np.random.normal(10000, 3000, n),
            'Year_Built': np.random.randint(1950, 2020, n),
            'Total_Bsmt_SF': np.random.normal(1000, 300, n),
            'Gr_Liv_Area': np.random.normal(1500, 400, n),
            'Garage_Area': np.random.normal(500, 150, n),
            'Overall_Qual': np.random.randint(1, 11, n),
            'Overall_Cond': np.random.randint(1, 11, n)
        })
        # Create synthetic target based on features
        df['Sale_Price'] = (
            100 + 
            0.02 * df['Lot_Area'] + 
            50 * (df['Year_Built'] - 1950) +
            0.1 * df['Total_Bsmt_SF'] +
            0.08 * df['Gr_Liv_Area'] +
            0.05 * df['Garage_Area'] +
            5000 * df['Overall_Qual'] +
            2000 * df['Overall_Cond'] +
            np.random.normal(0, 10000, n)
        )
    
    # Handle missing values and ensure numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Prepare features and target
    if 'Sale_Price' not in df_numeric.columns:
        raise ValueError("Sale_Price column not found in dataset")
    
    X = df_numeric.drop('Sale_Price', axis=1)
    y = df_numeric['Sale_Price']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    # Fit basic MARS model (degree = 1, no interactions)
    print("\nFitting MARS model 1 (degree=1)...")
    ames1 = Earth(max_degree=1, max_terms=20)
    ames1.fit(X, y)
    
    # Fit MARS model with interactions (degree = 2)
    print("Fitting MARS model 2 (degree=2)...")
    ames2 = Earth(max_degree=2, max_terms=30)
    ames2.fit(X, y)
    
    # Model summaries
    print(f"\nMODEL 1 (degree=1):")
    print(f"Number of terms: {len(ames1.coef_)}")
    print(f"R-squared: {ames1.score(X, y):.4f}")
    
    print(f"\nMODEL 2 (degree=2):")
    print(f"Number of terms: {len(ames2.coef_)}")
    print(f"R-squared: {ames2.score(X, y):.4f}")
    
    # Make predictions
    predictions1 = ames1.predict(X)
    predictions2 = ames2.predict(X)
    
    # Calculate RMSE (note: original code had mean of sqrt of squared errors)
    rmse1 = np.sqrt(mean_squared_error(y, predictions1))
    rmse2 = np.sqrt(mean_squared_error(y, predictions2))
    
    print(f"\nModel 1 RMSE: ${rmse1:,.2f}")
    print(f"Model 2 RMSE: ${rmse2:,.2f}")
    
    return ames1, ames2, predictions1, predictions2, rmse1, rmse2




#######################################################################



# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn16
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn16(amesdata=None):
    """
    Perform grid search for MARS hyperparameters and create variable importance plot.
    """
    # Use data from previous function or create synthetic data
    if amesdata is None:
        print("Creating synthetic housing data...")
        np.random.seed(42)
        n = 1000
        amesdata = pd.DataFrame({
            'Lot_Area': np.random.normal(10000, 3000, n),
            'Year_Built': np.random.randint(1950, 2020, n),
            'Total_Bsmt_SF': np.random.normal(1000, 300, n),
            'Gr_Liv_Area': np.random.normal(1500, 400, n),
            'Garage_Area': np.random.normal(500, 150, n),
            'Overall_Qual': np.random.randint(1, 11, n),
            'Overall_Cond': np.random.randint(1, 11, n),
            'Bedroom_AbvGr': np.random.randint(1, 6, n),
            'Full_Bath': np.random.randint(1, 4, n),
            'Half_Bath': np.random.randint(0, 3, n)
        })
        # Create synthetic target
        amesdata['Sale_Price'] = (
            100 + 0.02 * amesdata['Lot_Area'] + 50 * (amesdata['Year_Built'] - 1950) +
            0.1 * amesdata['Total_Bsmt_SF'] + 0.08 * amesdata['Gr_Liv_Area'] +
            5000 * amesdata['Overall_Qual'] + np.random.normal(0, 10000, n)
        )
    
    # Prepare data
    X = amesdata.drop('Sale_Price', axis=1)
    y = amesdata['Sale_Price']
    
    print(f"Dataset shape: {X.shape}")
    
    # Create parameter grid
    degrees = [1, 2, 3]
    nprune_values = np.linspace(2, 50, 10).astype(int)  # Reduced for synthetic data
    
    grid_df = pd.DataFrame([
        {'degree': d, 'nprune': n} 
        for d in degrees for n in nprune_values
    ])
    print("Parameter grid (first 6 rows):")
    print(grid_df.head(6))
    
    # Grid search with cross-validation
    param_grid = {
        'max_degree': degrees,
        'max_terms': nprune_values
    }
    
    print("\nPerforming grid search with 5-fold CV...")
    np.random.seed(123)
    
    # RMSE scorer
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                             greater_is_better=False)
    
    mars_grid = GridSearchCV(
        Earth(),
        param_grid,
        cv=5,  # Reduced from 10 for faster execution
        scoring=rmse_scorer,
        n_jobs=-1
    )
    
    mars_grid.fit(X, y)
    
    print(f"Best parameters: {mars_grid.best_params_}")
    print(f"Best CV RMSE: ${-mars_grid.best_score_:,.2f}")
    
    # Get best model
    best_mars = mars_grid.best_estimator_
    
    # Create performance plot
    results_df = pd.DataFrame(mars_grid.cv_results_)
    results_df['mean_rmse'] = -results_df['mean_test_score']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance heatmap
    pivot_table = results_df.pivot_table(
        values='mean_rmse', 
        index='param_max_degree', 
        columns='param_max_terms'
    )
    
    im = ax1.imshow(pivot_table.values, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(pivot_table.columns)))
    ax1.set_xticklabels(pivot_table.columns)
    ax1.set_yticks(range(len(pivot_table.index)))
    ax1.set_yticklabels(pivot_table.index)
    ax1.set_xlabel('Max Terms (nprune)')
    ax1.set_ylabel('Max Degree')
    ax1.set_title('Cross-Validation RMSE')
    plt.colorbar(im, ax=ax1)
    
    # Variable importance plot (top 10 features)
    feature_importance = np.abs(best_mars.coef_[1:len(X.columns)+1])  # exclude intercept
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True).tail(10)
    
    ax2.barh(range(len(importance_df)), importance_df['Importance'])
    ax2.set_yticks(range(len(importance_df)))
    ax2.set_yticklabels(importance_df['Feature'])
    ax2.set_xlabel('Variable Importance (GCV)')
    ax2.set_title('Top 10 Variable Importance')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mars_grid, best_mars



###########################################################################


# ════════════════════════════════════════════════════════════════════════════════════════
#                                      chunk_bn17
# ════════════════════════════════════════════════════════════════════════════════════════

def chunk_bn17(data_file=None):
    """
    Fit GAM (Generalized Additive Model) to hedonic housing price data.
    """
    # Load or create hedonic housing data
    if data_file and data_file.endswith('.dat'):
        try:
            data = pd.read_csv(data_file, delim_whitespace=True)
        except:
            print("Could not load data file, creating synthetic hedonic data...")
            data = None
    else:
        data = None
    
    if data is None:
        # Create synthetic hedonic housing data
        print("Creating synthetic hedonic housing data...")
        np.random.seed(42)
        n = 500
        
        data = pd.DataFrame({
            'T1': np.random.binomial(1, 0.2, n),  # Time dummy 1
            'T2': np.random.binomial(1, 0.2, n),  # Time dummy 2
            'T3': np.random.binomial(1, 0.2, n),  # Time dummy 3
            'T4': np.random.binomial(1, 0.2, n),  # Time dummy 4
            'T5': np.random.binomial(1, 0.2, n),  # Time dummy 5
            'HOUSE': np.random.binomial(1, 0.7, n),  # House type dummy
            'PARKING': np.random.binomial(1, 0.6, n),  # Parking dummy
            'GREEN': np.random.uniform(0, 100, n),  # Green space proximity
            'COORD1': np.random.uniform(-5, 5, n),  # Spatial coordinate 1
            'COORD2': np.random.uniform(-5, 5, n)   # Spatial coordinate 2
        })
        
        # Create log price as function of features with nonlinear effects
        data['LPRIX'] = (
            10 + 0.1 * data['T1'] + 0.15 * data['T2'] + 0.2 * data['T3'] + 
            0.25 * data['T4'] + 0.3 * data['T5'] + 0.4 * data['HOUSE'] + 
            0.2 * data['PARKING'] + 
            0.05 * np.sin(data['GREEN'] / 10) +  # Nonlinear GREEN effect
            0.03 * (data['COORD1']**2 - 2) +     # Nonlinear COORD1 effect
            0.02 * np.cos(data['COORD2']) +      # Nonlinear COORD2 effect
            np.random.normal(0, 0.1, n)
        )
    
    print(f"Dataset shape: {data.shape}")
    print("\nDataset summary:")
    print(data.describe())
    
    # Prepare linear terms (dummy variables)
    X_linear = data[['T1', 'T2', 'T3', 'T4', 'T5', 'HOUSE', 'PARKING']].values
    
    # Prepare smooth terms
    green = data['GREEN'].values
    coord1 = data['COORD1'].values
    coord2 = data['COORD2'].values
    
    # Target variable
    y = data['LPRIX'].values
    
    # Combine all features for GAM
    X_all = np.column_stack([X_linear, green, coord1, coord2])
    
    # Fit GAM model
    # Linear terms for first 7 features, smooth terms for last 3
    gam = LinearGAM(
        terms=s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) +  # Linear terms
              s(7, n_splines=10) + s(8, n_splines=10) + s(9, n_splines=10)  # Smooth terms
    )
    
    print("\nFitting GAM model...")
    gam.fit(X_all, y)
    
    # Model summary
    print(f"\nGAM Model Summary:")
    print(f"Number of terms: {len(gam.terms)}")
    print(f"R-squared: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
    print(f"AIC: {gam.statistics_['AIC']:.2f}")
    
    # Create plots for smooth terms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot smooth effects
    smooth_features = ['GREEN', 'COORD1', 'COORD2']
    smooth_indices = [7, 8, 9]  # Indices in X_all
    
    for i, (feature_name, feature_idx) in enumerate(zip(smooth_features, smooth_indices)):
        # Generate prediction grid
        feature_values = X_all[:, feature_idx]
        XX = np.linspace(feature_values.min(), feature_values.max(), 100)
        
        # Create prediction matrix with other features at mean
        X_pred = np.tile(X_all.mean(axis=0), (100, 1))
        X_pred[:, feature_idx] = XX
        
        # Get partial dependence
        pdep = gam.partial_dependence(feature_idx, X_pred)
        conf_int = gam.partial_dependence(feature_idx, X_pred, width=0.95)
        
        # Plot
        axes[i].plot(XX, pdep, 'b-', linewidth=2, label='Smooth')
        axes[i].fill_between(XX, conf_int[:, 0], conf_int[:, 1], 
                           alpha=0.3, color='pink', label='95% CI')
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel(f's({feature_name})')
        axes[i].set_title(f'Smooth effect of {feature_name}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.5, 0.7)
        
        # Add rug plot
        axes[i].plot(feature_values, np.full_like(feature_values, -0.45), 
                    '|', alpha=0.4, markersize=1)
    
    plt.tight_layout()
    plt.show()
    
    return gam, data



#########################################################################################


# ====================================
#            chunk_bn18
# ====================================

def chunk_bn18(data, target_col='LPRIX', x_col='X', green_col='GREEN', 
               coord1_col='COORD1', coord2_col='COORD2', phi=20):
    """
    Fit GAM model and create 3D visualization of spatial coordinates effect.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    target_col : str
        Target variable column name
    x_col : str
        Linear predictor column name  
    green_col : str
        Green space column for smooth term
    coord1_col : str
        First coordinate column
    coord2_col : str
        Second coordinate column
    phi : float
        Elevation angle for 3D plot
    
    Returns:
    --------
    gam_model : GAM object
        Fitted GAM model
    """
    
    # Fit GAM model: LPRIX ~ X + s(GREEN) + s(COORD1, COORD2)
    gam_model = GAM(s(0) + s(1) + te(2, 3))
    X = data[[x_col, green_col, coord1_col, coord2_col]].values
    y = data[target_col].values
    
    gam_model.fit(X, y)
    
    print(f"GAM Model Summary - Deviance Explained: {gam_model.statistics_['pseudo_r2']:.3f}")
    
    # Create 3D visualization of coordinate effects
    coord1_range = np.linspace(data[coord1_col].min(), data[coord1_col].max(), 30)
    coord2_range = np.linspace(data[coord2_col].min(), data[coord2_col].max(), 30)
    coord1_grid, coord2_grid = np.meshgrid(coord1_range, coord2_range)
    
    # Generate predictions for visualization (using mean values for other variables)
    x_mean = data[x_col].mean()
    green_mean = data[green_col].mean()
    
    grid_points = np.column_stack([
        np.full(coord1_grid.size, x_mean),
        np.full(coord1_grid.size, green_mean),
        coord1_grid.ravel(),
        coord2_grid.ravel()
    ])
    
    predictions = gam_model.predict(grid_points).reshape(coord1_grid.shape)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(coord1_grid, coord2_grid, predictions, 
                          cmap='viridis', alpha=0.8)
    
    ax.set_xlabel(coord1_col)
    ax.set_ylabel(coord2_col) 
    ax.set_zlabel('Predicted ' + target_col)
    ax.set_title('$m_2(coord1) + m_3(coord2)$')
    ax.view_init(elev=phi, azim=45)
    
    plt.colorbar(surf, shrink=0.5)
    plt.tight_layout()
    plt.show()
    
    return gam_model




##############################################################################


# ====================================
#            chunk_bn19
# ====================================

def chunk_bn19(data, target_col='LPRIX', x_col='X', green_col='GREEN', 
               coord1_col='COORD1', coord2_col='COORD2', phi=20):
    """
    Fit GAM model with linear coordinate terms and create 3D visualization.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    target_col : str
        Target variable column name
    x_col : str
        Linear predictor column name  
    green_col : str
        Green space column for smooth term
    coord1_col : str
        First coordinate column (linear term)
    coord2_col : str
        Second coordinate column (linear term)
    phi : float
        Elevation angle for 3D plot
    
    Returns:
    --------
    gam_model : GAM object
        Fitted GAM model
    """
    
    # Fit GAM model: LPRIX ~ X + s(GREEN) + COORD1 + COORD2
    gam_model = GAM(l(0) + s(1) + l(2) + l(3))
    X = data[[x_col, green_col, coord1_col, coord2_col]].values
    y = data[target_col].values
    
    gam_model.fit(X, y)
    
    print(f"GAM Model Summary - Deviance Explained: {gam_model.statistics_['pseudo_r2']:.3f}")
    
    # Create 3D visualization of linear coordinate effects
    coord1_range = np.linspace(data[coord1_col].min(), data[coord1_col].max(), 30)
    coord2_range = np.linspace(data[coord2_col].min(), data[coord2_col].max(), 30)
    coord1_grid, coord2_grid = np.meshgrid(coord1_range, coord2_range)
    
    # Generate predictions for visualization (using mean values for other variables)
    x_mean = data[x_col].mean()
    green_mean = data[green_col].mean()
    
    grid_points = np.column_stack([
        np.full(coord1_grid.size, x_mean),
        np.full(coord1_grid.size, green_mean),
        coord1_grid.ravel(),
        coord2_grid.ravel()
    ])
    
    predictions = gam_model.predict(grid_points).reshape(coord1_grid.shape)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(coord1_grid, coord2_grid, predictions, 
                          cmap='viridis', alpha=0.8)
    
    ax.set_xlabel(coord1_col)
    ax.set_ylabel(coord2_col) 
    ax.set_zlabel('Predicted ' + target_col)
    ax.set_title(r'$\beta_1 \cdot coord1 + \beta_2 \cdot coord2$')
    ax.view_init(elev=phi, azim=45)
    
    plt.colorbar(surf, shrink=0.5)
    plt.tight_layout()
    plt.show()
    
    return gam_model



#########################################################################


# ====================================
#            chunk_bn20
# ====================================

def chunk_bn20(data, target_col='LPRIX', x_col='X', green_col='GREEN', 
               coord1_col='COORD1', coord2_col='COORD2'):
    """
    Fit linear GAM model (equivalent to linear regression) with all linear terms.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    target_col : str
        Target variable column name
    x_col : str
        Linear predictor column name  
    green_col : str
        Green space column (linear term)
    coord1_col : str
        First coordinate column (linear term)
    coord2_col : str
        Second coordinate column (linear term)
    
    Returns:
    --------
    gam_model : GAM object
        Fitted GAM model (all linear terms)
    """
    
    # Fit GAM model: LPRIX ~ X + GREEN + COORD1 + COORD2 (all linear)
    gam_model = GAM(l(0) + l(1) + l(2) + l(3))
    X = data[[x_col, green_col, coord1_col, coord2_col]].values
    y = data[target_col].values
    
    gam_model.fit(X, y)
    
    print(f"Linear GAM Model - R²: {gam_model.statistics_['pseudo_r2']:.3f}")
    
    return gam_model