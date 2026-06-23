import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ================================================
#                   chunk_p1
# ================================================

def chunk_p1(vehicles_data=None):
    """
    Convert vehicles data to binary MPG outcome based on highway MPG.
    
    Args:
        vehicles_data: DataFrame with vehicle data. If None, assumes you have your own data source.
    
    Returns:
        DataFrame with added binary 'mpg' column
    """
    if vehicles_data is None:
        raise ValueError("Please provide vehicles_data DataFrame")
    
    df = vehicles_data.copy()
    
    # Remove NAs
    original_shape = df.shape
    data = df.dropna()
    print(f"Shape after removing NAs: {original_shape} -> {data.shape}")
    
    # Binary outcome mpg = 1 if hwy > mean(hwy), 0 otherwise
    hwy_mean = data['hwy'].mean()
    data['mpg'] = (data['hwy'] > hwy_mean).astype(int)
    
    print(f"Binary MPG distribution: {data['mpg'].value_counts().to_dict()}")
    
    return data



############################################################

# ================================================
#                   chunk_p2
# ================================================

def chunk_p2(data):
    """
    Convert string columns to categorical dtype and display data info.
    
    Args:
        data: DataFrame to process
        
    Returns:
        DataFrame with string columns converted to categorical
    """
    df = data.copy()
    
    # Convert string columns to categorical
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    return df


###############################################################

# ================================================
#                   chunk_p3
# ================================================

def chunk_p3(data):
    """
    Fit linear regression model: mpg ~ fuel (no intercept).
    
    Args:
        data: DataFrame with 'mpg' and 'fuel' columns
        
    Returns:
        Fitted statsmodels regression results
    """
    df = data.copy()
    
    # Create dummy variables for categorical fuel column (no intercept)
    X = pd.get_dummies(df['fuel'], drop_first=False)
    y = df['mpg']
    
    # Fit model without intercept using statsmodels for R-like summary
    model = sm.OLS(y, X).fit()
    
    print("Linear Regression Summary (mpg ~ fuel, no intercept):")
    print(model.summary())
    
    return model



##############################################################################


# ================================================
#                   chunk_p4
# ================================================

def chunk_p4(data):
    """
    Create cross-tabulation of fuel type vs mpg with margins and proportions.
    
    Args:
        data: DataFrame with 'fuel' and 'mpg' columns
        
    Returns:
        tuple: (crosstab_with_margins, row_proportions)
    """
    df = data.copy()
    
    # Create cross-tabulation
    tab = pd.crosstab(df['fuel'], df['mpg'], margins=True)
    
    # Row proportions (equivalent to prop.table(tab, 1))
    row_props = pd.crosstab(df['fuel'], df['mpg'], normalize='index')
    
    print("Cross-tabulation with margins:")
    print(tab)
    print("\nRow proportions:")
    print(row_props.round(4))
    
    return tab, row_props



#####################################################################


# ================================================
#                   chunk_p5
# ================================================

def chunk_p5(data):
    """
    Fit linear regression model: mpg ~ fuel + drive + cyl (with intercept).
    
    Args:
        data: DataFrame with 'mpg', 'fuel', 'drive', and 'cyl' columns
        
    Returns:
        Fitted statsmodels regression results
    """
    df = data.copy()
    
    # Create design matrix with categorical variables
    X = pd.get_dummies(df[['fuel', 'drive']], drop_first=True)
    X['cyl'] = df['cyl']
    X = sm.add_constant(X)  # Add intercept
    y = df['mpg']
    
    # Fit model with intercept using statsmodels
    model = sm.OLS(y, X).fit()
    
    print("Linear Regression Summary (mpg ~ fuel + drive + cyl):")
    print(model.summary())
    
    return model



################################################################


# ================================================
#                   chunk_p6
# ================================================

def chunk_p6(data, model2=None):
    """
    Fit GLM with gaussian family and compare coefficients with previous linear model.
    
    Args:
        data: DataFrame with 'mpg', 'fuel', 'drive', and 'cyl' columns
        model2: Previous linear model for comparison (optional)
        
    Returns:
        tuple: (glm_model, coefficients_identical)
    """
    df = data.copy()
    
    # Create design matrix (same as model2)
    X = pd.get_dummies(df[['fuel', 'drive']], drop_first=True)
    X['cyl'] = df['cyl']
    X = sm.add_constant(X)
    y = df['mpg']
    
    # Fit GLM with gaussian family
    model3 = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
    
    # Compare coefficients if model2 provided
    coeffs_identical = False
    if model2 is not None:
        coef2_rounded = np.round(model2.params.values, 2)
        coef3_rounded = np.round(model3.params.values, 2)
        coeffs_identical = np.array_equal(coef2_rounded, coef3_rounded)
        print(f"Coefficients identical (rounded to 2 decimals): {coeffs_identical}")
    
    return model3, coeffs_identical


#######################################################


# ================================================
#                   chunk_p7
# ================================================

def chunk_p7(data, model2):
    """
    Analyze binary MPG distribution and in-sample predictions.
    
    Args:
        data: DataFrame with 'mpg' column
        model2: Fitted linear regression model
        
    Returns:
        tuple: (mpg_counts, fitted_values, predictions_above_05, predictions_below_05)
    """
    df = data.copy()
    
    # Count binary MPG outcomes
    mpg_counts = df['mpg'].value_counts().sort_index()
    print(f"MPG distribution: {mpg_counts.to_dict()}")
    
    # Get in-sample fitted values (predicted probabilities)
    mpg_hat = model2.fittedvalues
    
    # Count predictions above and below 0.5 threshold
    above_05 = (mpg_hat > 0.5).sum()
    below_05 = (mpg_hat <= 0.5).sum()
    
    print(f"Predictions > 0.5: {above_05}")
    print(f"Predictions <= 0.5: {below_05}")
    
    return mpg_counts, mpg_hat, above_05, below_05


#################################################################


# ================================================
#                   chunk_p8
# ================================================

def chunk_p8(mpg_hat):
    """
    Display summary statistics for fitted values (predicted probabilities).
    
    Args:
        mpg_hat: Array or Series of fitted values from regression model
        
    Returns:
        pandas Series with summary statistics
    """
    # Convert to pandas Series if numpy array
    if isinstance(mpg_hat, np.ndarray):
        mpg_hat = pd.Series(mpg_hat)
    
    # Get summary statistics
    summary_stats = mpg_hat.describe()
    
    print("Summary of fitted values (mpg_hat):")
    print(summary_stats)
    
    return summary_stats



################################################################


# ================================================
#                   chunk_p9
# ================================================

def chunk_p9(data):
    """
    Fit simple linear regression mpg ~ cyl and create scatter plot with fitted line.
    
    Args:
        data: DataFrame with 'mpg' and 'cyl' columns
        
    Returns:
        tuple: (model, matplotlib figure)
    """
    df = data.copy()
    
    # Fit simple linear regression model
    X = sm.add_constant(df['cyl'])
    y = df['mpg']
    model_n = sm.OLS(y, X).fit()
    
    # Create scatter plot with fitted line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['cyl'], df['mpg'], alpha=0.6)
    ax.plot(df['cyl'], model_n.fittedvalues, color='red', linewidth=2)
    ax.set_xlabel('Cylinders')
    ax.set_ylabel('MPG (Binary)')
    ax.set_ylim(-1.2, 1.2)
    ax.set_title('Linear Regression: MPG ~ Cylinders')
    plt.tight_layout()
    plt.show()
    
    return model_n, fig



########################################################


# ================================================
#                   chunk_p10
# ================================================

def chunk_p10():
    """
    Generate sigmoid function plot using random normal data.
    
    Returns:
        tuple: (x_values, sigma_values, matplotlib figure)
    """
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Generate data
    n = 500
    x = np.random.normal(0, 2, n)
    sigma = 1 / (1 + np.exp(-x))
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, sigma, color='blue', alpha=0.6, s=20)
    ax.set_xlabel('x')
    ax.set_ylabel('sigma')
    ax.set_title('Sigmoid Function: σ = 1/(1 + exp(-x))')
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.show()
    
    return x, sigma, fig


##############################################################


# ================================================
#                   chunk_p11
# ================================================

def chunk_p11(x, sigma):
    """
    Calculate logit transformation and create scatter plot.
    
    Args:
        x: Input values
        sigma: Sigma values from sigmoid function
        
    Returns:
        tuple: (logit_values, matplotlib figure)
    """
    # Calculate logit transformation
    p_x = sigma
    logit = np.log(p_x / (1 - p_x))  # Natural logarithm by default
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, logit, color='red', alpha=0.6, s=20)
    ax.set_xlabel('x')
    ax.set_ylabel('Logit')
    ax.set_title('Logit Transformation: log(p/(1-p))')
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.show()
    
    return logit, fig



#######################################################################


# ================================================
#                   chunk_p12
# ================================================

def chunk_p12():
    """
    Generate synthetic logistic regression data with Bernoulli outcomes.
    
    Returns:
        DataFrame with binary outcome y and predictor x
    """
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Generate random data
    n = 500
    x = np.random.normal(0, 1, n)
    z = -2 + 3 * x
    
    # Probability defined by logistic function
    p = 1 / (1 + np.exp(-z))
    
    # Bernoulli distribution for binary outcomes
    y = np.random.binomial(1, p, n)
    
    # Create DataFrame
    data = pd.DataFrame({'y': y, 'x': x})
    
    print("First 6 rows:")
    print(data.head(6))
    print(f"\nBinary outcome distribution: {data['y'].value_counts().sort_index().to_dict()}")
    
    return data




###########################################################################

# ================================================
#                   chunk_p13
# ================================================

def chunk_p13(data, x, p):
    """
    Fit Linear Probability Model and compare with true probabilities.
    
    Args:
        data: DataFrame with 'y' and 'x' columns
        x: Original x values
        p: True probabilities from logistic function
        
    Returns:
        tuple: (lpm_model, matplotlib figure)
    """
    # Fit Linear Probability Model
    X = sm.add_constant(data['x'])
    lpm = sm.OLS(data['y'], X).fit()
    
    print("Linear Probability Model Summary:")
    print(lpm.summary())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true probabilities
    ax.scatter(x, p, color='green', alpha=0.6, s=20, label='Probability')
    
    # Plot LPM fitted line
    x_sorted = np.sort(data['x'])
    lpm_pred = lpm.params[0] + lpm.params[1] * x_sorted
    ax.plot(x_sorted, lmp_pred, color='red', linewidth=2, label='Estimated Probability by LPM')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper left', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    plt.show()
    
    return lpm, fig


#################################################################

# ================================================
#                   chunk_p14
# ================================================

def chunk_p14(data, x, p):
    """
    Fit logistic regression and compare with true probabilities.
    
    Args:
        data: DataFrame with 'y' and 'x' columns
        x: Original x values
        p: True probabilities from logistic function
        
    Returns:
        tuple: (logistic_model, matplotlib figure)
    """
    # Fit logistic regression (GLM with binomial family)
    X = sm.add_constant(data['x'])
    logis = sm.GLM(data['y'], X, family=sm.families.Binomial()).fit()
    
    print("Logistic Regression Summary:")
    print(logis.summary())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true probabilities
    ax.scatter(x, p, color='green', alpha=0.6, s=20, label='Probability')
    
    # Plot logistic regression predictions
    x_range = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_range)
    logis_pred = logis.predict(X_pred)
    ax.plot(x_range, logis_pred, color='red', linestyle='--', linewidth=2, 
            label='Estimated Probability by GLM')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper left', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    plt.show()
    
    return logis, fig