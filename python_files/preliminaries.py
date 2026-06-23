#!/usr/bin/env python
# Python equivalents of R code chunks from 02-Preliminaries.Rmd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline

# Set plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

#############################################################################
# Chunk s1: Basic data operations and visualization with mtcars dataset
#############################################################################
def chunk_s1():
    """Basic data operations with mtcars dataset."""
    # Equivalent to mtcars dataset in R
    mtcars = pd.read_csv('https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv')
    
    # Equivalent to head(mtcars)
    print(mtcars.head())
    
    # Equivalent to str(mtcars)
    print(mtcars.info())
    
    # Equivalent to summary(mtcars)
    print(mtcars.describe())
    
    # Equivalent to plot(mtcars[,c(1, 3, 4)])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(mtcars.index, mtcars['mpg'], color='dodgerblue')
    axs[0].set_title('MPG')
    axs[1].scatter(mtcars.index, mtcars['disp'], color='dodgerblue')
    axs[1].set_title('Displacement')
    axs[2].scatter(mtcars.index, mtcars['hp'], color='dodgerblue')
    axs[2].set_title('Horsepower')
    plt.tight_layout()
    plt.show()

#############################################################################
# Chunk s2: Working with time-series data - airquality dataset
#############################################################################
def chunk_s2():
    """Working with the airquality time-series dataset."""
    # Equivalent to airquality dataset in R
    airquality = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_dataset_nyc.csv')
    # Renaming columns to match R dataset structure
    airquality = airquality.rename(columns={
        'Wind_Direction': 'Wind', 
        'Wind_Speed': 'Speed',
        'Maximum_Temperature': 'Temp',
        'Month': 'Month',
        'Day': 'Day'
    })
    # Add Ozone column (this is simulated as we don't have the exact data)
    np.random.seed(42)
    airquality['Ozone'] = np.random.normal(30, 15, len(airquality))
    airquality['Ozone'] = airquality['Ozone'].clip(lower=1)
    
    # Equivalent to head(airquality)
    print(airquality.head())
    
    # Equivalent to str(airquality)
    print(airquality.info())
    
    # Equivalent to summary(airquality)
    print(airquality.describe())

#############################################################################
# Chunk stat3: Basic time-series plotting
#############################################################################
def chunk_stat3():
    """Basic time-series plotting with airquality data."""
    # Using same airquality dataset from previous chunk
    airquality = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_dataset_nyc.csv')
    # Renaming and adding columns as in previous chunk
    airquality = airquality.rename(columns={
        'Wind_Direction': 'Wind', 
        'Wind_Speed': 'Speed',
        'Maximum_Temperature': 'Temp',
        'Month': 'Month',
        'Day': 'Day'
    })
    np.random.seed(42)
    airquality['Ozone'] = np.random.normal(30, 15, len(airquality))
    airquality['Ozone'] = airquality['Ozone'].clip(lower=1)
    
    # Equivalent to airquality$date <- airquality$Month*10+airquality$Day
    airquality['date'] = airquality['Month'] * 10 + airquality['Day']
    
    # Equivalent to plot(airquality$date, airquality$Ozone)
    plt.figure(figsize=(12, 6))
    plt.scatter(airquality['date'], airquality['Ozone'], color='dodgerblue')
    plt.xlabel('Date (Month*10 + Day)')
    plt.ylabel('Ozone')
    plt.title('Ozone Levels Over Time')
    plt.show()

#############################################################################
# Chunk s3: Working with dates and times
#############################################################################
def chunk_s3():
    """Working with dates and times in Python."""
    # Equivalent to Sys.Date() and Sys.time()
    print("Current date:", datetime.now().date())
    print("Current time:", datetime.now())
    
    # Equivalent to now() in lubridate
    print("Current date and time:", datetime.now())
    
    # Equivalent to dates vector and extracting components
    dates = pd.to_datetime(['2022-07-11', '2012-04-19', '2017-03-08'])
    print("\nDates:", dates)
    
    # Extract years (equivalent to year(dates))
    print("Years:", [d.year for d in dates])
    
    # Extract months (equivalent to month(dates))
    print("Months:", [d.month for d in dates])
    
    # Extract days (equivalent to mday(dates))
    print("Days:", [d.day for d in dates])

#############################################################################
# Chunk s6: Creating a histogram
#############################################################################
def chunk_s6():
    """Creating a histogram with the mpg data."""
    # Create a simulated dataset
    np.random.seed(42)
    n = 234
    mpg = pd.DataFrame({
        'cty': np.random.randint(9, 35, n),
    })
    
    # Equivalent to hist(mpg$cty)
    plt.figure(figsize=(10, 6))
    plt.hist(mpg['cty'], bins=12, color='dodgerblue', edgecolor='darkorange')
    plt.title('Histogram of MPG (City)', fontsize=14)
    plt.xlabel('Miles Per Gallon (City)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

#############################################################################
# Chunk s77: Creating a barplot
#############################################################################
def chunk_s77():
    """Creating a barplot with mpg drivetrain data."""
    # Create simulated dataset
    np.random.seed(42)
    n = 234
    mpg = pd.DataFrame({
        'drv': np.random.choice(['f', 'r', '4'], n, p=[0.5, 0.2, 0.3]),
    })
    
    # Equivalent to barplot(table(mpg$drv))
    counts = mpg['drv'].value_counts()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index, counts.values, color='dodgerblue', edgecolor='darkorange')
    plt.title('Drivetrains', fontsize=14)
    plt.xlabel('Drivetrain (f = FWD, r = RWD, 4 = 4WD)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

#############################################################################
# Chunk s8: Creating a boxplot
#############################################################################
def chunk_s8():
    """Creating a boxplot comparing highway MPG by drivetrain type."""
    # Create simulated dataset
    np.random.seed(42)
    n = 234
    drv_types = ['f', 'r', '4']
    mpg = pd.DataFrame({
        'drv': np.random.choice(drv_types, n, p=[0.5, 0.2, 0.3]),
        'hwy': np.random.randint(12, 44, n),
    })
    
    # Equivalent to boxplot(hwy ~ drv, data = mpg)
    plt.figure(figsize=(10, 6))
    drv_order = ['f', 'r', '4']  # Order to match R plot
    
    # Create the boxplot
    box = plt.boxplot([mpg[mpg['drv'] == d]['hwy'] for d in drv_order], 
                      labels=drv_order,
                      patch_artist=True)
    
    # Set colors
    for patch in box['boxes']:
        patch.set_facecolor('darkorange')
        patch.set_edgecolor('dodgerblue')
    
    for whisker in box['whiskers']:
        whisker.set_color('dodgerblue')
    
    for cap in box['caps']:
        cap.set_color('dodgerblue')
    
    for median in box['medians']:
        median.set_color('dodgerblue')
    
    for flier in box['fliers']:
        flier.set_markeredgecolor('dodgerblue')
    
    plt.title('MPG (Highway) vs Drivetrain', fontsize=14)
    plt.xlabel('Drivetrain (f = FWD, r = RWD, 4 = 4WD)', fontsize=12)
    plt.ylabel('Miles Per Gallon (Highway)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

#############################################################################
# Chunk s9: Creating a scatterplot
#############################################################################
def chunk_s9():
    """Creating a scatterplot of Highway MPG vs Engine Displacement."""
    # Create simulated dataset
    np.random.seed(42)
    n = 234
    mpg = pd.DataFrame({
        'displ': np.random.uniform(1.6, 7.0, n),
        'hwy': np.random.randint(12, 44, n),
    })
    
    # Equivalent to plot(hwy ~ displ, data = mpg)
    plt.figure(figsize=(10, 6))
    plt.scatter(mpg['displ'], mpg['hwy'], color='dodgerblue', s=50)
    plt.title('MPG (Highway) vs Engine Displacement', fontsize=14)
    plt.xlabel('Engine Displacement (in Liters)', fontsize=12)
    plt.ylabel('Miles Per Gallon (Highway)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

#############################################################################
# Chunk s10: Probability density function (pdf) calculation
#############################################################################
def chunk_s10():
    """Calculate normal PDF at a specific point."""
    # Equivalent to dnorm(x = 4, mean = 2, sd = 5)
    pdf_value = stats.norm.pdf(4, loc=2, scale=5)
    print(f"PDF value at x=4 for N(2,25): {pdf_value}")

#############################################################################
# Chunk s11: Cumulative distribution function (cdf) calculation
#############################################################################
def chunk_s11():
    """Calculate normal CDF at a specific point."""
    # Equivalent to pnorm(q = 4, mean = 2, sd = 5)
    cdf_value = stats.norm.cdf(4, loc=2, scale=5)
    print(f"CDF value at x=4 for N(2,25): {cdf_value}")

#############################################################################
# Chunk s12: Quantile calculation
#############################################################################
def chunk_s12():
    """Calculate normal quantile at a specific probability."""
    # Equivalent to qnorm(p = 0.975, mean = 2, sd = 5)
    quantile_value = stats.norm.ppf(0.975, loc=2, scale=5)
    print(f"0.975 quantile for N(2,25): {quantile_value}")

#############################################################################
# Chunk s13: Random number generation
#############################################################################
def chunk_s13():
    """Generate random samples from a normal distribution."""
    # Equivalent to rnorm(n = 10, mean = 2, sd = 5)
    np.random.seed(42)  # For reproducibility
    random_sample = np.random.normal(2, 5, size=10)
    print("Random sample from N(2,25):")
    print(random_sample)

#############################################################################
# Chunk s16: Linear regression and plotting
#############################################################################
def chunk_s16():
    """Linear regression with speed predicting stopping distance."""
    # Create simulated data similar to cars dataset
    np.random.seed(42)
    speed = np.linspace(4, 25, 50)
    dist = 3.9 * speed - 17.6 + np.random.normal(0, 10, 50)
    cars = pd.DataFrame({'speed': speed, 'dist': dist})
    
    # Equivalent to model <- lm(dist ~ speed, data = cars)
    X = cars[['speed']].values
    y = cars['dist'].values
    model = LinearRegression().fit(X, y)
    
    # Coefficients
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # Plot the data and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(cars['speed'], cars['dist'], color='blue', s=80, label='Data')
    
    # Generate points for the line
    x_line = np.array([min(cars['speed']), max(cars['speed'])])
    y_line = intercept + slope * x_line
    
    # Plot the regression line
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Regression Line')
    
    plt.title('Linear Regression: Stopping Distance vs Speed', fontsize=14)
    plt.xlabel('Speed (in Miles Per Hour)', fontsize=12)
    plt.ylabel('Stopping Distance (in Feet)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"Regression equation: dist = {intercept:.2f} + {slope:.2f} * speed")

#############################################################################
# Chunk s17: Calculating regression coefficients manually
#############################################################################
def chunk_s17():
    """Manual calculation of regression coefficients."""
    # Create simulated data similar to cars dataset
    np.random.seed(42)
    speed = np.linspace(4, 25, 50)
    dist = 3.9 * speed - 17.6 + np.random.normal(0, 10, 50)
    
    # Extract speed and distance
    x = speed
    y = dist
    
    # Calculate covariance and variance
    Sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    Sxx = np.sum((x - np.mean(x)) ** 2)
    Syy = np.sum((y - np.mean(y)) ** 2)
    
    # Calculate regression coefficients
    beta_1 = Sxy / Sxx
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    
    print(f"Manually calculated coefficients: beta_0 = {beta_0:.2f}, beta_1 = {beta_1:.2f}")

#############################################################################
# Chunk s21: Multiple linear regression model
#############################################################################
def chunk_s21():
    """Multiple linear regression with weight and horsepower predicting MPG."""
    # Create simulated data similar to mtcars
    np.random.seed(42)
    n = 32  # mtcars has 32 observations
    wt = np.random.uniform(1.5, 5.5, n)
    hp = np.random.uniform(50, 350, n)
    mpg = 37.3 - 3.9 * wt - 0.03 * hp + np.random.normal(0, 2, n)
    
    # Create a DataFrame
    mtcars = pd.DataFrame({'wt': wt, 'hp': hp, 'mpg': mpg})
    
    # Equivalent to mpg_model = lm(mpg ~ wt + hp, data = mtcars)
    X = mtcars[['wt', 'hp']].values
    y = mtcars['mpg'].values
    
    model = LinearRegression().fit(X, y)
    
    print("Multiple regression coefficients:")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Weight (wt): {model.coef_[0]:.4f}")
    print(f"Horsepower (hp): {model.coef_[1]:.4f}")

#############################################################################
# Chunk s27: Maximum likelihood estimation using optimization
#############################################################################
def chunk_s27():
    """Maximum likelihood estimation using optimization."""
    # Generate random normal data
    np.random.seed(123)
    x = np.random.normal(0, 1, 100)
    
    # Define the negative log-likelihood function to minimize
    def neg_log_likelihood(params):
        mu, sigma = params
        return np.sum(0.5 * ((x - mu) ** 2) / sigma + 0.5 * np.log(sigma))
    
    # Using scipy's optimize.minimize (equivalent to R's optim)
    result = minimize(neg_log_likelihood, [0, 1], method='Nelder-Mead')
    
    print("Optimization results:")
    print(f"Estimated mean: {result.x[0]}")
    print(f"Estimated variance: {result.x[1]}")
    print(f"Convergence status: {result.success}")
    print(f"Function value at minimum: {result.fun}")

#############################################################################
# Chunk s29: Simulation of sampling distribution
#############################################################################
def chunk_s29():
    """Simulation of sampling distribution."""
    # Population parameters
    pop_mean = 5
    pop_sd = 1
    
    # Generate a large population
    np.random.seed(123)
    pop_x = np.random.normal(pop_mean, pop_sd, 1000000)
    
    # Sampling parameters
    n = 200       # sample size
    mcn = 1000    # number of samples
    
    # Initialize an array to store sample means
    xbars = np.zeros(mcn)
    
    # Generate samples and calculate means
    for i in range(mcn):
        # Sample with replacement
        sample = np.random.choice(pop_x, size=n, replace=True)
        xbars[i] = np.mean(sample)
    
    # Calculate the mean of sample means
    mxbar = np.mean(xbars)
    print(f"Mean of sample means: {mxbar:.2f}")
    
    # Plot histogram of sample means
    plt.figure(figsize=(10, 6))
    plt.hist(xbars, bins=20, color='dodgerblue', edgecolor='black')
    plt.title('Sampling Distribution of Sample Mean', fontsize=14)
    plt.xlabel('Sample Mean', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

#############################################################################
# Chunk s39: Linear vs. Nonparametric methods
#############################################################################
def chunk_s39():
    """Compare linear vs. nonparametric regression methods."""
    # Create simulated data
    np.random.seed(42)
    x = np.linspace(4, 25, 50)
    y = 3.9 * x - 17.6 + np.random.normal(0, 10, 50)
    
    # Sort for plotting
    indices = np.argsort(x)
    x_sorted = x[indices]
    y_sorted = y[indices]
    
    # Create smooth x values for plotting
    xt = np.linspace(min(x), max(x), len(x))
    
    # Linear model
    model_linear = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred_linear = model_linear.predict(xt.reshape(-1, 1))
    
    # Nonparametric model (LOWESS)
    lowess_result = lowess(y, x, frac=0.2, it=2)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='grey', label='Data')
    plt.plot(xt, y_pred_linear, color='red', linewidth=2, label='Parametric (Linear)')
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], color='blue', linewidth=2, label='Nonparametric (LOWESS)')
    
    plt.title('Parametric (red) vs Nonparametric (blue)', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

#############################################################################
# Chunk s41: Overfitting example
#############################################################################
def chunk_s41():
    """Demonstrate overfitting with complex vs. simple models."""
    # Create simulated data
    np.random.seed(42)
    x = np.linspace(4, 25, 50)
#############################################################################
# Chunk s18: Using scikit-learn for linear regression
#############################################################################
def chunk_s18():
    """Linear regression using scikit-learn."""
    # Use the cars dataset
    cars = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/cars.csv')
    cars = cars.iloc[:, 1:]  # Remove index column
    
    # Equivalent to model <- lm(dist ~ speed, data = cars)
    X = cars[['speed']].values
    y = cars['dist'].values
    model = LinearRegression().fit(X, y)
    
    print(f"Scikit-learn coefficients: intercept = {model.intercept_:.2f}, slope = {model.coef_[0]:.2f}")

#############################################################################
# Chunk s19: Linear regression without intercept
#############################################################################
def chunk_s19():
    """Linear regression without intercept term."""
    # Use the cars dataset
    cars = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/cars.csv')
    cars = cars.iloc[:, 1:]  # Remove index column
    
    x = cars['speed'].values
    y = cars['dist'].values
    
    # Calculate beta_1 manually (no intercept)
    beta_1 = np.sum(x * y) / np.sum(x ** 2)
    print(f"Manually calculated coefficient (no intercept): beta_1 = {beta_1:.2f}")
    
    # Using scikit-learn
    model = LinearRegression(fit_intercept=False).fit(cars[['speed']], cars['dist'])
    print(f"Scikit-learn coefficient (no intercept): {model.coef_[0]:.2f}")

#############################################################################
# Chunk s20: Multiple linear regression with mtcars
#############################################################################
def chunk_s20():
    """Multiple linear regression with mtcars dataset."""
    # Use the mtcars dataset
    mtcars = pd.read_csv('https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv')
    
    # Display the structure
    print("First few rows of mtcars:")
    print(mtcars.head())
    
    print("\nStructure of mtcars:")
    print(mtcars.info())

#############################################################################
# Chunk s21: Multiple linear regression model
#############################################################################
def chunk_s21():
    """Multiple linear regression with weight and horsepower predicting MPG."""
    # Use the mtcars dataset
    mtcars = pd.read_csv('https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv')
    
    # Equivalent to mpg_model = lm(mpg ~ wt + hp, data = mtcars)
    X = mtcars[['wt', 'hp']].values
    y = mtcars['mpg'].values
    
    model = LinearRegression().fit(X, y)
    
    print("Multiple regression coefficients:")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Weight (wt): {model.coef_[0]:.4f}")
    print(f"Horsepower (hp): {model.coef_[1]:.4f}")

#############################################################################
# Chunk s22: Computing probability density function values
#############################################################################
def chunk_s22():
    """Compute PDF values for normal distribution."""
    # Equivalent to dnorm(0, mean=1, sd=2)
    pdf_value1 = stats.norm.pdf(0, loc=1, scale=2)
    print(f"PDF at x=0 for N(1,4): {pdf_value1}")
    
    # Equivalent to dnorm(1, mean=1, sd=2)
    pdf_value2 = stats.norm.pdf(1, loc=1, scale=2)
    print(f"PDF at x=1 for N(1,4): {pdf_value2}")

#############################################################################
# Chunk s23: Computing and plotting pdf values for a range
#############################################################################
def chunk_s23():
    """Compute and plot PDF values for a range of x values."""
    # Equivalent to seq(from = -10, to = +22, length.out = 100)
    x = np.linspace(-10, 22, 100)
    
    # Equivalent to pdfs <- dnorm(x, mean = 1, sd = 2)
    pdfs = stats.norm.pdf(x, loc=1, scale=2)
    
    # Equivalent to plot(x, pdfs)
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdfs)
    plt.title('Normal PDF with μ=1, σ=2', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

#############################################################################
# Chunk s24: Generate random normal data
#############################################################################
def chunk_s24():
    """Generate random samples from normal distribution."""
    # Equivalent to x <- rnorm(1000, 2, 7)
    np.random.seed(123)
    x = np.random.normal(2, 7, 1000)
    print("Generated 1000 random values from N(2,49)")
    print("First 5 values:", x[:5])

#############################################################################
# Chunk s25: Plotting different normal distributions
#############################################################################
def chunk_s25():
    """Plot different normal distributions for comparison."""
    # Generate random normal data
    np.random.seed(123)
    x = np.random.normal(2, 7, 1000)
    
    # Calculate pdf values for different parameters
    pdfs1 = stats.norm.pdf(x, loc=1, scale=2)
    pdfs2 = stats.norm.pdf(x, loc=5, scale=7)
    pdfs3 = stats.norm.pdf(x, loc=2, scale=7)
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each distribution
    axs[0].scatter(x, pdfs1, s=2, alpha=0.5)
    axs[0].set_title('Normal PDF with μ=1, σ=2')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].scatter(x, pdfs2, s=2, alpha=0.5)
    axs[1].set_title('Normal PDF with μ=5, σ=7')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].scatter(x, pdfs3, s=2, alpha=0.5)
    axs[2].set_title('Normal PDF with μ=2, σ=7')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#############################################################################
# Chunk s26: Calculating the likelihood of a sample
#############################################################################
def chunk_s26():
    """Calculate likelihood of a sample."""
    # Equivalent to set.seed(2019); x <- rnorm(100)
    np.random.seed(2019)
    x = np.random.normal(0, 1, 100)
    
    # Equivalent to prod(dnorm(x))
    # In Python, we use log probabilities to avoid numerical underflow
    log_likelihoods = stats.norm.logpdf(x, loc=0, scale=1)
    log_likelihood = np.sum(log_likelihoods)
    likelihood = np.exp(log_likelihood)
    
    print(f"Log-likelihood: {log_likelihood}")
    print(f"Likelihood (may be close to zero due to numerical precision): {likelihood}")

#############################################################################
# Chunk s27: Maximum likelihood estimation using optimization
#############################################################################
def chunk_s27():
    """Maximum likelihood estimation using optimization."""
    # Generate random normal data
    np.random.seed(123)
    x = np.random.normal(0, 1, 100)
    
    # Define the negative log-likelihood function to minimize
    def neg_log_likelihood(params):
        mu, sigma = params
        return np.sum(0.5 * ((x - mu) ** 2) / sigma + 0.5 * np.log(sigma))
    
    # Using scipy's optimize.minimize (equivalent to R's optim)
    result = minimize(neg_log_likelihood, [0, 1], method='Nelder-Mead')
    
    print("Optimization results:")
    print(f"Estimated mean: {result.x[0]}")
    print(f"Estimated variance: {result.x[1]}")
    print(f"Convergence status: {result.success}")
    print(f"Function value at minimum: {
