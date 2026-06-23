import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
    

############################################################
# Chunk bvt1
############################################################
def chunk_bvt1():
    # Population
    population_x = [0, 3, 12]

    # Random samples from population with replacement
    np.random.seed(123)
    samples = np.random.choice(population_x, size=(2000, 3), replace=True)

    # Convert to DataFrame with column names
    samples_df = pd.DataFrame(samples, columns=["FirstX", "SecondX", "ThirdX"])

    # Display first few rows
    print(samples_df.head())



############################################################
# Chunk bvt2
############################################################
def chunk_bvt2():
    # Check if E(x_1)=E(x_2)=E(x_3)
    print("Column means:", np.round(samples.mean(axis=0), 2))

    # Check if Var(x_1)=Var(x_2)=Var(x_3)
    print("Column variances:", samples.var(axis=0, ddof=1))

    # Check correlation
    print("Correlation matrix:")
    print(samples.corr())

    # Using only unique set of samples
    uniqsam = samples.drop_duplicates()
    print("\nUnique samples - means:", uniqsam.mean(axis=0))
    print("Unique samples - variances:", uniqsam.var(axis=0, ddof=1))
    print("Unique samples - correlation:")
    print(uniqsam.corr())


############################################################
# Chunk bvt3
############################################################
def chunk_bvt3():
    # Check if E(x_1)=E(x_2)=E(x_3)
    print("Column means:", np.round(samples.mean(axis=0), 2))

    # Check if Var(x_1)=Var(x_2)=Var(x_3)
    print("Column variances:", samples.var(axis=0, ddof=1))

    # Check correlation
    print("Correlation matrix:")
    print(samples.corr())

    # Using only unique set of samples
    uniqsam = samples.drop_duplicates()
    print("\nUnique samples - means:", uniqsam.mean(axis=0))
    print("Unique samples - variances:", uniqsam.var(axis=0, ddof=1))
    print("Unique samples - correlation:")
    print(uniqsam.corr())

    # Xbar - row means
    X_bar = samples.mean(axis=1)
    print(f"\nMean of X_bar: {X_bar.mean()}")

    # Xhat - weighted combination
    X_hat = 0.5 * samples.iloc[:, 0] + 0.5 * samples.iloc[:, 2]
    print(f"Mean of X_hat: {X_hat.mean()}")

    # Xtilde - third column
    X_tilde = samples.iloc[:, 2]
    print(f"Mean of X_tilde: {X_tilde.mean()}")



############################################################
# Chunk bvt4
############################################################
def chunk_bvt4():
    # Calculate variances
    np.var(X_bar, ddof=1), np.var(X_hat, ddof=1), np.var(X_tilde, ddof=1)




############################################################
# Chunk bvt5
############################################################
def chunk_bvt5():
    # Population and sampling setup
    populationX = np.array([0, 3, 12])
    Ms = 2000

    # Generate all samples at once (with replacement)
    np.random.seed(123)
    samples = np.random.choice(populationX, size=(Ms, 3), replace=True)

    # Convert to DataFrame with column names
    samples = pd.DataFrame(samples, columns=["FirstX", "SecondX", "ThirdX"])
    samples.head()



############################################################
# Chunk bvt6
############################################################
def chunk_bvt6():
    # Generate predictions
    predictions = np.column_stack([
        np.full(Ms, 9),  # fhat_1 = 9
        samples.mean(axis=1)  # fhat_2 = mean of each row
    ])

    predictions[:5]  # Show first 5 rows



############################################################
# Chunk bvt7
############################################################
def chunk_bvt7():
    # MSPE calculation
    MSPE = np.column_stack([
        np.mean((populationX[:, None] - predictions[:, 0])**2, axis=0),
        np.mean((populationX[:, None] - predictions[:, 1])**2, axis=0)
    ])

    # Bias
    pop_mean = np.mean(populationX)
    bias1 = pop_mean - np.mean(predictions[:, 0])
    bias2 = pop_mean - np.mean(predictions[:, 1])

    # Variance (predictor)
    var1 = np.var(predictions[:, 0], ddof=1)
    var2 = np.var(predictions[:, 1], ddof=1)

    # Variance (epsilon)
    var_eps = np.mean((populationX - pop_mean)**2)

    MSPE[:5]  # Show first 5 rows



############################################################
# Chunk bvt8
############################################################
def chunk_bvt8():
    # Create variance-bias tradeoff table
    VBtradeoff = pd.DataFrame({
        'Bias': [bias1**2, bias2**2],
        'Var(fhat)': [var1, var2],
        'Var(eps)': [var_eps, var_eps],
        'MSPE': [np.mean(MSPE[:, 0]), np.mean(MSPE[:, 1])]
    }, index=['fhat_1', 'fhat_2'])

    VBtradeoff.round(3)



############################################################
# Chunk bvt9
############################################################
def chunk_bvt9():
    # Calculate alpha (magnitude of bias)
    alpha = pop_mean**2 / (pop_mean**2 + var_eps/3)

    # Biased predictor
    pred = alpha * predictions[:, 1]

    # Check if E(alpha*Xbar) = alpha*mu_x
    mean_pred = np.mean(pred)
    expected_mean = alpha * pop_mean

    # MSPE for biased predictor
    MSPE_biased = np.mean((populationX[:, None] - pred)**2, axis=0)
    mean_MSPE_biased = np.mean(MSPE_biased)

    print(f"Alpha: {alpha:.6f}")
    print(f"Mean of pred: {mean_pred:.6f}")
    print(f"Alpha * mu_x: {expected_mean:.6f}")
    print(f"Mean MSPE biased: {mean_MSPE_biased:.6f}")



############################################################
# Chunk bvt10
############################################################
def chunk_bvt10():
    # Updated variance-bias tradeoff table with biased predictor
    VBtradeoff = pd.DataFrame({
        'Bias': [bias1**2, bias2**2, (pop_mean - np.mean(pred))**2],
        'Var(fhat)': [var1, var2, np.var(pred, ddof=1)],
        'Var(eps)': [var_eps, var_eps, var_eps],
        'MSPE': [np.mean(MSPE[:, 0]), np.mean(MSPE[:, 1]), mean_MSPE_biased]
    }, index=['fhat_1', 'fhat_2', 'fhat_3'])

    VBtradeoff.round(3)



############################################################
# Chunk bvt11
############################################################
def chunk_bvt11():

    def xfunc(n, l):
        np.random.seed(123)
        x_1 = np.random.normal(0, 25, n)
        x_2 = l * x_1 + np.random.normal(0, 0.2, n)
        return pd.DataFrame({'x_1': x_1, 'x_2': x_2})

    def unseen(n, sigma, l):
        np.random.seed(1)
        x_11 = np.random.normal(0, 25, n)
        x_22 = l * x_11 + np.random.normal(0, 0.2, n)
        f = 0 + 2*x_11 + 2*x_22
        y_u = f + np.random.normal(0, sigma, n)
        return pd.DataFrame({'y': y_u, 'x_1': x_11, 'x_2': x_22})

    def sim(M, n, sigma, l):
        X = xfunc(n, l)
        un = unseen(n, sigma, l)
        
        MSPE_ols = np.zeros(M)
        MSPE_b = np.zeros(M)
        
        for i in range(M):
            f = 0 + 2*X['x_1'] + 2*X['x_2']
            y = f + np.random.normal(0, sigma, n)
            
            # Unbiased OLS (both features)
            ols = LinearRegression().fit(X, y)
            yhat = ols.predict(un[['x_1', 'x_2']])
            MSPE_ols[i] = np.mean((un['y'] - yhat)**2)
            
            # Biased OLS (only x_1)
            ols_b = LinearRegression().fit(X[['x_1']], y)
            yhat_b = ols_b.predict(un[['x_1']])
            MSPE_b[i] = np.mean((un['y'] - yhat_b)**2)
        
        return np.mean(MSPE_ols) - np.mean(MSPE_b)

    # Sensitivity analysis
    sigma_vals = np.arange(1, 21)
    MSPE_dif = [sim(1000, 100, sig, 0.01) for sig in sigma_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_vals, MSPE_dif, 'ro-', markersize=4)
    plt.xlabel('Sigma', fontsize=10)
    plt.ylabel('MSPE Difference', fontsize=10)
    plt.title('Difference in MSPE vs. sigma', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()



############################################################
# Chunk bvt12
############################################################
def chunk_bvt12():
    def xfunc(n, l):
        np.random.seed(123)
        x_1 = np.random.normal(0, 25, n)
        x_2 = l * x_1 + np.random.normal(0, 0.2, n)
        return pd.DataFrame({'x_1': x_1, 'x_2': x_2})

    # Check correlations for different l values
    for l in [0.001, 0.0011, 0.01]:
        X = xfunc(100, l)
        print(f"l = {l}:")
        print(X.corr().round(4))
        print()



############################################################
# Chunk bvt13
############################################################
def chunk_bvt13():


    def xfunc(n, l):
        np.random.seed(123)
        x_1 = np.random.normal(0, 25, n)
        x_2 = l * x_1 + np.random.normal(0, 0.2, n)
        return pd.DataFrame({'x_1': x_1, 'x_2': x_2})

    def unseen(n, sigma, l):
        np.random.seed(1)
        x_11 = np.random.normal(0, 25, n)
        x_22 = l * x_11 + np.random.normal(0, 0.2, n)
        f = 0 + 2*x_11 + 2*x_22
        y_u = f + np.random.normal(0, sigma, n)
        return pd.DataFrame({'y': y_u, 'x_1': x_11, 'x_2': x_22})

    def sim(M, n, sigma, l):
        X = xfunc(n, l)
        un = unseen(n, sigma, l)
        
        MSPE_ols = np.zeros(M)
        MSPE_b = np.zeros(M)
        
        for i in range(M):
            f = 0 + 2*X['x_1'] + 2*X['x_2']
            y = f + np.random.normal(0, sigma, n)
            
            # Unbiased OLS (both features)
            ols = LinearRegression().fit(X, y)
            yhat = ols.predict(un[['x_1', 'x_2']])
            MSPE_ols[i] = np.mean((un['y'] - yhat)**2)
            
            # Biased OLS (only x_1)
            ols_b = LinearRegression().fit(X[['x_1']], y)
            yhat_b = ols_b.predict(un[['x_1']])
            MSPE_b[i] = np.mean((un['y'] - yhat_b)**2)
        
        return np.mean(MSPE_ols) - np.mean(MSPE_b)

    # Sensitivity analysis for sigma
    sigma_vals = np.arange(1, 21)
    MSPE_dif_sigma = [sim(1000, 100, sig, 0.01) for sig in sigma_vals]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sigma_vals, MSPE_dif_sigma, 'ro-', markersize=4)
    plt.xlabel('Sigma', fontsize=10)
    plt.ylabel('MSPE Difference', fontsize=10)
    plt.title('Difference in MSPE vs. sigma', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Sensitivity analysis for correlation
    l_vals = np.arange(0.001, 0.011, 0.0001)
    MSPE_dif_corr = [sim(1000, 100, 7, l) for l in l_vals]

    plt.subplot(1, 2, 2)
    plt.plot(l_vals, MSPE_dif_corr, 'ro-', markersize=4)
    plt.xlabel('l (correlation parameter)', fontsize=10)
    plt.ylabel('MSPE Difference', fontsize=10)
    plt.title("Difference in MSPE vs Correlation b/w X's", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



############################################################
# Chunk bvt14
############################################################
def chunk_bvt14():

    # Generate population
    np.random.seed(123)
    popx = np.floor(np.random.normal(10, 2, 10000)).astype(int)
    print("Population summary:")
    print(pd.Series(popx).describe())

    # Generate samples
    np.random.seed(1)
    samples = np.random.choice(popx, size=(1000, 200), replace=True)
    print("\nFirst 10 columns of first 6 samples:")
    print(samples[:6, :10])

    # Calculate sample means and plot histogram
    sample_means = np.mean(samples, axis=1)
    plt.figure(figsize=(8, 6))
    plt.hist(sample_means, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Histogram of X_bar's", fontsize=12)
    plt.xlabel("X_bar", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nSample means summary:")
    print(pd.Series(sample_means).describe())
    print(f"\nPopulation mean: {np.mean(popx):.4f}")



############################################################
# Chunk bvt15
############################################################
def chunk_bvt15():

    # Our sample (201st sample, 0-indexed as 200)
    sample_0 = samples[200]
    sample_mean = np.mean(sample_0)

    # sd(F) and confidence interval
    sdF = np.sqrt(np.var(sample_0, ddof=1)) * (1 + 1/np.sqrt(len(sample_0)))
    ci = sample_mean + np.array([-1.96, 1.96]) * sdF

    print(f"Sample mean: {sample_mean:.6f}")
    print(f"Confidence interval: [{ci[0]:.6f}, {ci[1]:.6f}]")



############################################################
# Chunk bvt16
############################################################
def chunk_bvt16():

    # Generate in-sample data
    np.random.seed(123)
    x_1_in = np.random.normal(0, 1, 100)
    f_in = 1 - 2 * x_1_in
    y_in = f_in + np.random.normal(0, 1, 100)
    inn = pd.DataFrame({'y': y_in, 'x_1': x_1_in})

    # Generate out-of-sample data
    np.random.seed(321)
    x_1_out = np.random.normal(0, 10, 100)
    f_out = 1 - 2 * x_1_out
    y_out = f_out + np.random.normal(0, 1, 100)
    out = pd.DataFrame({'y': y_out, 'x_1': x_1_out})

    # OLS regression
    ols = LinearRegression().fit(inn[['x_1']], inn['y'])
    yhat = ols.predict(out[['x_1']])

    # Calculate prediction variance for each x_0
    n = len(inn)
    residuals = inn['y'] - ols.predict(inn[['x_1']])
    sigma2_hat = np.sum(residuals**2) / (n - 2)

    x_mean = np.mean(inn['x_1'])
    x_centered_sum = np.sum((inn['x_1'] - x_mean)**2)

    var_f = sigma2_hat * (1 + 1/n + (out['x_1'] - x_mean)**2 / x_centered_sum)
    sd_f = np.sqrt(var_f)

    # Create prediction intervals
    upper = yhat + 1.96 * sd_f
    lower = yhat - 1.96 * sd_f

    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(out['x_1'], yhat, yerr=1.96*sd_f, fmt='o', color='red', 
                ecolor='blue', capsize=3, markersize=4, alpha=0.7)
    plt.xlabel('x_0', fontsize=10)
    plt.ylabel('yhat ± 1.96sd', fontsize=10)
    plt.title('Prediction interval for each y_0', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()