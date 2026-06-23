import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.datasets import fetch_openml
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# ===================================
#           chunk_en1
# ===================================

def chunk_en1(titanic_data, B=9):
    """
    Convert R script en1 to Python: bootstrap sampling with decision trees on Titanic data
    
    Args:
        titanic_data: DataFrame with columns 'survived', 'sex', 'age', 'pclass', 'sibsp', 'parch'
        B: number of bootstrap samples (default 9)
    
    Returns:
        list: trained decision tree models from bootstrap samples
    """
    # Color palette for trees
    colors = ['pink', 'red', 'blue', 'yellow', 'darkgreen', 
              'orange', 'brown', 'purple', 'darkblue']
    
    n = len(titanic_data)
    trees = []
    
    # Set up subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i in range(B):
        # Bootstrap sampling with replacement
        np.random.seed(i*2)
        idx = np.random.choice(n, n, replace=True)
        tr = titanic_data.iloc[idx].copy()
        
        # Prepare features and target
        features = ['sex', 'age', 'pclass', 'sibsp', 'parch']
        X = pd.get_dummies(tr[features], drop_first=True)  # Handle categorical variables
        y = tr['survived']
        
        # Build unpruned classification tree (cp=0 equivalent)
        cart = DecisionTreeClassifier(
            random_state=42, 
            min_samples_split=2, 
            min_samples_leaf=1,
            max_depth=None
        )
        cart.fit(X, y)
        trees.append(cart)
        
        # Plot tree with color
        plot_tree(cart, 
                 feature_names=X.columns,
                 class_names=['Not Survived', 'Survived'],
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 ax=axes[i])
        
        axes[i].set_title(f'Bootstrap Tree {i+1}', color=colors[i], fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Bootstrap Decision Trees - Titanic Survival', fontsize=16, y=1.02)
    plt.show()
    
    return trees





####################################################################################################################




# ===================================
#           chunk_en2
# ===================================

def chunk_en2(titanic_data, random_state=1):
    """
    Convert R script en2 to Python: single decision tree with AUC evaluation on Titanic data
    
    Args:
        titanic_data: DataFrame with columns 'survived', 'sex', 'age', 'pclass', 'sibsp', 'parch'
        random_state: seed for reproducible train/test split
    
    Returns:
        dict: AUC score and trained model
    """
    # Test/train split (80% train, 20% test)
    features = ['sex', 'age', 'pclass', 'sibsp', 'parch']
    X = pd.get_dummies(titanic_data[features], drop_first=True)
    y = titanic_data['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=random_state
    )
    
    # Single pruned tree (default parameters provide pruning)
    cart = DecisionTreeClassifier(random_state=42)
    cart.fit(X_train, y_train)
    
    # Predict probabilities (equivalent to type="prob" in R)
    phat1 = cart.predict_proba(X_test)
    
    # Calculate AUC using probability of positive class (survived=1)
    auc_score = roc_auc_score(y_test, phat1[:, 1])
    
    print(f"AUC: {auc_score:.4f}")
    
    return {'auc': auc_score, 'model': cart}





####################################################################################################################



# ===================================
#           chunk_en3
# ===================================

def chunk_en3(train_data, test_data, B=100):
    """
    Convert R script en3 to Python: bootstrap ensemble of unpruned trees
    
    Args:
        train_data: training DataFrame with survival prediction features
        test_data: test DataFrame for predictions
        B: number of bootstrap trees (default 100)
    
    Returns:
        numpy.ndarray: matrix of predictions (B x n_test_samples)
    """
    features = ['sex', 'age', 'pclass', 'sibsp', 'parch']
    
    # Prepare test data
    X_test = pd.get_dummies(test_data[features], drop_first=True)
    n_test = len(test_data)
    
    # Initialize prediction matrix
    phat2 = np.zeros((B, n_test))
    
    # Bootstrap loop
    for i in range(B):
        np.random.seed(i)
        
        # Bootstrap sample from training data
        idx = np.random.choice(len(train_data), len(train_data), replace=True)
        dt = train_data.iloc[idx].copy()
        
        # Prepare bootstrap training features
        X_train_boot = pd.get_dummies(dt[features], drop_first=True)
        y_train_boot = dt['survived']
        
        # Align test features with bootstrap training features
        X_test_aligned = X_test.reindex(columns=X_train_boot.columns, fill_value=0)
        
        # Build unpruned tree (cp=0 equivalent)
        cart_B = DecisionTreeClassifier(
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None
        )
        cart_B.fit(X_train_boot, y_train_boot)
        
        # Store probabilities for positive class (survived=1)
        phat2[i, :] = cart_B.predict_proba(X_test_aligned)[:, 1]
    
    print(f"Prediction matrix shape: {phat2.shape}")
    
    return phat2




####################################################################################################################





# ===================================
#           chunk_en4
# ===================================

def chunk_en4(phat2, test_labels):
    """
    Convert R script en4 to Python: ensemble averaging and AUC evaluation
    
    Args:
        phat2: prediction matrix from bootstrap ensemble (B x n_test_samples)
        test_labels: true labels for test set
    
    Returns:
        dict: ensemble AUC score and averaged predictions
    """
    # Take the average across all bootstrap predictions
    phat_f = np.mean(phat2, axis=0)
    
    # Calculate AUC using ensemble averaged predictions
    auc_ensemble = roc_auc_score(test_labels, phat_f)
    
    print(f"Ensemble AUC: {auc_ensemble:.4f}")
    
    return {'auc': auc_ensemble, 'predictions': phat_f}





####################################################################################################################





# ===================================
#           chunk_en5
# ===================================

def chunk_en5(train_data, test_data, B=300):
    """
    Convert R script en5 to Python: progressive ensemble AUC evaluation
    
    Args:
        train_data: training DataFrame with survival prediction features
        test_data: test DataFrame for predictions
        B: number of bootstrap trees (default 300)
    
    Returns:
        dict: AUC progression array and final prediction matrix
    """
    features = ['sex', 'age', 'pclass', 'sibsp', 'parch']
    
    # Prepare test data
    X_test = pd.get_dummies(test_data[features], drop_first=True)
    y_test = test_data['survived']
    n_test = len(test_data)
    
    # Initialize arrays
    phat3 = np.zeros((B, n_test))
    AUC = np.zeros(B)
    
    # Progressive ensemble building
    for i in range(B):
        np.random.seed(i)
        
        # Bootstrap sample
        idx = np.random.choice(len(train_data), len(train_data), replace=True)
        dt = train_data.iloc[idx].copy()
        
        # Prepare bootstrap training features
        X_train_boot = pd.get_dummies(dt[features], drop_first=True)
        y_train_boot = dt['survived']
        
        # Align test features
        X_test_aligned = X_test.reindex(columns=X_train_boot.columns, fill_value=0)
        
        # Build unpruned tree
        fit = DecisionTreeClassifier(
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None
        )
        fit.fit(X_train_boot, y_train_boot)
        
        # Store predictions
        phat3[i, :] = fit.predict_proba(X_test_aligned)[:, 1]
        
        # Calculate cumulative ensemble AUC
        phat_f = np.mean(phat3[:i+1, :], axis=0)
        AUC[i] = roc_auc_score(y_test, phat_f)
    
    # Plot AUC progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, B+1), AUC, 'r-', linewidth=2)
    plt.xlabel('B - Number of trees')
    plt.ylabel('AUC')
    plt.title('Ensemble AUC vs Number of Trees')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {'auc_progression': AUC, 'predictions': phat3}




####################################################################################################################






# ===================================
#           chunk_en6
# ===================================

def chunk_en6(random_state=1, n=500):
    """
    Convert R script en6 to Python: compare CART vs Random Forest on simulated data
    
    Args:
        random_state: seed for reproducible results
        n: number of data points to simulate
    
    Returns:
        dict: fitted models and generated data
    """
    # Generate simulated data
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, n)
    y = np.sin(12 * (x + 0.2)) / (x + 0.2) + np.random.normal(0, 0.5, n)
    
    # Reshape for sklearn
    X = x.reshape(-1, 1)
    
    # Fit models
    fit_tree = DecisionTreeRegressor(random_state=42)  # CART
    fit_rf1 = RandomForestRegressor(random_state=42, n_estimators=100)  # No depth control
    fit_rf2 = RandomForestRegressor(random_state=42, n_estimators=100, max_leaf_nodes=20)  # Control nodes
    
    fit_tree.fit(X, y)
    fit_rf1.fit(X, y)
    fit_rf2.fit(X, y)
    
    # Generate prediction points
    z = np.linspace(x.min(), x.max(), 1000)
    Z = z.reshape(-1, 1)
    
    # Make predictions
    pred_tree = fit_tree.predict(Z)
    pred_rf1 = fit_rf1.predict(Z)
    pred_rf2 = fit_rf2.predict(Z)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='gray', alpha=0.6, s=20)
    plt.plot(z, pred_rf1, color='green', linewidth=2, label='Random Forest: max_leaf_nodes=max')
    plt.plot(z, pred_rf2, color='red', linewidth=2, label='Random Forest: max_leaf_nodes=20')
    plt.plot(z, pred_tree, color='blue', linewidth=1.5, label='CART: single regression tree')
    
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.title('Comparison of CART vs Random Forest Models')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'data': {'x': x, 'y': y},
        'models': {'cart': fit_tree, 'rf_full': fit_rf1, 'rf_limited': fit_rf2}
    }




####################################################################################################################





# ===================================
#           chunk_en7
# ===================================

def chunk_en7(X, y, random_state=42):
    """
    Convert R script en7 to Python: evaluate Random Forest MSE across different max_leaf_nodes
    
    Args:
        X: feature matrix (n_samples, 1) 
        y: target values
        random_state: seed for reproducible results
    
    Returns:
        dict: MSE results for different configurations
    """
    maxnode = [10, 50, 100, 500]
    results = {'standard': [], 'high_trees': []}
    
    print("Standard Random Forest (100 trees):")
    for max_nodes in maxnode:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_leaf_nodes=max_nodes,
            random_state=random_state,
            oob_score=True
        )
        rf.fit(X, y)
        
        # Use OOB error as proxy for out-of-bag MSE
        oob_mse = mean_squared_error(y, rf.oob_prediction_)
        results['standard'].append((max_nodes, oob_mse))
        print(f"Max nodes: {max_nodes}, OOB MSE: {oob_mse:.6f}")
    
    print("\nHigh tree count Random Forest (2500 trees):")
    for max_nodes in maxnode:
        rf = RandomForestRegressor(
            n_estimators=2500,
            max_leaf_nodes=max_nodes,
            random_state=random_state,
            oob_score=True
        )
        rf.fit(X, y)
        
        oob_mse = mean_squared_error(y, rf.oob_prediction_)
        results['high_trees'].append((max_nodes, oob_mse))
        print(f"Max nodes: {max_nodes}, OOB MSE: {oob_mse:.6f}")
    
    return results




####################################################################################################################





# ===================================
#           chunk_en8
# ===================================

def chunk_en8(titanic_data, random_state=42):
    """
    Convert R script en8 to Python: Random Forest learning curve for Titanic survival
    
    Args:
        titanic_data: DataFrame with columns 'survived', 'sex', 'age', 'pclass', 'sibsp', 'parch'
        random_state: seed for reproducible results
    
    Returns:
        RandomForestClassifier: trained model with learning curves plotted
    """
    # Prepare data (equivalent to na.action=na.omit)
    features = ['sex', 'age', 'pclass', 'sibsp', 'parch']
    data_clean = titanic_data[features + ['survived']].dropna()
    
    X = pd.get_dummies(data_clean[features], drop_first=True)
    y = data_clean['survived'].astype(int)
    
    # Fit Random Forest with warm_start to track progress
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        oob_score=True,
        warm_start=True
    )
    
    # Track error rates as trees are added
    n_trees = range(1, 501, 10)
    oob_errors = []
    class_0_errors = []
    class_1_errors = []
    
    for n in n_trees:
        rf.set_params(n_estimators=n)
        rf.fit(X, y)
        
        # Calculate OOB error
        oob_error = 1 - rf.oob_score_
        oob_errors.append(oob_error)
        
        # Calculate class-specific errors using OOB predictions
        oob_proba = rf.oob_decision_function_
        oob_pred = np.argmax(oob_proba, axis=1)
        
        # Error for class 0 (Dead) and class 1 (Survived)
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        
        error_0 = np.mean(oob_pred[mask_0] != 0) if np.sum(mask_0) > 0 else 0
        error_1 = np.mean(oob_pred[mask_1] != 1) if np.sum(mask_1) > 0 else 0
        
        class_0_errors.append(error_0)
        class_1_errors.append(error_1)
    
    # Create learning curve plot
    plt.figure(figsize=(12, 8))
    plt.plot(n_trees, class_1_errors, 'g-', linewidth=2, label="Error for 'Survived'")
    plt.plot(n_trees, oob_errors, 'k-', linewidth=2, label='Misclassification error')
    plt.plot(n_trees, class_0_errors, 'r-', linewidth=2, label="Error for 'Dead'")
    
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Learning curve of the forest')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rf




####################################################################################################################





# No specific imports needed for this simple function

# ============================================================
#                        chunk_en9
# ============================================================

def chunk_en9(rf=None):
    """
    Converted from R script 'en9'.
    Returns the rf variable/object.
    
    Args:
        rf: The rf object/variable (default: None)
    
    Returns:
        The rf object
    """
    return rf



####################################################################################################################




# ============================================================
#                        chunk_en10
# ============================================================

def chunk_en10(n=300, seed=1, plot=True):
    """
    Simulate sinusoidal data with noise and optionally plot it.
    
    Args:
        n (int): Number of data points (default: 300)
        seed (int): Random seed for reproducibility (default: 1)
        plot (bool): Whether to create the plot (default: True)
    
    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns
    """
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 2 * np.pi, n))
    y = np.sin(x) + np.random.normal(0, 0.25, n)
    df = pd.DataFrame({"x": x, "y": y})
    
    if plot:
        plt.scatter(df["x"], df["y"], color="grey", alpha=0.6)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    return df



####################################################################################################################





#######################################
#           chunk_en11                #
#######################################

def chunk_en11(df, x_col='x', y_col='y', plot=True):
    """
    Fit regression tree and optionally plot results.
    
    Parameters:
    df: DataFrame with x and y columns
    x_col: name of x column (default 'x')
    y_col: name of y column (default 'y')
    plot: whether to create plot (default True)
    
    Returns:
    tuple: (fitted_model, predictions)
    """
    # Fit regression tree
    X = df[[x_col]].values
    y = df[y_col].values
    fit = DecisionTreeRegressor(random_state=42)
    fit.fit(X, y)
    yp = fit.predict(X)
    
    if plot:
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.7, label='Data')
        
        # Sort for step plot
        sort_idx = np.argsort(df[x_col])
        plt.plot(df[x_col].iloc[sort_idx], yp[sort_idx], 
                drawstyle='steps-post', color='blue', linewidth=3, label='Tree fit')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    
    return fit, yp



####################################################################################################################


#######################################
#           chunk_en12                #
#######################################

def chunk_en12(df, yp, h=0.1, y_col='y'):
    """
    Apply shrinkage adjustment to predictions and update dataframe.
    
    Parameters:
    df: DataFrame to modify
    yp: predictions from previous model
    h: shrinkage parameter (default 0.1)
    y_col: name of y column (default 'y')
    
    Returns:
    tuple: (modified_df, YP_cumulative)
    """
    # Calculate prediction error adjusted by shrinkage
    yr = df[y_col] - h * yp
    
    # Add adjusted residuals to dataframe
    df = df.copy()
    df['yr'] = yr
    
    # Store shrunk predictions
    YP = h * yp
    
    return df, YP



####################################################################################################################




#######################################
#           chunk_en13                #
#######################################

def chunk_en13(df, YP, h=0.1, n_trees=100, x_col='x'):
    """
    Perform boosting loop with multiple regression trees.
    
    Parameters:
    df: DataFrame with yr and x columns
    YP: initial predictions array/matrix
    h: shrinkage parameter (default 0.1)
    n_trees: number of boosting iterations (default 100)
    x_col: name of x column (default 'x')
    
    Returns:
    tuple: (modified_df, YP_matrix)
    """
    # Convert YP to numpy array if it isn't already
    if isinstance(YP, (list, pd.Series)):
        YP = np.array(YP).reshape(-1, 1)
    elif len(YP.shape) == 1:
        YP = YP.reshape(-1, 1)
    
    df = df.copy()
    
    # Boosting loop
    for t in range(n_trees):
        # Fit tree to current residuals
        fit = DecisionTreeRegressor(random_state=42+t)
        X = df[[x_col]].values
        fit.fit(X, df['yr'])
        
        # Predict on current data
        yp = fit.predict(X)
        
        # Add shrunk predictions to YP matrix
        YP = np.column_stack([YP, h * yp])
        
        # Update residuals for next iteration
        df['yr'] = df['yr'] - h * yp
    
    print(f"YP shape: {YP.shape}")
    
    return df, YP




####################################################################################################################





#######################################
#           chunk_en14                #
#######################################

def chunk_en14(df, YP, M_values=[5, 101], x_col='x', y_col='y'):
    """
    Plot comparison of boosted trees vs single tree for different M values.
    
    Parameters:
    df: DataFrame with x and y columns
    YP: predictions matrix from boosting
    M_values: list of M values to plot (default [5, 101])
    x_col: name of x column (default 'x')
    y_col: name of y column (default 'y')
    
    Returns:
    None (creates plots)
    """
    def pl(M):
        # Boosted prediction - sum first M columns
        yhat = np.sum(YP[:, :M], axis=1)
        
        # Single tree fit
        fit = DecisionTreeRegressor(random_state=42)
        X = df[[x_col]].values
        fit.fit(X, df[y_col])
        yp = fit.predict(X)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.7, label='Data')
        
        # Sort indices for step plots
        sort_idx = np.argsort(df[x_col])
        x_sorted = df[x_col].iloc[sort_idx]
        
        # Plot lines
        plt.plot(x_sorted, yhat[sort_idx], drawstyle='steps-post', 
                color='red', linewidth=3, label=f'Boosted (M={M})')
        plt.plot(x_sorted, yp[sort_idx], drawstyle='steps-post', 
                color='blue', linewidth=3, label='Single tree')
        plt.plot(x_sorted, np.sin(x_sorted), 
                color='black', linewidth=2, label='True DGM (sin(x))')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Boosting Comparison (M={M})')
        plt.show()
    
    # Run for each M value
    for M in M_values:
        pl(M)




####################################################################################################################




#######################################
#           chunk_en15                #
#######################################

def chunk_en15(df, yp, h=1.8, n_trees=100, x_col='x', y_col='y', plot=True):
    """
    Perform high shrinkage boosting and optionally plot results.
    
    Parameters:
    df: DataFrame with x and y columns
    yp: initial predictions from single tree
    h: shrinkage parameter (default 1.8)
    n_trees: number of boosting iterations (default 100)
    x_col: name of x column (default 'x')
    y_col: name of y column (default 'y')
    plot: whether to create comparison plot (default True)
    
    Returns:
    tuple: (modified_df, YP_matrix)
    """
    df = df.copy()
    
    # Initialize with high shrinkage
    df['yr'] = df[y_col] - h * yp
    YP = (h * yp).reshape(-1, 1)
    
    # Boosting loop
    for t in range(n_trees):
        fit = DecisionTreeRegressor(random_state=42+t)
        X = df[[x_col]].values
        fit.fit(X, df['yr'])
        
        yhat = fit.predict(X)
        df['yr'] = df['yr'] - h * yhat
        YP = np.column_stack([YP, h * yhat])
    
    if plot:
        # Plot comparison at M=101 (all trees)
        M = n_trees + 1
        boosted_pred = np.sum(YP[:, :M], axis=1)
        
        # Single tree for comparison
        fit_single = DecisionTreeRegressor(random_state=42)
        X = df[[x_col]].values
        fit_single.fit(X, df[y_col])
        single_pred = fit_single.predict(X)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.7, label='Data')
        
        sort_idx = np.argsort(df[x_col])
        x_sorted = df[x_col].iloc[sort_idx]
        
        plt.plot(x_sorted, boosted_pred[sort_idx], drawstyle='steps-post', 
                color='red', linewidth=3, label=f'Boosted (h={h})')
        plt.plot(x_sorted, single_pred[sort_idx], drawstyle='steps-post', 
                color='blue', linewidth=3, label='Single tree')
        plt.plot(x_sorted, np.sin(x_sorted), 
                color='black', linewidth=2, label='True DGM (sin(x))')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'High Shrinkage Boosting (h={h}, M={M})')
        plt.show()
    
    return df, YP



####################################################################################################################





#######################################
#           chunk_en16                #
#######################################

def chunk_en16(df, n_trees=100, t=None, x_col='x', y_col='y'):
    """
    Compare GBM models with different parameters using sklearn.
    
    Parameters:
    df: DataFrame with x and y columns
    n_trees: number of trees (default 100)
    t: number of trees to use for prediction (default uses all)
    x_col: name of x column (default 'x')
    y_col: name of y column (default 'y')
    
    Returns:
    tuple: (bo1_model, bo2_model)
    """
    X = df[[x_col]].values
    y = df[y_col].values
    
    if t is None:
        t = n_trees
    
    # Model 1: Custom parameters (no CV, subsample=1.0, shrinkage=0.1)
    bo1 = GradientBoostingRegressor(
        n_estimators=n_trees,
        learning_rate=0.1,
        subsample=1.0,  # bag.fraction = 1 (no subsampling)
        random_state=42
    )
    bo1.fit(X, y)
    
    # Model 2: Default parameters (includes subsampling)
    bo2 = GradientBoostingRegressor(
        random_state=42
    )
    bo2.fit(X, y)
    
    # Create predictions using first t trees
    pred1 = bo1.predict(X)  # sklearn doesn't support n_trees parameter like R
    pred2 = bo2.predict(X)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.7, label='Data')
    
    # Sort for step plots
    sort_idx = np.argsort(df[x_col])
    x_sorted = df[x_col].iloc[sort_idx]
    
    plt.plot(x_sorted, pred1[sort_idx], drawstyle='steps-post',
            color='red', linewidth=3, label='GBM (no subsample, lr=0.1)')
    plt.plot(x_sorted, pred2[sort_idx], drawstyle='steps-post', 
            color='green', linewidth=3, label='GBM (default params)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('GBM Comparison: Custom vs Default Parameters')
    plt.show()
    
    return bo1, bo2




####################################################################################################################



#######################################
#           chunk_en17                #
#######################################

def chunk_en17(n=100, err=0.2):
    """
    Calculate AdaBoost alpha parameter and exponential weight.
    
    Parameters:
    n: sample size (default 100)
    err: error rate (default 0.2)
    
    Returns:
    tuple: (alpha, exp_alpha)
    """
    # Calculate alpha (learning weight for weak learner)
    alpha = 0.5 * np.log((1 - err) / err)
    exp_alpha = np.exp(alpha)
    
    print(f"Alpha: {alpha:.6f}")
    print(f"exp(Alpha): {exp_alpha:.6f}")
    
    return alpha, exp_alpha



####################################################################################################################


#######################################
#           chunk_en18                #
#######################################

def chunk_en18(n=100, alpha=None, err=0.2):
    """
    Calculate AdaBoost weight updates for correct and misclassified observations.
    
    Parameters:
    n: sample size (default 100)
    alpha: alpha parameter (if None, calculated from err)
    err: error rate for alpha calculation (default 0.2)
    
    Returns:
    tuple: (weight_miss, weight_corr)
    """
    if alpha is None:
        alpha = 0.5 * np.log((1 - err) / err)
    
    # Weight for misclassified observations
    weight_miss = (1 / n) * np.exp(alpha)
    
    # Weight for correctly classified observations  
    weight_corr = (1 / n) * np.exp(-alpha)
    
    print(f"Weight for misclassified: {weight_miss:.6f}")
    print(f"Weight for correct: {weight_corr:.6f}")
    
    return weight_miss, weight_corr




####################################################################################################################



#######################################
#           chunk_en19                #
#######################################

def chunk_en19(filepath="myocarde.csv", n_rows=6):
    """
    Load myocarde dataset and initialize uniform weights.
    
    Parameters:
    filepath: path to CSV file (default "myocarde.csv")
    n_rows: number of rows to display (default 6)
    
    Returns:
    DataFrame: processed dataframe with weights
    """
    # Read CSV with semicolon delimiter
    myocarde = pd.read_csv(filepath, sep=';')
    
    # Get first n_rows
    df = myocarde.head(n_rows).copy()
    
    # Add uniform weights column
    df['Weights'] = 1 / len(df)
    
    print(df)
    
    return df




####################################################################################################################


#######################################
#           chunk_en20                #
#######################################

def chunk_en20(df):
    """
    Calculate AdaBoost parameters for myocarde dataset.
    
    Parameters:
    df: DataFrame (typically from chunk_en19)
    
    Returns:
    dict: alpha, exp_alpha, weight_miss, weight_corr
    """
    n = len(df)
    err = 1 / n  # Error rate = 1/n (minimum possible error)
    
    # Calculate alpha
    alpha = 0.5 * np.log((1 - err) / err)
    exp_alpha = np.exp(alpha)
    
    print(f"Alpha: {alpha:.6f}")
    print(f"exp(Alpha): {exp_alpha:.6f}")
    
    # Weight updates
    weight_miss = (1 / n) * exp_alpha
    weight_corr = (1 / n) * np.exp(-alpha)
    
    print(f"Weight for misclassified: {weight_miss:.6f}")
    print(f"Weight for correct: {weight_corr:.6f}")
    
    return {
        'alpha': alpha,
        'exp_alpha': exp_alpha, 
        'weight_miss': weight_miss,
        'weight_corr': weight_corr,
        'n': n,
        'err': err
    }



####################################################################################################################



# ====================================
#           chunk_en21
# ====================================

def chunk_en21(df, weight_miss, weight_corr):
    """
    Convert R script en21 to Python function.
    Adds new weights, normalizes them, and returns columns 8-11.
    """
    df = df.copy()
    df['New_weights'] = [weight_miss] + [weight_corr] * 5
    df['Norm_weights'] = df['New_weights'] / df['New_weights'].sum()
    return df.iloc[:, 7:11]  # columns 8-11 (0-indexed)




####################################################################################################################


# ====================================
#           chunk_gini
# ====================================

def chunk_gini():
    """
    Calculate Gini index for class labels with equal and weighted distributions.
    """
    # Define the class labels and weights
    class_labels = np.array([1, 1, 1, 0, 0, 0])
    weights = np.full(len(class_labels), 1/len(class_labels))
    
    # Calculate the proportion of each class
    unique, counts = np.unique(class_labels, return_counts=True)
    class_prop = counts / len(class_labels)
    
    # Calculate the Gini index
    gini_index = 1 - np.sum(class_prop**2)
    print(f"Initial Gini index: {gini_index}")
    
    # Change the weight of the first observation to 0.5
    weights[0] = 0.5
    weights[1:6] = 0.1
    
    # Recalculate the class proportions and Gini index
    weighted_sums = np.array([np.sum(weights[class_labels == label]) for label in unique])
    class_prop = weighted_sums / np.sum(weights)
    gini_index = 1 - np.sum(class_prop**2)
    print(f"Weighted Gini index: {gini_index}")
    
    return gini_index




####################################################################################################################




# ====================================
#        chunk_decision_tree
# ====================================

def chunk_decision_tree():
    """
    Build and visualize decision trees with equal and different weights.
    """
    # Define the data
    x = np.array([2, 4, 6, 8, 10, 12]).reshape(-1, 1)
    y = np.array([0, 1, 1, 0, 0, 0])
    weights1 = np.full(len(y), 1/len(y))
    weights2 = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
    
    # Build decision trees
    fit1 = DecisionTreeClassifier(min_samples_split=1, min_samples_leaf=1, 
                                  random_state=42)
    fit2 = DecisionTreeClassifier(min_samples_split=1, min_samples_leaf=1, 
                                  random_state=42)
    
    fit1.fit(x, y, sample_weight=weights1)
    fit2.fit(x, y, sample_weight=weights2)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_tree(fit1, ax=ax1, feature_names=['x'], class_names=['0', '1'], 
              filled=True, rounded=True)
    ax1.set_title('Decision Tree with Equal Weights')
    
    plot_tree(fit2, ax=ax2, feature_names=['x'], class_names=['0', '1'], 
              filled=True, rounded=True)
    ax2.set_title('Decision Tree with Different Weights')
    
    plt.tight_layout()
    plt.show()
    
    return fit1, fit2




####################################################################################################################


# ====================================
#           chunk_en24
# ====================================

def chunk_en24():
    """
    Generate error values, calculate alpha weights, and plot their relationship.
    """
    n = 1000
    np.random.seed(1)
    err = np.random.choice(np.arange(0, 1.01, 0.01), n, replace=True)
    alpha = 0.5 * np.log((1 - err) / err)
    ind = np.argsort(err)
    
    plt.figure(figsize=(8, 6))
    plt.plot(err[ind], alpha[ind], 'o-', color='red', linewidth=2, markersize=3)
    plt.xlabel('error (err)')
    plt.ylabel('alpha')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return err, alpha



####################################################################################################################


# ====================================
#           chunk_en25
# ====================================

def chunk_en25(csv_path="myocarde.csv"):
    """
    Implement AdaBoost algorithm using decision stumps on myocarde dataset.
    """
    # Data loading and preprocessing
    myocarde = pd.read_csv(csv_path, sep=';')
    y = (myocarde['PRONO'] == 'SURVIE').astype(int) * 2 - 1
    x = myocarde.iloc[:, :7]
    
    # Settings
    rnd = 100
    m = len(x)
    whts = np.full(m, 1/m)
    stumps = []
    alpha = np.zeros(rnd)
    y_hat = np.zeros(m)
    
    np.random.seed(123)
    
    # AdaBoost training loop
    for i in range(rnd):
        # Train decision stump
        stump = DecisionTreeClassifier(max_depth=1, random_state=i)
        stump.fit(x, y, sample_weight=whts)
        stumps.append(stump)
        
        # Predictions
        yhat = stump.predict(x)
        
        # Calculate weighted error
        e = np.sum((yhat != y) * whts)
        
        # Calculate alpha
        alpha[i] = 0.5 * np.log((1 - e) / e)
        
        # Update weights
        whts = whts * np.exp(-alpha[i] * y * yhat)
        whts = whts / np.sum(whts)
    
    # Final predictions using all stumps
    for i in range(rnd):
        pred = stumps[i].predict(x)
        y_hat += alpha[i] * pred
    
    print(f"Final prediction scores (first 10): {y_hat[:10]}")
    
    # Final classification
    pred = np.sign(y_hat)
    
    # Confusion matrix
    cm = confusion_matrix(y, pred)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=['Actual -1', 'Actual 1'], 
                      columns=['Pred -1', 'Pred 1']))
    
    return stumps, alpha, y_hat, pred




####################################################################################################################




# ====================================
#           chunk_en26
# ====================================

def chunk_en26(stumps):
    """
    Plot decision stumps from AdaBoost algorithm in a 2x3 grid.
    """
    plt_indices = [0, 4, 9, 29, 59, 89]  # 0-indexed for Python
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(plt_indices):
        plot_tree(stumps[idx], ax=axes[i], 
                 filled=True, 
                 rounded=True,
                 fontsize=10,
                 feature_names=[f'X{j}' for j in range(7)],
                 class_names=['-1', '1'])
        axes[i].set_title(f'Stump {idx + 1}')
    
    plt.tight_layout()
    plt.show()
    
    return fig



####################################################################################################################



# ====================================
#           chunk_en27
# ====================================

def chunk_en27(x, y, rnd=100):
    """
    Use sklearn's AdaBoostClassifier with decision stumps and display results.
    """
    # Create AdaBoost classifier with decision stumps
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=rnd,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=123
    )
    
    # Fit the model
    ada.fit(x, y)
    
    # Model summary information
    print("AdaBoost Summary:")
    print(f"Number of estimators: {ada.n_estimators}")
    print(f"Number of features: {ada.n_features_in_}")
    print(f"Training score: {ada.score(x, y):.4f}")
    
    # Make predictions
    pred = ada.predict(x)
    
    # Confusion matrix
    cm = confusion_matrix(y, pred)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=['Actual -1', 'Actual 1'], 
                      columns=['Pred -1', 'Pred 1']))
    
    return ada, pred




####################################################################################################################





# ====================================
#           chunk_en28
# ====================================

def chunk_en28():
    """
    Load the Ames housing dataset and display its dimensions.
    """
    # Load Ames housing dataset from OpenML
    ames = fetch_openml(name='house_prices', as_frame=True, version=1)
    ames_df = ames.data
    
    print(f"Ames dataset dimensions: {ames_df.shape}")
    
    return ames_df



####################################################################################################################



# ====================================
#           chunk_en29
# ====================================

def chunk_en29(ames_df):
    """
    One-hot encode the Ames dataset and create train/test splits with bootstrap sampling.
    """
    # One-hot encode categorical columns
    categorical_cols = ames_df.select_dtypes(include=['object', 'category']).columns
    ct = make_column_transformer(
        (OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
        remainder='passthrough'
    )
    
    # Transform the data
    ames_encoded = ct.fit_transform(ames_df)
    
    # Get feature names
    feature_names = ct.get_feature_names_out()
    df = pd.DataFrame(ames_encoded, columns=feature_names)
    
    # Bootstrap sampling for train set
    np.random.seed(42)
    ind = np.random.choice(len(df), len(df), replace=True)
    train = df.iloc[ind].reset_index(drop=True)
    
    # Test set (out-of-bag samples)
    test_mask = ~np.isin(np.arange(len(df)), ind)
    test = df[test_mask].reset_index(drop=True)
    
    # Separate features and target (assuming Sale_Price is the target)
    price_col = [col for col in train.columns if 'sale_price' in col.lower()]
    if price_col:
        X = train.drop(columns=price_col).values
        Y = train[price_col[0]].values
    else:
        print("Sale_Price column not found, using last column as target")
        X = train.iloc[:, :-1].values
        Y = train.iloc[:, -1].values
    
    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Feature matrix X shape: {X.shape}")
    
    return df, train, test, X, Y




####################################################################################################################


# ====================================
#           chunk_en30
# ====================================

def chunk_en30(X, Y):
    """
    Perform XGBoost cross-validation with hyperparameter tuning.
    """
    # Set parameters
    params = {
        'eta': 0.1,                    # Learning rate (step size in boosting)
        'max_depth': 3,                # Maximum depth of the tree
        'min_child_weight': 3,         # Minimum number of instances in each node
        'subsample': 0.8,              # Subsample ratio of training instances
        'colsample_bytree': 1.0,       # Fraction of columns to be subsampled
        'objective': 'reg:squarederror', # Regression objective
        'seed': 123                    # Random seed
    }
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X, label=Y)
    
    # Perform cross-validation
    boost = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,          # Maximum number of iterations
        nfold=10,                      # 10-fold cross-validation
        early_stopping_rounds=50,      # Stop if no improvement after 50 rounds
        verbose_eval=False,            # Silent mode
        seed=123
    )
    
    print(f"Best iteration: {boost.shape[0]}")
    print(f"Best CV score: {boost['test-rmse-mean'].min():.6f}")
    
    return boost




####################################################################################################################





# ====================================
#           chunk_en31
# ====================================

def chunk_en31(boost):
    """
    Extract best iteration and RMSE from XGBoost CV results and demonstrate grid search setup.
    """
    # Extract best iteration and minimum RMSE
    best_it = boost.shape[0]  # Number of actual iterations (with early stopping)
    min_rmse = boost['test-rmse-mean'].min()
    
    print(f"Best iteration: {best_it}")
    print(f"Minimum test RMSE: {min_rmse:.6f}")
    
    # Example parameter grid for hyperparameter tuning
    param_grid = {
        'eta': [0.01],
        'max_depth': [3],
        'min_child_weight': [3],
        'subsample': [0.5],
        'colsample_bytree': [0.5],
        'gamma': [0, 1, 10, 100, 1000],
        'reg_lambda': [0, 0.01, 0.1, 1, 100, 1000],  # L2 regularization
        'reg_alpha': [0, 0.01, 0.1, 1, 100, 1000]    # L1 regularization
    }
    
    # Generate all parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nTotal parameter combinations: {len(param_combinations)}")
    print("Example parameter combination:")
    print(param_combinations[0])
    
    # Note: In practice, you would loop through param_combinations
    # and run xgb.cv for each combination to find optimal parameters
    
    return best_it, min_rmse, param_combinations



####################################################################################################################





# ====================================
#           chunk_en32
# ====================================

def chunk_en32(X, Y, params, best_it):
    """
    Train final XGBoost model using optimal number of rounds from CV.
    """
    # Create DMatrix for training
    dtrain = xgb.DMatrix(X, label=Y)
    
    # Train the final model
    tr_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_it,
        verbose_eval=False
    )
    
    print(f"Model trained with {best_it} boosting rounds")
    
    return tr_model




####################################################################################################################



# ====================================
#           chunk_en33
# ====================================

def chunk_en33(tr_model, feature_names=None):
    """
    Create variable importance plot for XGBoost model.
    """
    # Get feature importance
    importance = tr_model.get_score(importance_type='weight')  # Can also use 'gain' or 'cover'
    
    # Convert to DataFrame and sort
    if feature_names is None:
        feature_names = list(importance.keys())
    
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                    color='orange', edgecolor='green', linewidth=1.5)
    
    plt.xlabel('Importance')
    plt.title('Variable Importance Plot')
    plt.tight_layout()
    
    # Show only top 20 features if there are many
    if len(importance_df) > 20:
        top_20 = importance_df.tail(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top_20['feature'], top_20['importance'], 
                 color='orange', edgecolor='green', linewidth=1.5)
        plt.xlabel('Importance')
        plt.title('Top 20 Variable Importance')
        plt.tight_layout()
    
    plt.show()
    
    return importance_df




####################################################################################################################




# ====================================
#           chunk_en34
# ====================================

def chunk_en34(tr_model, test):
    """
    Make predictions on test set and calculate RMSE.
    """
    # Find Sale_Price column
    price_col = [col for col in test.columns if 'sale_price' in col.lower()]
    
    if price_col:
        # Separate features and target
        X_test = test.drop(columns=price_col).values
        y_test = test[price_col[0]].values
    else:
        # Fallback: assume last column is target
        X_test = test.iloc[:, :-1].values
        y_test = test.iloc[:, -1].values
    
    # Create DMatrix and make predictions
    dtest = xgb.DMatrix(X_test)
    yhat = tr_model.predict(dtest)
    
    # Calculate RMSE
    rmse_test = np.sqrt(np.mean((y_test - yhat) ** 2))
    
    print(f"Test RMSE: {rmse_test:.6f}")
    
    return yhat, rmse_test