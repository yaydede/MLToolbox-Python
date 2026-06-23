import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




# ============================================
#              chunk_tr1
# ============================================

def chunk_tr1(image_path="png/DT.png"):
    """
    Display an image file.
    
    Args:
        image_path: str, path to image file
    """
    try:
        img = mpimg.imread(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found.")





###################################################################################################################


# ============================================
#              chunk_tr2
# ============================================

def chunk_tr2():
    """
    Create scatter plot of x1 vs x2 colored by y values.
    
    Returns:
        pandas.DataFrame: The created dataset
    """
    y = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    x1 = [0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.62, 0.83, 0.88]
    x2 = [0.5, 0.82, 0.2, 0.09, 0.58, 0.5, 0.93, 0.8, 0.3, 0.83]
    
    data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
    
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'black' for val in y]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.8)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return data





###################################################################################################################



# ============================================
#              chunk_tr3
# ============================================

def chunk_tr3(data):
    """
    Create scatter plot with horizontal line at x2 = 0.62.
    
    Args:
        data: pandas.DataFrame with columns y, x1, x2
    """
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'black' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.8)
    
    # Add horizontal line at x2 = 0.62
    plt.axhline(y=0.62, color='blue', linestyle='--', linewidth=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()



###################################################################################################################



# ============================================
#            chunk_tree1
# ============================================

def chunk_tree1(image_path="png/tree1.png"):
    """
    Display tree1 image file.
    
    Args:
        image_path: str, path to tree1 image file
    """
    try:
        img = mpimg.imread(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found.")





###################################################################################################################



# ============================================
#              chunk_tr4
# ============================================

def chunk_tr4(data):
    """
    Create scatter plot with horizontal and vertical lines.
    
    Args:
        data: pandas.DataFrame with columns y, x1, x2
    """
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'black' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.8)
    
    # Add horizontal line at x2 = 0.62 (blue)
    plt.axhline(y=0.62, color='blue', linestyle='--', linewidth=2)
    
    # Add vertical line at x1 = 0.2 (darkgreen)
    plt.axvline(x=0.2, color='darkgreen', linestyle='--', linewidth=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()



###################################################################################################################



# ============================================
#              chunk_tr5
# ============================================

def chunk_tr5(image_path="png/tree2.png"):
    """
    Display tree2 image file.
    
    Args:
        image_path: str, path to tree2 image file
    """
    try:
        img = mpimg.imread(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found.")





###################################################################################################################


# ============================================
#              chunk_tr6
# ============================================

def chunk_tr6(file_path="myocarde.csv"):
    """
    Load and display structure of myocarde dataset.
    
    Args:
        file_path: str, path to CSV file
        
    Returns:
        pandas.DataFrame: The loaded myocarde dataset
    """
    # Load data with semicolon delimiter
    myocarde = pd.read_csv(file_path, sep=';', skipinitialspace=True)
    
    # Display dataset info
    print("Dataset shape:", myocarde.shape)
    print("\nColumn data types:")
    print(myocarde.dtypes)
    print("\nFirst few rows:")
    print(myocarde.head())
    
    return myocarde






###################################################################################################################


# ============================================================================
#                                chunk_tr7
# ============================================================================

def chunk_tr7(myocarde):
    """
    Calculate G(N) statistic for binary outcome data.
    
    Args:
        myocarde: DataFrame with PRONO column
        
    Returns:
        float: G(N) statistic value
    """
    # Recode PRONO
    y = np.where(myocarde['PRONO'] == 'SURVIE', 1, 0)
    
    # Find G(N) without L and R
    mean_y = np.mean(y)
    G = 2 * mean_y * (1 - mean_y)
    
    return G





###################################################################################################################


# ============================================================================
#                                chunk_tr8
# ============================================================================

def chunk_tr8(myocarde, y):
    """
    Create contingency table between binary outcome and FRCAR variable.
    
    Args:
        myocarde: DataFrame with FRCAR column
        y: Binary outcome variable (array-like)
        
    Returns:
        pandas.DataFrame: Contingency table
    """
    # Pick FRCAR variable
    x_1 = myocarde['FRCAR']
    
    # Create contingency table
    tab = pd.crosstab(y, x_1, margins=False)
    
    return tab





###################################################################################################################



# ============================================================================
#                                chunk_tr9
# ============================================================================

def chunk_tr9(y, x_1, threshold=60):
    """
    Calculate GL, GR statistics and proportions for binary split.
    
    Args:
        y: Binary outcome variable (array-like)
        x_1: Predictor variable (array-like)
        threshold: Split threshold (default=60)
        
    Returns:
        dict: Dictionary with GL, GR, pL, pR values
    """
    # Convert to numpy arrays for easier indexing
    y = np.array(y)
    x_1 = np.array(x_1)
    
    # Calculate GL and GR
    left_mask = x_1 <= threshold
    right_mask = x_1 > threshold
    
    mean_y_left = np.mean(y[left_mask])
    mean_y_right = np.mean(y[right_mask])
    
    GL = 2 * mean_y_left * (1 - mean_y_left)
    GR = 2 * mean_y_right * (1 - mean_y_right)
    
    # Calculate proportions
    pL = np.sum(left_mask) / len(x_1)  # Proportion of obs. on Left
    pR = np.sum(right_mask) / len(x_1)  # Proportion of obs. on Right
    
    return {'GL': GL, 'GR': GR, 'pL': pL, 'pR': pR}





###################################################################################################################



# ============================================================================
#                                chunk_tr10
# ============================================================================

def chunk_tr10(G, pL, GL, pR, GR):
    """
    Calculate delta statistic for binary split improvement.
    
    Args:
        G: Overall G statistic
        pL: Proportion of observations on left
        GL: G statistic for left split
        pR: Proportion of observations on right  
        GR: G statistic for right split
        
    Returns:
        float: Delta improvement statistic
    """
    delta = G - pL * GL - pR * GR
    return delta





###################################################################################################################


# ============================================================================
#                                chunk_tr11
# ============================================================================

def chunk_tr11(y, x_1, G):
    """
    Create GI function that calculates delta improvement for any threshold.
    
    Args:
        y: Binary outcome variable (array-like)
        x_1: Predictor variable (array-like)
        G: Overall G statistic
        
    Returns:
        function: GI function that takes threshold x and returns delta
    """
    # Convert to numpy arrays
    y = np.array(y)
    x_1 = np.array(x_1)
    
    def GI(x):
        left_mask = x_1 <= x
        right_mask = x_1 > x
        
        mean_y_left = np.mean(y[left_mask])
        mean_y_right = np.mean(y[right_mask])
        
        GL = 2 * mean_y_left * (1 - mean_y_left)
        GR = 2 * mean_y_right * (1 - mean_y_right)
        pL = np.sum(left_mask) / len(x_1)
        pR = np.sum(right_mask) / len(x_1)
        
        delta = G - pL * GL - pR * GR
        return delta
    
    return GI





###################################################################################################################


# ============================================================================
#                                chunk_tr12
# ============================================================================

def chunk_tr12(x_1, GI_func):
    """
    Calculate delta values for all unique split points except the last.
    
    Args:
        x_1: Predictor variable (array-like)
        GI_func: GI function that calculates delta for given threshold
        
    Returns:
        numpy.array: Array of delta values for each split point
    """
    xm = np.sort(np.unique(x_1))
    delta = []
    
    # Since we don't split at the last number
    for i in range(len(xm) - 1):
        delta.append(GI_func(xm[i]))
    
    return np.array(delta)



###################################################################################################################


# ============================================================================
#                                chunk_tr13
# ============================================================================

def chunk_tr13(delta, x_1):
    """
    Find maximum delta value and corresponding optimal split point.
    
    Args:
        delta: Array of delta values for each split point
        x_1: Predictor variable (array-like)
        
    Returns:
        dict: Dictionary with max_delta and optimal_split values
    """
    xm = np.sort(np.unique(x_1))
    
    max_delta = np.max(delta)
    optimal_split = xm[np.argmax(delta)]
    
    return {'max_delta': max_delta, 'optimal_split': optimal_split}




###################################################################################################################


# ============================================================================
#                                chunk_tr14
# ============================================================================

def chunk_tr14(myocarde, y):
    """
    Find optimal splits for all variables using adjusted GI function.
    
    Args:
        myocarde: DataFrame with predictor variables
        y: Binary outcome variable (array-like)
        
    Returns:
        pandas.DataFrame: Variables with their maximum delta values
    """
    # Convert y to numpy array
    y = np.array(y)
    
    # Adjusted GI function with cutoff parameter
    def GI(x, tr):
        G = 2 * np.mean(y) * (1 - np.mean(y))
        left_mask = x <= tr
        right_mask = x > tr
        
        GL = 2 * np.mean(y[left_mask]) * (1 - np.mean(y[left_mask]))
        GR = 2 * np.mean(y[right_mask]) * (1 - np.mean(y[right_mask]))
        pL = np.sum(left_mask) / len(x)
        pR = np.sum(right_mask) / len(x)
        
        delta = G - pL * GL - pR * GR
        return delta
    
    # Apply GI on every variable
    d = myocarde.iloc[:, 0:7]
    split = []
    maxdelta = []
    
    for j in range(d.shape[1]):
        x_col = d.iloc[:, j].values
        xm = np.sort(np.unique(x_col))
        delta = []
        
        for i in range(len(xm) - 1):
            delta.append(GI(x_col, xm[i]))
        
        maxdelta.append(np.max(delta))
        split.append(xm[np.argmax(delta)])
    
    return pd.DataFrame({
        'variables': d.columns,
        'delta': maxdelta
    })




###################################################################################################################


# ============================================================================
#                                chunk_tr15
# ============================================================================

def chunk_tr15(split, maxdelta):
    """
    Find optimal split point for variable with highest delta.
    
    Args:
        split: Array of optimal split points for each variable
        maxdelta: Array of maximum delta values for each variable
        
    Returns:
        int: Rounded optimal split point for best variable
    """
    optimal_split = split[np.argmax(maxdelta)]
    return round(optimal_split, 0)




###################################################################################################################


# ============================================================================
#                                chunk_tr16
# ============================================================================

def chunk_tr16(maxdelta, myocarde):
    """
    Create horizontal bar plot of variable importance at first split.
    
    Args:
        maxdelta: Array of maximum delta values for each variable
        myocarde: DataFrame to get variable names
        
    Returns:
        matplotlib.figure.Figure: The created plot figure
    """
    # Create matrix and sort by delta values
    variable_names = list(myocarde.columns[0:7])
    dm = np.array(maxdelta).reshape(-1, 1)
    
    # Sort indices by delta values
    sort_idx = np.argsort(dm[:, 0])
    sorted_deltas = dm[sort_idx, 0]
    sorted_names = [variable_names[i] for i in sort_idx]
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_names, sorted_deltas, color='darkgreen')
    
    ax.set_xlim(0, 0.3)
    ax.set_title('Variable Importance at the 1st Split', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout()
    return fig



###################################################################################################################


# ============================================================================
#                                chunk_tr17
# ============================================================================

def chunk_tr17(myocarde, target_col='PRONO'):
    """
    Build and visualize decision tree for classification.
    
    Args:
        myocarde: DataFrame with predictor variables and target
        target_col: Name of target column (default='PRONO')
        
    Returns:
        tuple: (fitted DecisionTreeClassifier, matplotlib figure)
    """
    # Prepare data
    X = myocarde.drop(columns=[target_col])
    y = myocarde[target_col]
    
    # Build decision tree
    tree = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree.fit(X, y)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(tree, 
             feature_names=X.columns,
             class_names=tree.classes_,
             filled=True,
             rounded=True,
             fontsize=10,
             ax=ax)
    
    ax.set_title('Decision Tree for PRONO Classification', fontsize=14, pad=20)
    plt.tight_layout()
    
    return tree, fig




###################################################################################################################


# ============================================================================
#                                chunk_tr18
# ============================================================================

def chunk_tr18(tree, X, y):
    """
    Display complexity parameter table for decision tree (equivalent to printcp).
    
    Args:
        tree: Fitted DecisionTreeClassifier
        X: Feature matrix
        y: Target variable
        
    Returns:
        pandas.DataFrame: CP table with tree complexity statistics
    """
    # Get tree complexity parameters
    path = tree.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    
    # Build CP table
    cp_table = []
    for i, alpha in enumerate(ccp_alphas):
        # Create pruned tree
        pruned_tree = DecisionTreeClassifier(
            random_state=42, 
            ccp_alpha=alpha,
            max_depth=tree.max_depth
        )
        pruned_tree.fit(X, y)
        
        # Calculate cross-validation error
        cv_scores = cross_val_score(pruned_tree, X, y, cv=10, scoring='accuracy')
        xerror = 1 - cv_scores.mean()
        xstd = cv_scores.std()
        
        cp_table.append({
            'CP': alpha,
            'nsplit': pruned_tree.tree_.node_count // 2,  # Approximate splits
            'rel error': 1 - pruned_tree.score(X, y),
            'xerror': xerror,
            'xstd': xstd
        })
    
    return pd.DataFrame(cp_table)




###################################################################################################################


# ============================================================================
#                                chunk_tr19
# ============================================================================

def chunk_tr19(myocarde):
    """
    Build regression tree with binary encoded target and display CP table.
    
    Args:
        myocarde: DataFrame with PRONO column
        
    Returns:
        tuple: (fitted DecisionTreeRegressor, CP table DataFrame)
    """
    # Create modified dataset with binary encoding
    myocarde_v2 = myocarde.copy()
    myocarde_v2['PRONO'] = np.where(myocarde['PRONO'] == 'SURVIE', 1, 0)
    
    # Prepare data
    X = myocarde_v2.drop(columns=['PRONO'])
    y = myocarde_v2['PRONO']
    
    # Build regression tree (equivalent to rpart with continuous target)
    cart = DecisionTreeRegressor(random_state=42)
    cart.fit(X, y)
    
    # Get complexity parameters
    path = cart.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    
    # Build CP table
    cp_table = []
    for i, alpha in enumerate(ccp_alphas):
        pruned_tree = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        pruned_tree.fit(X, y)
        
        # Calculate metrics
        train_score = pruned_tree.score(X, y)
        cv_scores = cross_val_score(pruned_tree, X, y, cv=10, scoring='r2')
        
        cp_table.append({
            'CP': alpha,
            'nsplit': max(0, pruned_tree.tree_.node_count - pruned_tree.tree_.n_leaves),
            'rel error': 1 - train_score,
            'xerror': 1 - cv_scores.mean(),
            'xstd': cv_scores.std()
        })
    
    cp_df = pd.DataFrame(cp_table)
    return cart, cp_df




###################################################################################################################


# ============================================================================
#                                chunk_tr20
# ============================================================================

def chunk_tr20(tree, feature_names):
    """
    Create horizontal bar plot of variable importance from decision tree.
    
    Args:
        tree: Fitted decision tree (DecisionTreeClassifier or DecisionTreeRegressor)
        feature_names: List of feature names
        
    Returns:
        matplotlib.figure.Figure: The created plot figure
    """
    # Get variable importance
    vi = tree.feature_importances_
    
    # Create dictionary mapping feature names to importance
    importance_dict = dict(zip(feature_names, vi))
    
    # Sort by importance (ascending order to match R)
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])
    sorted_names = [item[0] for item in sorted_items]
    sorted_values = [item[1] / 100 for item in sorted_items]  # Divide by 100 as in R
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_names, sorted_values, color='lightgreen')
    
    ax.set_title('Variable Importance - rpart()', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout()
    return fig



###################################################################################################################


# ============================================================================
#                                chunk_tr21
# ============================================================================

def chunk_tr21(myocarde, target_col='PRONO'):
    """
    Build decision tree with relaxed constraints and create fancy visualization.
    
    Args:
        myocarde: DataFrame with predictor variables and target
        target_col: Name of target column (default='PRONO')
        
    Returns:
        tuple: (fitted DecisionTreeClassifier, matplotlib figure)
    """
    # Prepare data
    X = myocarde.drop(columns=[target_col])
    y = myocarde[target_col]
    
    # Build decision tree with relaxed parameters
    # minsplit=2 -> min_samples_split=2
    # minbucket=1 -> min_samples_leaf=1
    # cp=0 -> ccp_alpha=0 (no complexity pruning)
    tree2 = DecisionTreeClassifier(
        min_samples_split=2,
        min_samples_leaf=1,
        ccp_alpha=0,
        random_state=42
    )
    tree2.fit(X, y)
    
    # Create fancy visualization (equivalent to fancyRpartPlot)
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(tree2,
             feature_names=X.columns,
             class_names=tree2.classes_,
             filled=True,
             rounded=True,
             fontsize=8,
             proportion=True,
             impurity=False,
             ax=ax)
    
    # Styling to make it "fancy"
    ax.set_facecolor('white')
    plt.tight_layout()
    
    return tree2, fig




###################################################################################################################


# ===============================================
#                  chunk_tr22
# ===============================================

def chunk_tr22(tree_model):
    """
    Display complexity parameter table, plot CP values, and find optimal CP.
    
    Args:
        tree_model: Fitted decision tree model (sklearn DecisionTreeClassifier/Regressor)
    
    Returns:
        float: Minimum complexity parameter value
    """
    # Print CP table equivalent (tree path info)
    print(f"Tree depth: {tree_model.get_depth()}")
    print(f"Number of leaves: {tree_model.get_n_leaves()}")
    
    # For sklearn trees, we work with the regularization parameter alpha (ccp_alpha)
    # which is equivalent to R's CP parameter
    if hasattr(tree_model, 'ccp_alpha'):
        min_cp = tree_model.ccp_alpha
        print(f"Current CCP Alpha (CP): {min_cp}")
        
        # Simple visualization of tree complexity
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Tree structure summary
        ax1.bar(['Depth', 'Leaves'], [tree_model.get_depth(), tree_model.get_n_leaves()])
        ax1.set_title('Tree Complexity')
        ax1.set_ylabel('Count')
        
        # Plot 2: Feature importance (as proxy for CP plot)
        if hasattr(tree_model, 'feature_importances_'):
            importances = tree_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            ax2.bar(range(len(indices)), importances[indices])
            ax2.set_title('Top Feature Importances')
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Importance')
        
        plt.tight_layout()
        plt.show()
        
        return min_cp
    else:
        print("Model does not have complexity parameter information")
        return None




###################################################################################################################




# ===============================================
#                  chunk_tr23
# ===============================================

def chunk_tr23(tree_model, min_cp, X_train=None, y_train=None, feature_names=None, class_names=None):
    """
    Prune decision tree with optimal CP and display results.
    
    Args:
        tree_model: Original fitted decision tree model
        min_cp: Minimum complexity parameter for pruning
        X_train: Training features (needed for retraining with new CP)
        y_train: Training target (needed for retraining with new CP)
        feature_names: List of feature names for plotting
        class_names: List of class names for classification trees
    
    Returns:
        Pruned tree model
    """
    # Create pruned tree with optimal CP
    if isinstance(tree_model, DecisionTreeClassifier):
        pruned_tree = DecisionTreeClassifier(
            ccp_alpha=min_cp,
            random_state=getattr(tree_model, 'random_state', None),
            **{k: v for k, v in tree_model.get_params().items() 
               if k not in ['ccp_alpha', 'random_state']}
        )
    else:
        pruned_tree = DecisionTreeRegressor(
            ccp_alpha=min_cp,
            random_state=getattr(tree_model, 'random_state', None),
            **{k: v for k, v in tree_model.get_params().items() 
               if k not in ['ccp_alpha', 'random_state']}
        )
    
    if X_train is not None and y_train is not None:
        pruned_tree.fit(X_train, y_train)
        
        # Print pruned tree info
        print(f"Pruned tree depth: {pruned_tree.get_depth()}")
        print(f"Pruned tree leaves: {pruned_tree.get_n_leaves()}")
        print(f"Applied CCP Alpha: {min_cp}")
        
        # Fancy tree plot
        plt.figure(figsize=(15, 10))
        plot_tree(pruned_tree, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title('Pruned Decision Tree', fontsize=14, fontweight='bold')
        plt.show()
        
        return pruned_tree
    else:
        print("Training data required for pruning. Returning original tree.")
        return tree_model





###################################################################################################################




# ===============================================
#                  chunk_tr24
# ===============================================

def chunk_tr24():
    """
    Load Titanic dataset and display structure information.
    
    Returns:
        pandas.DataFrame: Titanic dataset
    """
    # Load Titanic dataset from seaborn (equivalent to PASWR::titanic3)
    titanic3 = sns.load_dataset('titanic')
    
    # Display structure equivalent to str() in R
    print("Titanic Dataset Structure:")
    print(f"Shape: {titanic3.shape}")
    print(f"Columns: {list(titanic3.columns)}")
    print("\nData types:")
    print(titanic3.dtypes)
    print(f"\nMemory usage: {titanic3.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    return titanic3




###################################################################################################################



# ===============================================
#                  chunk_tr25
# ===============================================

def chunk_tr25(titanic_data):
    """
    Train decision tree on Titanic survival data and create visualizations.
    
    Args:
        titanic_data: Titanic dataset (pandas DataFrame)
    
    Returns:
        Trained DecisionTreeClassifier model
    """
    # Prepare data - handle missing values and encode categoricals
    df = titanic_data.copy()
    df = df.dropna(subset=['age', 'sex', 'pclass', 'sibsp', 'parch', 'survived'])
    
    # Encode sex as binary
    df['sex_encoded'] = (df['sex'] == 'male').astype(int)
    
    # Select features and target
    features = ['sex_encoded', 'age', 'pclass', 'sibsp', 'parch']
    X = df[features]
    y = df['survived']
    
    # Train decision tree
    titan = DecisionTreeClassifier(random_state=42, max_depth=6)
    titan.fit(X, y)
    
    # Create tree plot with colors
    plt.figure(figsize=(15, 10))
    plot_tree(titan, 
             feature_names=['sex', 'age', 'pclass', 'sibsp', 'parch'],
             class_names=['Died', 'Survived'],
             filled=True,
             rounded=True,
             fontsize=9)
    plt.title('Titanic Survival Decision Tree', fontsize=14, fontweight='bold')
    plt.show()
    
    # Variable importance barplot
    importances = titan.feature_importances_
    feature_names = ['sex', 'age', 'pclass', 'sibsp', 'parch']
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importances, color='goldenrod')
    plt.yticks(y_pos, feature_names, fontsize=10)
    plt.xlabel('Variable Importance', fontsize=12)
    plt.title('Feature Importance in Titanic Survival Model', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print(f"Model accuracy on training data: {titan.score(X, y):.3f}")
    
    return titan




###################################################################################################################



# ===============================================
#                  chunk_tr26
# ===============================================

def chunk_tr26(titan_model, X_data, y_data):
    """
    Display complexity parameter information and cross-validation plot for decision tree.
    
    Args:
        titan_model: Trained DecisionTreeClassifier
        X_data: Training features
        y_data: Training target
    
    Returns:
        None
    """
    # Print CP table equivalent
    print("Decision Tree Complexity Information:")
    print(f"Tree depth: {titan_model.get_depth()}")
    print(f"Number of leaves: {titan_model.get_n_leaves()}")
    print(f"Number of nodes: {titan_model.tree_.node_count}")
    print(f"Current CCP Alpha: {titan_model.ccp_alpha}")
    
    # Create validation curve for different alpha values (CP equivalent)
    alpha_range = np.logspace(-4, -1, 20)
    train_scores, val_scores = validation_curve(
        titan_model.__class__(random_state=42),
        X_data, y_data,
        param_name='ccp_alpha',
        param_range=alpha_range,
        cv=5,
        scoring='accuracy'
    )
    
    # Plot CP curve equivalent
    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.semilogx(alpha_range, train_mean, 'o-', color='blue', label='Training accuracy')
    plt.fill_between(alpha_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.semilogx(alpha_range, val_mean, 'o-', color='red', label='Cross-validation accuracy')
    plt.fill_between(alpha_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('CCP Alpha (Complexity Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy vs Complexity Parameter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find and print optimal alpha
    optimal_idx = np.argmax(val_mean)
    optimal_alpha = alpha_range[optimal_idx]
    print(f"Optimal CCP Alpha: {optimal_alpha:.6f}")
    print(f"CV Accuracy at optimal alpha: {val_mean[optimal_idx]:.3f}")
    
    return optimal_alpha



###################################################################################################################




# ===============================================
#                  chunk_tr27
# ===============================================

def chunk_tr27(titanic_data, random_state=1):
    """
    Perform train/test split, train decision tree, and calculate AUC.
    
    Args:
        titanic_data: Titanic dataset (pandas DataFrame)
        random_state: Random seed for reproducible results
    
    Returns:
        tuple: (trained_model, test_auc_score, predictions_proba)
    """
    # Prepare data - handle missing values and encode categoricals
    df = titanic_data.copy()
    df = df.dropna(subset=['age', 'sex', 'pclass', 'sibsp', 'parch', 'survived'])
    
    # Encode sex as binary
    df['sex_encoded'] = (df['sex'] == 'male').astype(int)
    
    # Select features and target
    features = ['sex_encoded', 'age', 'pclass', 'sibsp', 'parch']
    X = df[features]
    y = df['survived']
    
    # Train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Train decision tree on training data
    titan2 = DecisionTreeClassifier(random_state=random_state, max_depth=6)
    titan2.fit(X_train, y_train)
    
    # Get probability predictions on test set
    phat = titan2.predict_proba(X_test)
    
    # Calculate AUC using probability of survival (class 1)
    test_auc = roc_auc_score(y_test, phat[:, 1])
    
    print(f"Test set size: {len(X_test)} samples")
    print(f"Training set size: {len(X_train)} samples")
    print(f"AUC Score: {test_auc:.4f}")
    
    return titan2, test_auc, phat




###################################################################################################################



# ===============================================
#                  chunk_tr28
# ===============================================

def chunk_tr28(random_state=1):
    """
    Generate simulated polynomial data and fit simple decision tree.
    
    Args:
        random_state: Random seed for reproducible results
    
    Returns:
        tuple: (fitted_tree_model, dataframe_with_data)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate simulated data
    x = np.random.uniform(-2, 2, 100)
    y = 1 + 1*x + 4*(x**2) - 4*(x**3) + np.random.normal(0, 6, 100)
    dt = pd.DataFrame({'x': x, 'y': y})
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='gray', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simulated Polynomial Data')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Fit regression tree with minimal splits (equivalent to minsplit=83)
    fit1 = DecisionTreeRegressor(
        min_samples_split=83,
        max_depth=2,  # Limit depth to get simple tree
        random_state=random_state
    )
    fit1.fit(dt[['x']], dt['y'])
    
    # Fancy tree plot
    plt.figure(figsize=(12, 8))
    plot_tree(fit1,
             feature_names=['x'],
             filled=True,
             rounded=True,
             fontsize=12)
    plt.title('Regression Tree (Single Split)', fontsize=14, fontweight='bold')
    plt.show()
    
    print(f"Tree depth: {fit1.get_depth()}")
    print(f"Number of leaves: {fit1.get_n_leaves()}")
    
    return fit1, dt




###################################################################################################################



# ===============================================
#                  chunk_tr29
# ===============================================

def chunk_tr29(dt, split_value=-0.65):
    """
    Calculate mean y values for data split at threshold.
    
    Args:
        dt: DataFrame with 'x' and 'y' columns
        split_value: Threshold value for splitting data
    
    Returns:
        tuple: (mean_left, mean_right)
    """
    # Calculate means for left and right splits
    mean_left = dt[dt['x'] <= split_value]['y'].mean()
    mean_right = dt[dt['x'] > split_value]['y'].mean()
    
    print(f"Mean y for x <= {split_value}: {mean_left:.4f}")
    print(f"Mean y for x > {split_value}: {mean_right:.4f}")
    
    return mean_left, mean_right



###################################################################################################################



# ===============================================
#                  chunk_tr30
# ===============================================

def chunk_tr30(dt, tree_model, split_value=-0.65):
    """
    Plot data with tree predictions and split line.
    
    Args:
        dt: DataFrame with 'x' and 'y' columns
        tree_model: Fitted decision tree model
        split_value: Split threshold to highlight with vertical line
    
    Returns:
        None
    """
    # Create sequence for smooth prediction line
    x_range = np.linspace(dt['x'].min(), dt['x'].max(), 1000)
    x_pred = x_range.reshape(-1, 1)
    
    # Get tree predictions
    y_pred = tree_model.predict(x_pred)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of original data
    plt.scatter(dt['x'], dt['y'], color='gray', alpha=0.7, label='Data')
    
    # Tree prediction line
    plt.plot(x_range, y_pred, color='blue', linewidth=3, label='Tree Predictions')
    
    # Vertical line at split point
    plt.axvline(x=split_value, color='red', linewidth=2, label=f'Split at x={split_value}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Tree Regression with Split Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Split occurs at x = {split_value}")
    print(f"Tree has {tree_model.get_n_leaves()} leaf nodes")




###################################################################################################################




# ===============================================
#                  chunk_tr31
# ===============================================

def chunk_tr31(dt, random_state=1):
    """
    Fit more complex decision tree and visualize predictions.
    
    Args:
        dt: DataFrame with 'x' and 'y' columns
        random_state: Random seed for reproducible results
    
    Returns:
        DecisionTreeRegressor: Fitted complex tree model
    """
    # Fit regression tree with lower minsplit (more complex tree)
    fit2 = DecisionTreeRegressor(
        min_samples_split=6,
        random_state=random_state
    )
    fit2.fit(dt[['x']], dt['y'])
    
    # Fancy tree plot
    plt.figure(figsize=(15, 10))
    plot_tree(fit2,
             feature_names=['x'],
             filled=True,
             rounded=True,
             fontsize=10)
    plt.title('Complex Regression Tree (Multiple Splits)', fontsize=14, fontweight='bold')
    plt.show()
    
    # Create prediction plot
    x_range = np.linspace(dt['x'].min(), dt['x'].max(), 1000)
    x_pred = x_range.reshape(-1, 1)
    y_pred = fit2.predict(x_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(dt['x'], dt['y'], color='gray', alpha=0.7, label='Data')
    plt.plot(x_range, y_pred, color='green', linewidth=3, label='Complex Tree Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Complex Decision Tree Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Complex tree depth: {fit2.get_depth()}")
    print(f"Number of leaves: {fit2.get_n_leaves()}")
    
    return fit2




###################################################################################################################




# ===============================================
#                  chunk_tr32
# ===============================================

def chunk_tr32():
    """
    Load Hitters dataset and display structure information.
    
    Returns:
        pandas.DataFrame: Hitters dataset
    """
    # Load Hitters dataset from seaborn (equivalent to ISLR::Hitters)
    hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv')
    
    # Display structure equivalent to str() in R
    print("Hitters Dataset Structure:")
    print(f"Shape: {hitters.shape}")
    print(f"Columns: {list(hitters.columns)}")
    print("\nData types:")
    print(hitters.dtypes)
    print(f"\nMemory usage: {hitters.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\nMissing values per column:")
    print(hitters.isnull().sum().sum())
    
    return hitters



###################################################################################################################




# ===================================
#           chunk_tr33
# ===================================

def chunk_tr33(hitters_data):
    """
    Convert R script tr33 to Python: builds decision tree for log(Salary) prediction
    
    Args:
        hitters_data: DataFrame with columns including 'Salary', 'Years', 'Hits', 'AtBat'
    
    Returns:
        tuple: (trained_tree_model, filtered_dataframe)
    """
    # Remove NA's (equivalent to complete.cases)
    df = hitters_data.dropna(subset=['Salary']).copy()
    
    # Select equivalent columns: Salary, Years, Hits, AtBat
    dfshort = df[['Salary', 'Years', 'Hits', 'AtBat']].copy()
    
    # Prepare features and target (log-transformed salary)
    X = dfshort[['Years', 'Hits', 'AtBat']]
    y = np.log(dfshort['Salary'])
    
    # Build fully grown tree (equivalent to cp=0 in R)
    tree = DecisionTreeRegressor(random_state=42, min_samples_split=2, min_samples_leaf=1)
    tree.fit(X, y)
    
    # Plot tree (equivalent to prp with extra=1)
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=['Years', 'Hits', 'AtBat'], 
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree: log(Salary) ~ Years + Hits + AtBat")
    plt.show()
    
    return tree, dfshort




###################################################################################################################



# ===================================
#           chunk_tr34
# ===================================

def chunk_tr34(dfshort):
    """
    Convert R script tr34 to Python: builds pruned decision tree for log(Salary) prediction
    
    Args:
        dfshort: DataFrame with columns 'Salary', 'Years', 'Hits', 'AtBat'
    
    Returns:
        DecisionTreeRegressor: trained pruned tree model
    """
    # Prepare features and target (log-transformed salary)
    X = dfshort[['Years', 'Hits', 'AtBat']]
    y = np.log(dfshort['Salary'])
    
    # Build pruned tree (default parameters provide automatic pruning)
    ptree = DecisionTreeRegressor(random_state=42)
    ptree.fit(X, y)
    
    # Plot tree (equivalent to prp with extra=1)
    plt.figure(figsize=(10, 8))
    plot_tree(ptree, feature_names=['Years', 'Hits', 'AtBat'], 
              filled=True, rounded=True, fontsize=12)
    plt.title("Pruned Decision Tree: log(Salary) ~ Years + Hits + AtBat")
    plt.show()
    
    return ptree



###################################################################################################################





# ===================================
#           chunk_tr35
# ===================================

def chunk_tr35(dfshort, random_state=123):
    """
    Convert R script tr35 to Python: compare decision tree vs linear regression performance
    
    Args:
        dfshort: DataFrame with columns 'Salary', 'Years', 'Hits', 'AtBat'
        random_state: seed for reproducible train/test split
    
    Returns:
        dict: RMSPE values for tree and linear models
    """
    # Prepare features and target
    X = dfshort[['Years', 'Hits', 'AtBat']]
    y = np.log(dfshort['Salary'])
    
    # Test/train split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=random_state
    )
    
    # Decision tree model
    ptree = DecisionTreeRegressor(random_state=42)
    ptree.fit(X_train, y_train)
    predtree = ptree.predict(X_test)
    
    # Linear regression model
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    predlin = lin.predict(X_test)
    
    # Calculate RMSPE (Root Mean Square Prediction Error)
    rmspe_tree = np.sqrt(mean_squared_error(y_test, predtree))
    rmspe_lin = np.sqrt(mean_squared_error(y_test, predlin))
    
    print(f"RMSPE Tree: {rmspe_tree:.4f}")
    print(f"RMSPE Linear: {rmspe_lin:.4f}")
    
    return {'rmspe_tree': rmspe_tree, 'rmspe_lin': rmspe_lin}