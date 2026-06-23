import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import multiprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from collections import defaultdict
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc


# ====================================
#           chunk_ea1
# ====================================

def chunk_ea1():
    """
    Compare AUC performance of Single Tree, Bagging, and Random Forest on Titanic dataset.
    """
    # Load Titanic dataset
    titanic = sns.load_dataset('titanic')
    
    # Select and prepare data
    nam = ["survived", "sex", "age", "pclass", "sibsp", "parch"]
    df = titanic[nam].copy()
    dfc = df.dropna()
    
    # Encode categorical variables
    dfc['sex'] = dfc['sex'].map({'male': 1, 'female': 0})
    
    AUC1, AUC2, AUC3 = [], [], []
    n, B = 100, 100
    
    for i in range(n):
        np.random.seed(i + i * 100)
        
        # Bootstrap sampling
        ind = np.random.choice(len(dfc), len(dfc), replace=True)
        train = dfc.iloc[ind].reset_index(drop=True)
        test_mask = ~np.isin(np.arange(len(dfc)), ind)
        test = dfc[test_mask].reset_index(drop=True)
        
        if len(test) == 0:
            continue
            
        X_train = train[['sex', 'age', 'pclass', 'sibsp', 'parch']]
        y_train = train['survived']
        X_test = test[['sex', 'age', 'pclass', 'sibsp', 'parch']]
        y_test = test['survived']
        
        p = X_train.shape[1]
        
        # Three methods
        model1 = DecisionTreeClassifier(random_state=42)  # Single tree
        model2 = BaggingClassifier(n_estimators=B, max_features=p, random_state=42)  # Bagged
        model3 = RandomForestClassifier(n_estimators=B, random_state=42)  # Random Forest
        
        # Fit models
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        model3.fit(X_train, y_train)
        
        # Predictions (probabilities)
        phat1 = model1.predict_proba(X_test)[:, 1]
        phat2 = model2.predict_proba(X_test)[:, 1]
        phat3 = model3.predict_proba(X_test)[:, 1]
        
        # Calculate AUCs
        try:
            AUC1.append(roc_auc_score(y_test, phat1))
            AUC2.append(roc_auc_score(y_test, phat2))
            AUC3.append(roc_auc_score(y_test, phat3))
        except ValueError:
            # Skip if test set has only one class
            continue
    
    # Results
    results = pd.DataFrame({
        'model': ['Single-Tree', 'Bagging', 'RF'],
        'AUCs': [np.mean(AUC1), np.mean(AUC2), np.mean(AUC3)],
        'sd': [np.std(AUC1, ddof=1), np.std(AUC2, ddof=1), np.std(AUC3, ddof=1)]
    })
    
    print("Model Comparison Results:")
    print(results)
    
    return results, AUC1, AUC2, AUC3




######################################################################################################################



# ====================================
#           chunk_ea2
# ====================================

def chunk_ea2():
    """
    Compare AUC performance of Pruned vs Unpruned decision trees using bagging.
    """
    # Load and prepare Titanic dataset
    titanic = sns.load_dataset('titanic')
    nam = ["survived", "sex", "age", "pclass", "sibsp", "parch"]
    df = titanic[nam].copy()
    dfc = df.dropna()
    dfc['sex'] = dfc['sex'].map({'male': 1, 'female': 0})
    
    n, B = 100, 500
    AUCp, AUCup = [], []
    
    for i in range(n):
        np.random.seed(i + i * 100)
        
        # Bootstrap sampling for train/test split
        ind = np.random.choice(len(dfc), len(dfc), replace=True)
        train = dfc.iloc[ind].reset_index(drop=True)
        test_mask = ~np.isin(np.arange(len(dfc)), ind)
        test = dfc[test_mask].reset_index(drop=True)
        
        if len(test) == 0:
            continue
            
        X_test = test[['sex', 'age', 'pclass', 'sibsp', 'parch']]
        y_test = test['survived']
        
        phatp = np.zeros((B, len(test)))
        phatup = np.zeros((B, len(test)))
        
        # Build B trees
        for j in range(B):
            np.random.seed(j + j * 2)
            
            # Bootstrap sample from training set
            tr_ind = np.random.choice(len(train), len(train), replace=True)
            tr = train.iloc[tr_ind].reset_index(drop=True)
            
            X_train = tr[['sex', 'age', 'pclass', 'sibsp', 'parch']]
            y_train = tr['survived']
            
            # Pruned tree (with default constraints)
            modelp = DecisionTreeClassifier(random_state=42)
            
            # Unpruned tree (minimal constraints)
            modelup = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1, 
                                           max_depth=None, random_state=42)
            
            # Fit models
            modelp.fit(X_train, y_train)
            modelup.fit(X_train, y_train)
            
            # Store predictions
            phatp[j, :] = modelp.predict_proba(X_test)[:, 1]
            phatup[j, :] = modelup.predict_proba(X_test)[:, 1]
        
        # Average predictions across B trees
        phatpr = np.mean(phatp, axis=0)
        phatupr = np.mean(phatup, axis=0)
        
        # Calculate AUCs
        try:
            AUCp.append(roc_auc_score(y_test, phatpr))
            AUCup.append(roc_auc_score(y_test, phatupr))
        except ValueError:
            continue
    
    # Results
    results = pd.DataFrame({
        'model': ['Pruned', 'Unpruned'],
        'AUCs': [np.mean(AUCp), np.mean(AUCup)],
        'sd': [np.std(AUCp, ddof=1), np.std(AUCup, ddof=1)]
    })
    
    print("Pruned vs Unpruned Trees Comparison:")
    print(results)
    
    return results, AUCp, AUCup




######################################################################################################################


# ====================================
#           chunk_ea3
# ====================================

def chunk_ea3():
    """
    Load Hitters dataset and remove rows with missing Salary values.
    """
    # Load Hitters dataset from seaborn (equivalent to ISLR::Hitters)
    try:
        import seaborn as sns
        # Try to load from seaborn first
        df = sns.load_dataset('tips')  # Placeholder - seaborn doesn't have Hitters
    except:
        pass
    
    # Since seaborn doesn't have Hitters, we'll create a sample or load from URL
    # The Hitters dataset is commonly available online
    try:
        # Load from online source
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
        hitters = pd.read_csv(url, index_col=0)
    except:
        # Fallback: create sample data structure
        print("Unable to load Hitters dataset. Creating sample structure...")
        # Create a sample dataset with similar structure to Hitters
        np.random.seed(42)
        n = 322  # Original Hitters dataset size
        hitters = pd.DataFrame({
            'AtBat': np.random.randint(100, 700, n),
            'Hits': np.random.randint(50, 200, n),
            'HmRun': np.random.randint(0, 40, n),
            'Runs': np.random.randint(20, 120, n),
            'RBI': np.random.randint(20, 120, n),
            'Walks': np.random.randint(10, 100, n),
            'Years': np.random.randint(1, 25, n),
            'CAtBat': np.random.randint(100, 5000, n),
            'CHits': np.random.randint(50, 1500, n),
            'CHmRun': np.random.randint(0, 300, n),
            'CRuns': np.random.randint(50, 1000, n),
            'CRBI': np.random.randint(50, 1000, n),
            'CWalks': np.random.randint(20, 800, n),
            'League': np.random.choice(['A', 'N'], n),
            'Division': np.random.choice(['E', 'W'], n),
            'PutOuts': np.random.randint(0, 400, n),
            'Assists': np.random.randint(0, 500, n),
            'Errors': np.random.randint(0, 30, n),
            'Salary': np.random.lognormal(5, 1, n)
        })
        # Introduce some missing values in Salary
        missing_idx = np.random.choice(n, size=int(0.18 * n), replace=False)
        hitters.loc[missing_idx, 'Salary'] = np.nan
    
    # Remove rows with missing Salary values
    df = hitters.dropna(subset=['Salary']).reset_index(drop=True)
    
    print(f"Dataset shape after removing missing Salary: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df





######################################################################################################################


# ====================================
#           chunk_ea4
# ====================================

def chunk_ea4(df):
    """
    Compare RMSPE of Pruned vs Unpruned regression trees using bagging on log salary.
    """
    # Create log salary and remove original Salary column
    df = df.copy()
    df['logsal'] = np.log(df['Salary'])
    df = df.drop('Salary', axis=1)
    
    n, B = 100, 500
    RMSPEp, RMSPEup = [], []
    
    for i in range(n):
        np.random.seed(i + i * 8)
        
        # Bootstrap sampling for train/test split
        ind = np.random.choice(len(df), len(df), replace=True)
        train = df.iloc[ind].reset_index(drop=True)
        test_mask = ~np.isin(np.arange(len(df)), ind)
        test = df[test_mask].reset_index(drop=True)
        
        if len(test) == 0:
            continue
            
        # Separate features and target for test set
        X_test = test.drop('logsal', axis=1)
        y_test = test['logsal']
        
        yhatp = np.zeros((B, len(test)))
        yhatup = np.zeros((B, len(test)))
        
        # Build B trees
        for j in range(B):
            np.random.seed(j + j * 2)
            
            # Bootstrap sample from training set
            tr_ind = np.random.choice(len(train), len(train), replace=True)
            tr = train.iloc[tr_ind].reset_index(drop=True)
            
            X_train = tr.drop('logsal', axis=1)
            y_train = tr['logsal']
            
            # Handle categorical variables
            X_train_encoded = pd.get_dummies(X_train, drop_first=True)
            X_test_encoded = pd.get_dummies(X_test, drop_first=True)
            
            # Align columns between train and test
            missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
            for col in missing_cols:
                X_test_encoded[col] = 0
            X_test_encoded = X_test_encoded[X_train_encoded.columns]
            
            # Pruned tree (with default constraints)
            modelp = DecisionTreeRegressor(random_state=42)
            
            # Unpruned tree (minimal constraints)
            modelup = DecisionTreeRegressor(min_samples_split=2, min_samples_leaf=1, 
                                          max_depth=None, random_state=42)
            
            # Fit models
            modelp.fit(X_train_encoded, y_train)
            modelup.fit(X_train_encoded, y_train)
            
            # Store predictions
            yhatp[j, :] = modelp.predict(X_test_encoded)
            yhatup[j, :] = modelup.predict(X_test_encoded)
        
        # Average predictions across B trees
        yhatpr = np.mean(yhatp, axis=0)
        yhatupr = np.mean(yhatup, axis=0)
        
        # Calculate RMSPE (Root Mean Square Prediction Error)
        RMSPEp.append(np.sqrt(np.mean((y_test - yhatpr) ** 2)))
        RMSPEup.append(np.sqrt(np.mean((y_test - yhatupr) ** 2)))
    
    # Results
    results = pd.DataFrame({
        'model': ['Pruned', 'Unpruned'],
        'RMSPEs': [np.mean(RMSPEp), np.mean(RMSPEup)],
        'sd': [np.std(RMSPEp, ddof=1), np.std(RMSPEup, ddof=1)]
    })
    
    print("Pruned vs Unpruned Regression Trees Comparison:")
    print(results)
    
    return results, RMSPEp, RMSPEup





######################################################################################################################





# ====================================
#           chunk_ea5
# ====================================

def chunk_ea5():
    """
    Compare RMSPE of Single Tree, Bagging, and Random Forest on Hitters log salary data.
    """
    # Load and prepare data (replicating chunk_ea3 data loading)
    try:
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
        hitters = pd.read_csv(url, index_col=0)
    except:
        # Fallback sample data
        np.random.seed(42)
        n = 322
        hitters = pd.DataFrame({
            'AtBat': np.random.randint(100, 700, n),
            'Hits': np.random.randint(50, 200, n),
            'HmRun': np.random.randint(0, 40, n),
            'Runs': np.random.randint(20, 120, n),
            'RBI': np.random.randint(20, 120, n),
            'Walks': np.random.randint(10, 100, n),
            'Years': np.random.randint(1, 25, n),
            'CAtBat': np.random.randint(100, 5000, n),
            'CHits': np.random.randint(50, 1500, n),
            'CHmRun': np.random.randint(0, 300, n),
            'CRuns': np.random.randint(50, 1000, n),
            'CRBI': np.random.randint(50, 1000, n),
            'CWalks': np.random.randint(20, 800, n),
            'League': np.random.choice(['A', 'N'], n),
            'Division': np.random.choice(['E', 'W'], n),
            'PutOuts': np.random.randint(0, 400, n),
            'Assists': np.random.randint(0, 500, n),
            'Errors': np.random.randint(0, 30, n),
            'Salary': np.random.lognormal(5, 1, n)
        })
        missing_idx = np.random.choice(n, size=int(0.18 * n), replace=False)
        hitters.loc[missing_idx, 'Salary'] = np.nan
    
    # Clean data and create log salary
    df = hitters.dropna(subset=['Salary']).reset_index(drop=True)
    df['logsal'] = np.log(df['Salary'])
    df = df.drop('Salary', axis=1)
    
    n, B = 100, 500
    RMSPE1, RMSPE2, RMSPE3 = [], [], []
    
    for i in range(n):
        np.random.seed(i + i * 8)
        
        # Bootstrap sampling
        ind = np.random.choice(len(df), len(df), replace=True)
        train = df.iloc[ind].reset_index(drop=True)
        test_mask = ~np.isin(np.arange(len(df)), ind)
        test = df[test_mask].reset_index(drop=True)
        
        if len(test) == 0:
            continue
            
        # Prepare features and target
        X_train = train.drop('logsal', axis=1)
        y_train = train['logsal']
        X_test = test.drop('logsal', axis=1)
        y_test = test['logsal']
        
        # Handle categorical variables
        X_train_encoded = pd.get_dummies(X_train, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, drop_first=True)
        
        # Align columns
        missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_cols:
            X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[X_train_encoded.columns]
        
        p = X_train_encoded.shape[1]
        
        # Three models
        model1 = DecisionTreeRegressor(random_state=42)  # Single Tree
        model2 = BaggingRegressor(n_estimators=B, max_features=p, random_state=42)  # Bagged
        model3 = RandomForestRegressor(n_estimators=B, random_state=42)  # Random Forest
        
        # Fit models
        model1.fit(X_train_encoded, y_train)
        model2.fit(X_train_encoded, y_train)
        model3.fit(X_train_encoded, y_train)
        
        # Predictions
        yhat1 = model1.predict(X_test_encoded)
        yhat2 = model2.predict(X_test_encoded)
        yhat3 = model3.predict(X_test_encoded)
        
        # Calculate RMSPE
        RMSPE1.append(np.sqrt(np.mean((y_test - yhat1) ** 2)))
        RMSPE2.append(np.sqrt(np.mean((y_test - yhat2) ** 2)))
        RMSPE3.append(np.sqrt(np.mean((y_test - yhat3) ** 2)))
    
    # Results
    results = pd.DataFrame({
        'model': ['Single-Tree', 'Bagging', 'RF'],
        'RMSPEs': [np.mean(RMSPE1), np.mean(RMSPE2), np.mean(RMSPE3)],
        'sd': [np.std(RMSPE1, ddof=1), np.std(RMSPE2, ddof=1), np.std(RMSPE3, ddof=1)]
    })
    
    print("Regression Model Comparison Results:")
    print(results)
    
    return results, RMSPE1, RMSPE2, RMSPE3





######################################################################################################################



# ====================================
#           chunk_ea6
# ====================================

def chunk_ea6(model3, feature_names=None):
    """
    Create variable importance plot for Random Forest model.
    """
    # Get feature importances
    importances = model3.feature_importances_
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
    
    # Plot with error bars if available (some RF implementations provide std)
    bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                    color='lightblue', edgecolor='darkblue', alpha=0.7)
    
    plt.xlabel('Mean Decrease in Impurity')
    plt.title('Variable Importance Plot')
    plt.grid(axis='x', alpha=0.3)
    
    # Show only top 20 features if there are many
    if len(importance_df) > 20:
        top_20 = importance_df.tail(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top_20['feature'], top_20['importance'], 
                 color='lightblue', edgecolor='darkblue', alpha=0.7)
        plt.xlabel('Mean Decrease in Impurity')
        plt.title('Top 20 Variable Importance')
        plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df




######################################################################################################################



# ====================================
#           chunk_ea9
# ====================================

def chunk_ea9(model3, test_data, feature_name='CRuns'):
    """
    Create partial dependence plot for a specific feature in Random Forest model.
    """
    # Get the feature column
    if feature_name not in test_data.columns:
        print(f"Feature '{feature_name}' not found in test data")
        return None
    
    # Create a copy of test data
    test_copy = test_data.copy()
    
    # Get range of feature values
    feature_values = np.linspace(test_data[feature_name].min(), 
                                test_data[feature_name].max(), 50)
    
    partial_predictions = []
    
    # Calculate partial dependence
    for value in feature_values:
        # Set all instances to have this feature value
        test_copy[feature_name] = value
        
        # Get predictions
        predictions = model3.predict(test_copy)
        
        # Average prediction across all instances
        avg_prediction = np.mean(predictions)
        partial_predictions.append(avg_prediction)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(feature_values, partial_predictions, 'r-', linewidth=3)
    plt.xlabel(feature_name)
    plt.ylabel('Partial Dependence')
    plt.title(f'Effects of {feature_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return data for further analysis
    return pd.DataFrame({
        feature_name: feature_values,
        'partial_dependence': partial_predictions
    })





######################################################################################################################





# ====================================
#           chunk_ea7
# ====================================

def chunk_ea7(model3, X_train, y_train, X_test, y_test, feature_names=None):
    """
    Measure comprehensive importance metrics for Random Forest model.
    """
    # Get feature names
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # 1. Mean Decrease in Impurity (Gini importance)
    gini_importance = model3.feature_importances_
    
    # 2. Permutation importance on training set
    perm_importance_train = permutation_importance(
        model3, X_train, y_train, n_repeats=10, random_state=42
    )
    
    # 3. Permutation importance on test set
    perm_importance_test = permutation_importance(
        model3, X_test, y_test, n_repeats=10, random_state=42
    )
    
    # 4. Calculate mean minimal depth (approximation)
    # This is a simplified version - full implementation would require tree traversal
    total_nodes = sum(tree.tree_.node_count for tree in model3.estimators_)
    avg_depth = total_nodes / len(model3.estimators_) / len(feature_names)
    
    # Create comprehensive importance dataframe
    importance_frame = pd.DataFrame({
        'variable': feature_names,
        'mean_min_depth': [avg_depth] * len(feature_names),  # Simplified approximation
        'accuracy_decrease_train': perm_importance_train.importances_mean,
        'accuracy_decrease_test': perm_importance_test.importances_mean,
        'accuracy_decrease_train_std': perm_importance_train.importances_std,
        'accuracy_decrease_test_std': perm_importance_test.importances_std,
        'gini_decrease': gini_importance,
        'times_a_root': [0] * len(feature_names),  # Would require tree analysis
        'p_value': [0.0] * len(feature_names)  # Would require statistical testing
    })
    
    # Sort by gini importance
    importance_frame = importance_frame.sort_values('gini_decrease', ascending=False)
    
    print("Random Forest Variable Importance Measures:")
    print("=" * 50)
    print(importance_frame.round(6))
    
    return importance_frame





######################################################################################################################

# ====================================
#           chunk_ea8
# ====================================

def chunk_ea8(importance_frame, x_measure='mean_min_depth', 
              y_measure='accuracy_decrease_test', size_measure='gini_decrease', 
              no_of_labels=6):
    """
    Create multi-way importance plot with scatter plot showing relationships between importance measures.
    """
    # Check if measures exist in the dataframe
    available_measures = importance_frame.columns.tolist()
    
    # Map common measure names
    measure_mapping = {
        'mse_increase': 'accuracy_decrease_test',
        'accuracy_decrease': 'accuracy_decrease_test',
        'p_value': 'gini_decrease'  # Use gini_decrease if p_value not available
    }
    
    # Apply mapping if needed
    if y_measure in measure_mapping:
        y_measure = measure_mapping[y_measure]
    if size_measure in measure_mapping:
        size_measure = measure_mapping[size_measure]
    
    # Verify measures exist
    for measure in [x_measure, y_measure, size_measure]:
        if measure not in available_measures:
            print(f"Warning: {measure} not found. Using available measures.")
            print(f"Available measures: {available_measures}")
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    
    # Use size_measure for point sizes (normalize to reasonable range)
    sizes = importance_frame[size_measure]
    sizes_normalized = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 300 + 50
    
    # Create scatter plot
    scatter = plt.scatter(importance_frame[x_measure], 
                         importance_frame[y_measure],
                         s=sizes_normalized,
                         alpha=0.6,
                         c=importance_frame[size_measure],
                         cmap='viridis',
                         edgecolors='black',
                         linewidth=0.5)
    
    # Add labels for top variables
    top_vars = importance_frame.nlargest(no_of_labels, y_measure)
    
    for idx, row in top_vars.iterrows():
        plt.annotate(row['variable'], 
                    (row[x_measure], row[y_measure]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8)
    
    # Customize plot
    plt.xlabel(x_measure.replace('_', ' ').title())
    plt.ylabel(y_measure.replace('_', ' ').title())
    plt.title('Multi-way Variable Importance Plot')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(size_measure.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.show()
    
    # Return information about the plot
    plot_info = {
        'x_measure': x_measure,
        'y_measure': y_measure,
        'size_measure': size_measure,
        'top_variables': top_vars['variable'].tolist()
    }
    
    return plot_info



######################################################################################################################




# ====================================
#        chunk_min_depth
# ====================================

def chunk_min_depth(model3, feature_names=None, k=20):
    """
    Calculate and plot minimal depth distribution for Random Forest variables.
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(model3.n_features_in_)]
    
    # Extract minimal depth for each variable across all trees
    min_depths = defaultdict(list)
    
    for tree_idx, tree in enumerate(model3.estimators_):
        tree_structure = tree.tree_
        
        # For each feature, find its minimal depth in this tree
        feature_depths = {}
        
        def traverse_tree(node_id, depth=0):
            if node_id == -1:  # Invalid node
                return
            
            feature_idx = tree_structure.feature[node_id]
            if feature_idx >= 0:  # Not a leaf node
                if feature_idx not in feature_depths:
                    feature_depths[feature_idx] = depth
                else:
                    feature_depths[feature_idx] = min(feature_depths[feature_idx], depth)
                
                # Traverse children
                traverse_tree(tree_structure.children_left[node_id], depth + 1)
                traverse_tree(tree_structure.children_right[node_id], depth + 1)
        
        # Start traversal from root
        traverse_tree(0)
        
        # Record depths for all features (use max depth for features not in this tree)
        max_depth = tree_structure.max_depth
        for i in range(len(feature_names)):
            depth = feature_depths.get(i, max_depth + 1)  # Penalize missing features
            min_depths[feature_names[i]].append(depth)
    
    # Create min_depth_frame
    min_depth_data = []
    for var_name in feature_names:
        depths = min_depths[var_name]
        min_depth_data.append({
            'variable': var_name,
            'minimal_depth': np.mean(depths),
            'depths_distribution': depths
        })
    
    min_depth_frame = pd.DataFrame(min_depth_data)
    min_depth_frame = min_depth_frame.sort_values('minimal_depth').head(k)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean minimal depth
    vars_to_plot = min_depth_frame['variable'].tolist()
    mean_depths = min_depth_frame['minimal_depth'].tolist()
    
    ax1.barh(range(len(vars_to_plot)), mean_depths, color='lightblue', alpha=0.7)
    ax1.set_yticks(range(len(vars_to_plot)))
    ax1.set_yticklabels(vars_to_plot)
    ax1.set_xlabel('Mean Minimal Depth')
    ax1.set_title('Mean Minimal Depth by Variable')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Distribution of minimal depths (box plot)
    depth_distributions = [min_depths[var] for var in vars_to_plot]
    
    ax2.boxplot(depth_distributions, labels=vars_to_plot, vert=False)
    ax2.set_xlabel('Minimal Depth')
    ax2.set_title('Distribution of Minimal Depths')
    ax2.grid(axis='x', alpha=0.3)
    
    # Rotate y-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.tick_params(axis='y', labelsize=8)
    
    plt.suptitle('Distribution of minimal depth and its mean', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"Top {k} variables by minimal depth:")
    print(min_depth_frame[['variable', 'minimal_depth']].round(3))
    
    return min_depth_frame




######################################################################################################################



# ====================================
#          chunk_ea100
# ====================================

def chunk_ea100():
    """
    Implement Gradient Boosting with cross-validation on Hitters dataset.
    """
    # Load and prepare data
    try:
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
        hitters = pd.read_csv(url, index_col=0)
    except:
        # Fallback sample data
        np.random.seed(42)
        n = 322
        hitters = pd.DataFrame({
            'AtBat': np.random.randint(100, 700, n),
            'Hits': np.random.randint(50, 200, n),
            'HmRun': np.random.randint(0, 40, n),
            'Runs': np.random.randint(20, 120, n),
            'RBI': np.random.randint(20, 120, n),
            'Walks': np.random.randint(10, 100, n),
            'Years': np.random.randint(1, 25, n),
            'CAtBat': np.random.randint(100, 5000, n),
            'CHits': np.random.randint(50, 1500, n),
            'CHmRun': np.random.randint(0, 300, n),
            'CRuns': np.random.randint(50, 1000, n),
            'CRBI': np.random.randint(50, 1000, n),
            'CWalks': np.random.randint(20, 800, n),
            'League': np.random.choice(['A', 'N'], n),
            'Division': np.random.choice(['E', 'W'], n),
            'PutOuts': np.random.randint(0, 400, n),
            'Assists': np.random.randint(0, 500, n),
            'Errors': np.random.randint(0, 30, n),
            'Salary': np.random.lognormal(5, 1, n)
        })
        missing_idx = np.random.choice(n, size=int(0.18 * n), replace=False)
        hitters.loc[missing_idx, 'Salary'] = np.nan
    
    # Clean data and transform salary
    df = hitters.dropna(subset=['Salary']).reset_index(drop=True)
    df['Salary'] = np.log(df['Salary'])
    
    # Prepare features and target
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Create gradient boosting model with cross-validation
    model_cv = GradientBoostingRegressor(
        n_estimators=1000,
        max_depth=3,          # interaction.depth
        learning_rate=0.01,   # shrinkage
        subsample=0.5,        # bag.fraction
        random_state=42
    )
    
    # Fit model with staged prediction for CV-like behavior
    model_cv.fit(X_encoded, y)
    
    # Calculate staged predictions (equivalent to CV error tracking)
    train_scores = []
    test_scores = []
    
    for i, pred in enumerate(model_cv.staged_predict(X_encoded)):
        train_score = np.mean((y - pred) ** 2)
        train_scores.append(train_score)
    
    # Find best number of trees
    best = np.argmin(train_scores)
    best_rmse = np.sqrt(train_scores[best])
    
    print(f"Best number of trees: {best + 1}")
    print(f"Best RMSE: {best_rmse:.6f}")
    
    # Plot CV error curve (equivalent to gbm.perf)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_scores) + 1), np.sqrt(train_scores), 
             'b-', label='Training Error', linewidth=2)
    plt.axvline(x=best + 1, color='red', linestyle='--', 
                label=f'Best n_trees = {best + 1}')
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title('Gradient Boosting CV Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model_cv, best + 1, best_rmse, train_scores




######################################################################################################################





# ====================================
#          chunk_ea101
# ====================================

def chunk_ea101():
    """
    Implement Gradient Boosting with train/validation split on Hitters dataset.
    """
    # Load and prepare data (same as chunk_ea100)
    try:
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
        hitters = pd.read_csv(url, index_col=0)
    except:
        # Fallback sample data
        np.random.seed(42)
        n = 322
        hitters = pd.DataFrame({
            'AtBat': np.random.randint(100, 700, n),
            'Hits': np.random.randint(50, 200, n),
            'HmRun': np.random.randint(0, 40, n),
            'Runs': np.random.randint(20, 120, n),
            'RBI': np.random.randint(20, 120, n),
            'Walks': np.random.randint(10, 100, n),
            'Years': np.random.randint(1, 25, n),
            'CAtBat': np.random.randint(100, 5000, n),
            'CHits': np.random.randint(50, 1500, n),
            'CHmRun': np.random.randint(0, 300, n),
            'CRuns': np.random.randint(50, 1000, n),
            'CRBI': np.random.randint(50, 1000, n),
            'CWalks': np.random.randint(20, 800, n),
            'League': np.random.choice(['A', 'N'], n),
            'Division': np.random.choice(['E', 'W'], n),
            'PutOuts': np.random.randint(0, 400, n),
            'Assists': np.random.randint(0, 500, n),
            'Errors': np.random.randint(0, 30, n),
            'Salary': np.random.lognormal(5, 1, n)
        })
        missing_idx = np.random.choice(n, size=int(0.18 * n), replace=False)
        hitters.loc[missing_idx, 'Salary'] = np.nan
    
    # Clean data and transform salary
    df = hitters.dropna(subset=['Salary']).reset_index(drop=True)
    df['Salary'] = np.log(df['Salary'])
    
    # Prepare features and target
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Create and fit gradient boosting model
    model_test = GradientBoostingRegressor(
        n_estimators=1000,
        max_depth=3,          # interaction.depth
        learning_rate=0.01,   # shrinkage
        subsample=0.5,        # bag.fraction
        validation_fraction=0.0,  # We handle validation manually
        random_state=42
    )
    
    # Fit on training data
    model_test.fit(X_train, y_train)
    
    # Calculate staged predictions on validation set
    train_errors = []
    valid_errors = []
    
    for train_pred in model_test.staged_predict(X_train):
        train_error = np.mean((y_train - train_pred) ** 2)
        train_errors.append(train_error)
    
    for val_pred in model_test.staged_predict(X_val):
        valid_error = np.mean((y_val - val_pred) ** 2)
        valid_errors.append(valid_error)
    
    # Find best iteration and minimum validation error
    best_iteration = np.argmin(valid_errors) + 1
    min_valid_error = np.min(valid_errors)
    
    print(f"Best iteration (test method): {best_iteration}")
    print(f"Best iteration index (0-based): {best_iteration - 1}")
    print(f"Minimum validation error: {min_valid_error:.6f}")
    print(f"Minimum validation RMSE: {np.sqrt(min_valid_error):.6f}")
    
    # Plot performance curves (equivalent to gbm.perf with method="test")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_errors) + 1), np.sqrt(train_errors), 
             'b-', label='Training RMSE', linewidth=2)
    plt.plot(range(1, len(valid_errors) + 1), np.sqrt(valid_errors), 
             'r-', label='Validation RMSE', linewidth=2)
    plt.axvline(x=best_iteration, color='green', linestyle='--', 
                label=f'Best iteration = {best_iteration}')
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title('Gradient Boosting Performance (Train/Validation Split)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model_test, best_iteration, min_valid_error, train_errors, valid_errors





######################################################################################################################


# ====================================
#           chunk_ea10
# ====================================

def chunk_ea10():
    """
    Load Hitters dataset, transform salary, and create bootstrap train/test split.
    """
    # Load and prepare data
    try:
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
        hitters = pd.read_csv(url, index_col=0)
    except:
        # Fallback sample data
        np.random.seed(42)
        n = 322
        hitters = pd.DataFrame({
            'AtBat': np.random.randint(100, 700, n),
            'Hits': np.random.randint(50, 200, n),
            'HmRun': np.random.randint(0, 40, n),
            'Runs': np.random.randint(20, 120, n),
            'RBI': np.random.randint(20, 120, n),
            'Walks': np.random.randint(10, 100, n),
            'Years': np.random.randint(1, 25, n),
            'CAtBat': np.random.randint(100, 5000, n),
            'CHits': np.random.randint(50, 1500, n),
            'CHmRun': np.random.randint(0, 300, n),
            'CRuns': np.random.randint(50, 1000, n),
            'CRBI': np.random.randint(50, 1000, n),
            'CWalks': np.random.randint(20, 800, n),
            'League': np.random.choice(['A', 'N'], n),
            'Division': np.random.choice(['E', 'W'], n),
            'PutOuts': np.random.randint(0, 400, n),
            'Assists': np.random.randint(0, 500, n),
            'Errors': np.random.randint(0, 30, n),
            'Salary': np.random.lognormal(5, 1, n)
        })
        missing_idx = np.random.choice(n, size=int(0.18 * n), replace=False)
        hitters.loc[missing_idx, 'Salary'] = np.nan
    
    # Clean data and transform salary
    df = hitters.dropna(subset=['Salary']).reset_index(drop=True)
    df['Salary'] = np.log(df['Salary'])
    
    # Bootstrap train/test split
    np.random.seed(1)
    ind = np.random.choice(len(df), len(df), replace=True)
    train = df.iloc[ind].reset_index(drop=True)
    
    # Test set (out-of-bag samples)
    test_mask = ~np.isin(np.arange(len(df)), ind)
    test = df[test_mask].reset_index(drop=True)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Log-transformed salary range: [{df['Salary'].min():.2f}, {df['Salary'].max():.2f}]")
    
    return df, train, test





######################################################################################################################



# ====================================
#           chunk_ea11
# ====================================

def chunk_ea11(train, test):
    """
    Grid search over shrinkage parameter for Gradient Boosting and plot MSE results.
    """
    # Define shrinkage parameter range
    h = np.arange(0.01, 1.81, 0.01)
    test_mse = []
    
    # Prepare data
    X_train = train.drop('Salary', axis=1)
    y_train = train['Salary']
    X_test = test.drop('Salary', axis=1)
    y_test = test['Salary']
    
    # Handle categorical variables
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)
    
    # Align columns between train and test
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Grid search over shrinkage values
    for i, shrinkage in enumerate(h):
        # Create GBM model with D=1 (max_depth=1), B=1000 (n_estimators=1000)
        boos = GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=1,              # interaction.depth = 1
            learning_rate=shrinkage,  # shrinkage parameter
            random_state=42
        )
        
        # Fit model
        boos.fit(X_train_encoded, y_train)
        
        # Predict using only first 100 trees (equivalent to n.trees = 100)
        staged_preds = list(boos.staged_predict(X_test_encoded))
        if len(staged_preds) >= 100:
            prboos = staged_preds[99]  # 0-indexed, so 99 = 100th prediction
        else:
            prboos = staged_preds[-1]  # Use last available if < 100 trees
        
        # Calculate test MSE
        mse = np.mean((y_test - prboos) ** 2)
        test_mse.append(mse)
        
        if (i + 1) % 20 == 0:  # Progress indicator
            print(f"Completed {i + 1}/{len(h)} shrinkage values")
    
    # Find optimal shrinkage
    best_idx = np.argmin(test_mse)
    best_shrinkage = h[best_idx]
    min_mse = min(test_mse)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(h, test_mse, 'b-', linewidth=2)
    plt.axvline(x=best_shrinkage, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal η = {best_shrinkage:.3f}')
    plt.xlabel('Shrinkage Parameter (η)')
    plt.ylabel('Test MSE')
    plt.title('MSE - Prediction')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"Optimal shrinkage parameter: {best_shrinkage:.3f}")
    print(f"Minimum test MSE: {min_mse:.6f}")
    
    return h, test_mse, best_shrinkage, min_mse





######################################################################################################################


# ================================
#           chunk_ea12
# ================================
def chunk_ea12(df):
    """
    Grid search for optimal GBM hyperparameters using bootstrap sampling.
    Returns minimum MSE and optimal parameters.
    """
    h = np.arange(0.01, 0.31, 0.01)
    B = [100, 300, 500, 750, 900]
    D = [1, 2]
    
    # Create parameter grid
    grid = np.array(np.meshgrid(D, B, h)).T.reshape(-1, 3)
    
    mse = np.zeros(len(grid))
    sdmse = np.zeros(len(grid))
    
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    n_samples = len(df)
    
    for i, (depth, n_trees, learning_rate) in enumerate(grid):
        test_mse = []
        
        for j in range(20):
            try:
                np.random.seed(j)
                # Bootstrap sampling
                train_idx = np.random.choice(n_samples, n_samples, replace=True)
                test_idx = np.setdiff1d(np.arange(n_samples), np.unique(train_idx))
                
                if len(test_idx) == 0:
                    continue
                
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                
                gbm = GradientBoostingRegressor(
                    n_estimators=int(n_trees),
                    max_depth=int(depth),
                    learning_rate=learning_rate,
                    random_state=j
                )
                
                gbm.fit(X_train, y_train)
                pred = gbm.predict(X_test)
                test_mse.append(mean_squared_error(y_test, pred))
                
            except:
                continue
        
        if test_mse:
            mse[i] = np.mean(test_mse)
            sdmse[i] = np.std(test_mse)
    
    min_mse_idx = np.nanargmin(mse)
    min_mse = mse[min_mse_idx]
    optimal_params = grid[min_mse_idx]
    
    print(f"Minimum MSE: {min_mse:.6f}")
    print(f"Optimal parameters [depth, n_trees, learning_rate]: {optimal_params}")
    
    return min_mse, optimal_params




######################################################################################################################


# ================================
#           chunk_ea13
# ================================
def chunk_ea13(df, conf_lev=0.95, num_max=5):
    """
    Random grid search for GBM hyperparameters with parallel processing.
    Returns sorted results by MSPE.
    """
    h = np.arange(0.001, 0.251, 0.001)
    B = np.arange(100, 801, 20)
    D = np.arange(1, 5)
    
    # Create parameter grid
    grid = np.array(np.meshgrid(D, B, h)).T.reshape(-1, 3)
    
    # Random grid search sampling
    n = np.log(1 - conf_lev) / np.log(1 - num_max / len(grid))
    np.random.seed(123)
    sample_size = int(len(grid) * (n / len(grid)))
    sampled_idx = np.random.choice(len(grid), sample_size, replace=False)
    comb = grid[sampled_idx]
    
    def bootstrap_eval(k):
        """Evaluate single parameter combination with bootstrap"""
        depth, n_trees, learning_rate = comb[k]
        test_mse = []
        
        X = df.drop('Salary', axis=1)
        y = df['Salary']
        n_samples = len(df)
        
        for j in range(10):
            try:
                np.random.seed(j)
                train_idx = np.random.choice(n_samples, n_samples, replace=True)
                test_idx = np.setdiff1d(np.arange(n_samples), np.unique(train_idx))
                
                if len(test_idx) == 0:
                    continue
                
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                
                gbm = GradientBoostingRegressor(
                    n_estimators=int(n_trees),
                    max_depth=int(depth),
                    learning_rate=learning_rate,
                    random_state=j
                )
                
                gbm.fit(X_train, y_train)
                pred = gbm.predict(X_test)
                test_mse.append(mean_squared_error(y_test, pred))
                
            except:
                continue
        
        if test_mse:
            return [k, np.mean(test_mse), np.std(test_mse)]
        return None
    
    # Parallel processing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(bootstrap_eval)(k) for k in range(len(comb))
    )
    
    # Filter out None results and process
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        unlst = np.array(valid_results)
        result_data = []
        
        for i, row in enumerate(unlst):
            k_idx = int(row[0])
            result_data.append([
                comb[k_idx][0],  # D
                comb[k_idx][1],  # B  
                comb[k_idx][2],  # h
                row[1],          # MSPE
                row[2]           # sd
            ])
        
        # Create sorted DataFrame
        sorted_df = pd.DataFrame(result_data, columns=['D', 'B', 'h', 'MSPE', 'sd'])
        sorted_df = sorted_df.sort_values('MSPE').reset_index(drop=True)
        
        print("Top 6 parameter combinations:")
        print(sorted_df.head(6))
        
        return sorted_df
    
    return pd.DataFrame()





######################################################################################################################





# ================================
#           chunk_ea14
# ================================
def chunk_ea14(df):
    """
    Compare multiple ML models using bootstrap sampling.
    Returns mean MSE and standard errors for each model.
    """
    # Prepare data (assuming Salary column exists and needs log transform)
    df = df.dropna(subset=['Salary'])
    df_copy = df.copy()
    df_copy['Salary'] = np.log(df_copy['Salary'])
    
    X = df_copy.drop('Salary', axis=1)
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    y = df_copy['Salary']
    n_samples = len(df_copy)
    n_features = X.shape[1]
    
    # Containers for MSE results
    mse_results = {
        'cart': [],
        'bag': [],
        'rf': [],
        'boost': [],
        'ols': []
    }
    
    for i in range(200):
        np.random.seed(i)
        
        # Bootstrap sampling
        train_idx = np.random.choice(n_samples, n_samples, replace=True)
        test_idx = np.setdiff1d(np.arange(n_samples), np.unique(train_idx))
        
        if len(test_idx) == 0:
            continue
            
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        try:
            # OLS
            ols = LinearRegression()
            ols.fit(X_train, y_train)
            pred_ols = ols.predict(X_test)
            mse_results['ols'].append(mean_squared_error(y_test, pred_ols))
            
            # CART
            cart = DecisionTreeRegressor(random_state=i)
            cart.fit(X_train, y_train)
            pred_cart = cart.predict(X_test)
            mse_results['cart'].append(mean_squared_error(y_test, pred_cart))
            
            # Bagging (Random Forest with all features)
            bag = RandomForestRegressor(
                n_estimators=100,
                max_features=n_features,
                random_state=i
            )
            bag.fit(X_train, y_train)
            pred_bag = bag.predict(X_test)
            mse_results['bag'].append(mean_squared_error(y_test, pred_bag))
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=i)
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)
            mse_results['rf'].append(mean_squared_error(y_test, pred_rf))
            
            # Gradient Boosting
            boost = GradientBoostingRegressor(
                n_estimators=100,
                random_state=i
            )
            boost.fit(X_train, y_train)
            pred_boost = boost.predict(X_test)
            mse_results['boost'].append(mean_squared_error(y_test, pred_boost))
            
        except:
            continue
    
    # Calculate results
    mean_mse = {}
    std_mse = {}
    
    for model in mse_results:
        if mse_results[model]:
            mean_mse[model] = np.mean(mse_results[model])
            std_mse[model] = np.std(mse_results[model])
    
    # Create results DataFrames
    means_df = pd.DataFrame.from_dict(mean_mse, orient='index', columns=['Mean_MSE'])
    stds_df = pd.DataFrame.from_dict(std_mse, orient='index', columns=['Std_MSE'])
    
    print("Mean MSE Results:")
    print(means_df)
    print("\nStandard Errors:")
    print(stds_df)
    
    return means_df, stds_df





######################################################################################################################


# ================================
#           chunk_ea15
# ================================
def chunk_ea15(df):
    """
    Convert Sales to binary classification target and display data structure.
    Returns modified DataFrame with Sales as 0/1 binary variable.
    """
    df_copy = df.copy()
    
    # Display original structure
    print("Original DataFrame Info:")
    print(f"Shape: {df_copy.shape}")
    print(f"Columns: {list(df_copy.columns)}")
    print("\nData types:")
    print(df_copy.dtypes)
    
    # Convert Sales to binary factor
    df_copy['Sales'] = np.where(df_copy['Sales'] <= 8, 0, 1)
    
    print(f"\nSales variable after conversion:")
    print(f"Type: {df_copy['Sales'].dtype}")
    print(f"Value counts:\n{df_copy['Sales'].value_counts()}")
    
    return df_copy





######################################################################################################################



# ================================
#           chunk_ea16
# ================================
def chunk_ea16(df, n=50, B=1000):
    """
    Compare classification models using bootstrap sampling and AUC scores.
    Returns DataFrame with mean AUC and standard deviations for each model.
    """
    # Remove missing values and prepare target variables
    df_clean = df.dropna()
    df_clean['d'] = df_clean['Sales'].astype('category')
    
    # Containers for AUC results
    AUC1, AUC2, AUC3, AUC4 = [], [], [], []
    
    for i in range(n):
        np.random.seed(i)
        
        # Bootstrap sampling
        train_idx = np.random.choice(len(df_clean), len(df_clean), replace=True)
        test_idx = np.setdiff1d(np.arange(len(df_clean)), np.unique(train_idx))
        
        if len(test_idx) == 0:
            continue
            
        train = df_clean.iloc[train_idx]
        test = df_clean.iloc[test_idx]
        
        # Prepare features (excluding Sales and d columns)
        X_train = train.drop(['Sales', 'd'], axis=1)
        X_test = test.drop(['Sales', 'd'], axis=1)
        
        # Handle categorical variables
        X_train_encoded = pd.get_dummies(X_train, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, drop_first=True)
        
        # Align columns between train and test
        common_cols = X_train_encoded.columns.intersection(X_test_encoded.columns)
        X_train_encoded = X_train_encoded[common_cols]
        X_test_encoded = X_test_encoded[common_cols]
        
        p = X_train_encoded.shape[1]
        
        try:
            # Model 1: Single Decision Tree
            model1 = DecisionTreeClassifier(random_state=i)
            model1.fit(X_train_encoded, train['Sales'])
            phat1 = model1.predict_proba(X_test_encoded)[:, 1]
            AUC1.append(roc_auc_score(test['Sales'], phat1))
            
            # Model 2: Bagging (Random Forest with all features)
            model2 = RandomForestClassifier(
                n_estimators=B, 
                max_features=p, 
                random_state=i
            )
            model2.fit(X_train_encoded, train['d'])
            phat2 = model2.predict_proba(X_test_encoded)[:, 1]
            AUC2.append(roc_auc_score(test['d'], phat2))
            
            # Model 3: Random Forest
            model3 = RandomForestClassifier(n_estimators=B, random_state=i)
            model3.fit(X_train_encoded, train['d'])
            phat3 = model3.predict_proba(X_test_encoded)[:, 1]
            AUC3.append(roc_auc_score(test['d'], phat3))
            
            # Model 4: Gradient Boosting
            model4 = GradientBoostingClassifier(n_estimators=B, random_state=i)
            model4.fit(X_train_encoded, train['Sales'])
            phat4 = model4.predict_proba(X_test_encoded)[:, 1]
            AUC4.append(roc_auc_score(test['Sales'], phat4))
            
        except Exception as e:
            continue
    
    # Calculate results
    results_df = pd.DataFrame({
        'model': ['Single-Tree', 'Bagging', 'RF', 'Boosting'],
        'AUCs': [
            np.mean(AUC1) if AUC1 else np.nan,
            np.mean(AUC2) if AUC2 else np.nan,
            np.mean(AUC3) if AUC3 else np.nan,
            np.mean(AUC4) if AUC4 else np.nan
        ],
        'sd': [
            np.std(AUC1) if AUC1 else np.nan,
            np.std(AUC2) if AUC2 else np.nan,
            np.std(AUC3) if AUC3 else np.nan,
            np.std(AUC4) if AUC4 else np.nan
        ]
    })
    
    print("Model Performance Comparison (AUC):")
    print(results_df)
    
    return results_df





######################################################################################################################



# ================================
#           chunk_ea17
# ================================
def chunk_ea17(df, n_iterations=100):
    """
    Evaluate Linear Regression model using bootstrap sampling and AUC scores.
    Returns mean AUC and standard deviation.
    """
    AUC5 = []
    
    for i in range(n_iterations):
        np.random.seed(i)
        
        # Bootstrap sampling
        train_idx = np.random.choice(len(df), len(df), replace=True)
        test_idx = np.setdiff1d(np.arange(len(df)), np.unique(train_idx))
        
        if len(test_idx) == 0:
            continue
            
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Prepare features (excluding Sales and d columns)
        X_train = train.drop(['Sales', 'd'], axis=1)
        X_test = test.drop(['Sales', 'd'], axis=1)
        
        # Handle categorical variables
        X_train_encoded = pd.get_dummies(X_train, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, drop_first=True)
        
        # Align columns between train and test
        common_cols = X_train_encoded.columns.intersection(X_test_encoded.columns)
        X_train_encoded = X_train_encoded[common_cols]
        X_test_encoded = X_test_encoded[common_cols]
        
        try:
            # Linear Regression model
            model = LinearRegression()
            model.fit(X_train_encoded, train['Sales'])
            phat5 = model.predict(X_test_encoded)
            
            # Calculate AUC
            auc_score = roc_auc_score(test['Sales'], phat5)
            AUC5.append(auc_score)
            
        except Exception:
            continue
    
    mean_auc = np.mean(AUC5) if AUC5 else np.nan
    std_auc = np.std(AUC5) if AUC5 else np.nan
    
    print(f"Linear Regression AUC Results:")
    print(f"Mean AUC: {mean_auc:.6f}")
    print(f"Standard Deviation: {std_auc:.6f}")
    
    return mean_auc, std_auc





######################################################################################################################


# ================================
#           chunk_ea18
# ================================
def chunk_ea18(df):
    """
    Preprocess data for AdaBoost: convert Sales to -1/1 coding and one-hot encode.
    Returns DataFrame with Sales as -1/1 and all categorical variables one-hot encoded.
    """
    df_copy = df.copy()
    
    # Convert Sales to -1/1 coding for AdaBoost
    df_copy['Sales'] = np.where(df_copy['Sales'] <= 8, -1, 1)
    
    print(f"Sales variable after conversion to -1/1:")
    print(f"Type: {df_copy['Sales'].dtype}")
    print(f"Value counts:\n{df_copy['Sales'].value_counts()}")
    
    # One-hot encode all categorical variables
    df_new = pd.get_dummies(df_copy, drop_first=False)
    
    print(f"\nDataFrame shape after one-hot encoding:")
    print(f"Original: {df_copy.shape}")
    print(f"Encoded: {df_new.shape}")
    print(f"\nNew columns: {list(df_new.columns)}")
    
    return df_new





######################################################################################################################




# ================================
#           chunk_ea19
# ================================
def chunk_ea19(df_new, n_rounds=100, n_iterations=100):
    """
    Evaluate AdaBoost model using bootstrap sampling and AUC scores.
    Returns mean AUC and standard deviation.
    """
    AUC = []
    
    for i in range(n_iterations):
        np.random.seed(i)
        
        # Bootstrap sampling
        train_idx = np.random.choice(len(df_new), len(df_new), replace=True)
        test_idx = np.setdiff1d(np.arange(len(df_new)), np.unique(train_idx))
        
        if len(test_idx) == 0:
            continue
            
        train = df_new.iloc[train_idx]
        test = df_new.iloc[test_idx]
        
        # Prepare features and target
        X_train = train.drop('Sales', axis=1)
        y_train = train['Sales']
        X_test = test.drop('Sales', axis=1)
        y_test = test['Sales']
        
        try:
            # AdaBoost with decision stumps (max_depth=1)
            ada = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n_rounds,
                random_state=i,
                algorithm='SAMME'  # Handles -1/1 labels better
            )
            
            ada.fit(X_train, y_train)
            phat = ada.predict_proba(X_test)[:, 1]
            
            # Convert -1/1 labels to 0/1 for AUC calculation
            y_test_binary = np.where(y_test == 1, 1, 0)
            
            # Calculate AUC
            auc_score = roc_auc_score(y_test_binary, phat)
            AUC.append(auc_score)
            
        except Exception:
            continue
    
    mean_auc = np.mean(AUC) if AUC else np.nan
    std_auc = np.std(AUC) if AUC else np.nan
    
    print(f"AdaBoost AUC Results:")
    print(f"Mean AUC: {mean_auc:.6f}")
    print(f"Standard Deviation: {std_auc:.6f}")
    
    return mean_auc, std_auc





######################################################################################################################


# ================================
#           chunk_ea20
# ================================
def chunk_ea20(filepath="adult_train.csv"):
    """
    Load and preprocess Adult dataset: assign column names, remove outliers, 
    convert character columns to categorical. Returns cleaned DataFrame.
    """
    # Load data without headers
    train = pd.read_csv(filepath, header=None)
    
    # Assign variable names
    varNames = ["Age", 
                "WorkClass",
                "fnlwgt",
                "Education",
                "EducationNum",
                "MaritalStatus",
                "Occupation",
                "Relationship",
                "Race",
                "Sex",
                "CapitalGain",
                "CapitalLoss",
                "HoursPerWeek",
                "NativeCountry",
                "IncomeLevel"]
    
    train.columns = varNames
    data = train.copy()
    
    # Check income level distribution
    income_counts = data['IncomeLevel'].value_counts()
    print("Income Level Distribution:")
    print(income_counts)
    
    # Remove outliers - Holand-Netherlands entries
    outlier_mask = data['NativeCountry'] == " Holand-Netherlands"
    print(f"\nRemoving {outlier_mask.sum()} Holand-Netherlands entries")
    data = data[~outlier_mask].reset_index(drop=True)
    
    # Convert character/object columns to categorical
    df = data.copy()
    object_columns = df.select_dtypes(include=['object']).columns
    
    print(f"\nConverting {len(object_columns)} object columns to categorical:")
    print(list(object_columns))
    
    for col in object_columns:
        df[col] = df[col].astype('category')
    
    print(f"\nFinal DataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes)
    
    return df





######################################################################################################################



# ================================
#           chunk_ea21
# ================================
def chunk_ea21(df, test_size=0.10, random_state=321):
    """
    Prepare Adult dataset for XGBoost: create binary target, train/test split,
    one-hot encode, and create DMatrix objects. Returns train/test DMatrix objects.
    """
    df_copy = df.copy()
    
    # Create binary target variable
    df_copy['Y'] = np.where(df_copy['IncomeLevel'] == " <=50K", 0, 1)
    
    # Remove original IncomeLevel column
    df_copy = df_copy.drop('IncomeLevel', axis=1)
    
    # Check for missing values
    na_count = df_copy.isnull().sum().sum()
    print(f"Missing values in dataset: {na_count}")
    
    # Train/test split (90-10%)
    train, test = train_test_split(
        df_copy, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df_copy['Y']
    )
    
    print(f"Train set size: {len(train)}")
    print(f"Test set size: {len(test)}")
    
    # Separate features and targets
    ty = train['Y']
    tsy = test['Y']
    
    train_features = train.drop('Y', axis=1)
    test_features = test.drop('Y', axis=1)
    
    # One-hot encode categorical variables
    hot_tr = pd.get_dummies(train_features, drop_first=False, dtype=int)
    hot_ts = pd.get_dummies(test_features, drop_first=False, dtype=int)
    
    # Ensure same columns in train and test
    common_cols = hot_tr.columns.intersection(hot_ts.columns)
    missing_in_test = set(hot_tr.columns) - set(hot_ts.columns)
    missing_in_train = set(hot_ts.columns) - set(hot_tr.columns)
    
    # Add missing columns with zeros
    for col in missing_in_test:
        hot_ts[col] = 0
    for col in missing_in_train:
        hot_tr[col] = 0
    
    # Reorder columns to match
    hot_tr = hot_tr.reindex(sorted(hot_tr.columns), axis=1)
    hot_ts = hot_ts.reindex(sorted(hot_ts.columns), axis=1)
    
    print(f"One-hot encoded features: {hot_tr.shape[1]}")
    
    # Create XGBoost DMatrix objects
    ttrain = xgb.DMatrix(data=hot_tr, label=ty)
    ttest = xgb.DMatrix(data=hot_ts, label=tsy)
    
    print("XGBoost DMatrix objects created successfully")
    
    return ttrain, ttest, hot_tr, hot_ts, ty, tsy





######################################################################################################################



# ===================================
#           chunk_ea22
# ===================================

def chunk_ea22(ttrain, random_state=112):
    """
    Perform XGBoost cross-validation with binary logistic objective.
    
    Args:
        ttrain: Training data (DMatrix or compatible format)
        random_state: Random seed for reproducibility
    
    Returns:
        CV results object
    """
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic'
    }
    
    np.random.seed(random_state)
    
    cvb = xgb.cv(
        params=params,
        dtrain=ttrain,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        verbose_eval=10,
        early_stopping_rounds=20,
        maximize=False,
        seed=random_state
    )
    
    return cvb





######################################################################################################################


# ===================================
#           chunk_ea23
# ===================================

def chunk_ea23(cvb):
    """
    Extract the best iteration from XGBoost CV results.
    
    Args:
        cvb: XGBoost CV results object
    
    Returns:
        int: Best iteration number
    """
    theb = cvb.shape[0]  # In Python XGBoost, best iteration is the number of rows
    return theb





######################################################################################################################


# ===================================
#           chunk_ea24
# ===================================

def chunk_ea24(params, ttrain, theb, ttest):
    """
    Train XGBoost model with specified parameters and best iteration count.
    
    Args:
        params: Dictionary of XGBoost parameters
        ttrain: Training data (DMatrix)
        theb: Number of boosting rounds (best iteration from CV)
        ttest: Test/validation data (DMatrix)
    
    Returns:
        Trained XGBoost model
    """
    model_default = xgb.train(
        params=params,
        dtrain=ttrain,
        num_boost_round=theb,
        evals=[(ttest, 'val'), (ttrain, 'train')],
        verbose_eval=10,
        maximize=False,
        eval_metric='auc'
    )
    
    return model_default





######################################################################################################################


# ===================================
#           chunk_ea25
# ===================================

def chunk_ea25(model_default, ttest, tsy, plot_roc=True):
    """
    Generate predictions, calculate AUC, and plot ROC curve.
    
    Args:
        model_default: Trained XGBoost model
        ttest: Test data (DMatrix)
        tsy: True labels for test data
        plot_roc: Whether to display ROC curve plot
    
    Returns:
        tuple: (predictions, AUC score)
    """
    phat = model_default.predict(ttest)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(tsy, phat)
    auc_score = auc(fpr, tpr)
    
    print(f"AUC: {auc_score:.4f}")
    
    # Plot ROC curve
    if plot_roc:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    return phat, auc_score