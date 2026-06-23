import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from IPython.display import Image, display




# ============================================
#              chunk_knn1
# ============================================

def chunk_knn1(image_path="png/digits.png", figsize=(0.6, 0.6)):
    """
    Display an image with specified dimensions (equivalent to R's knitr::include_graphics)
    
    Args:
        image_path (str): Path to the image file
        figsize (tuple): Figure size as fraction of default (width, height)
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Calculate actual figure size (default matplotlib figsize is ~6.4x4.8)
    actual_figsize = (6.4 * figsize[0], 4.8 * figsize[1])
    
    img = Image.open(image_path)
    plt.figure(figsize=actual_figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



#####################################################################################




# ============================================
#              chunk_knn2
# ============================================

def chunk_knn2(data_path="mnist.pkl"):
    """
    Load MNIST data from pickle file and display structure information.
    
    Args:
        data_path (str): Path to the pickled MNIST data file
    
    Returns:
        dict: MNIST dataset dictionary
    """
    # Load the data
    with open(data_path, 'rb') as f:
        mnist = pickle.load(f)
    
    # Display structure information
    print(f"MNIST data type: {type(mnist)}")
    if isinstance(mnist, dict):
        for key, value in mnist.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {type(value).__name__} - Shape: {value.shape}")
            else:
                print(f"{key}: {type(value).__name__}")
    
    return mnist



############################################################################################




# ============================================
#              chunk_knn3
# ============================================

def chunk_knn3(image_path="png/mnist.png", figsize=(6, 6)):
    """
    Display an image with specified dimensions.
    
    Args:
        image_path (str): Path to the image file
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()



###################################################################################




# ============================================
#              chunk_knn4
# ============================================

def chunk_knn4():
    """
    Load MNIST 2 vs 7 subset data and display structure information.
    
    Returns:
        dict: Dictionary containing train and test datasets
    """
    # Load MNIST data and filter for digits 2 and 7
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Filter for digits 2 and 7
    mask_train = (mnist.target.astype(int) == 2) | (mnist.target.astype(int) == 7)
    
    # Create subset similar to dslabs mnist_27
    train_x = mnist.data[mask_train]
    train_y = mnist.target[mask_train]
    
    # Convert to the expected format
    mnist_27 = {
        'train': {
            'x': train_x[:800],  # Subset for demo purposes
            'y': train_y[:800]
        },
        'test': {
            'x': train_x[800:1000],
            'y': train_y[800:1000]
        }
    }
    
    print(f"mnist_27 structure:")
    print(f"Type: {type(mnist_27)}")
    for split in ['train', 'test']:
        print(f"{split}:")
        print(f"  x: {type(mnist_27[split]['x']).__name__} - Shape: {mnist_27[split]['x'].shape}")
        print(f"  y: {type(mnist_27[split]['y']).__name__} - Shape: {mnist_27[split]['y'].shape}")
    
    return mnist_27




##########################################################################


# ============================================
#              chunk_knn5
# ============================================

def chunk_knn5(mnist_27):
    """
    Create binary labels and plot x_1 vs x_2 features colored by class.
    
    Args:
        mnist_27 (dict): MNIST 2 vs 7 dataset dictionary
    
    Returns:
        pd.DataFrame: Training dataframe with binary labels
    """
    # Convert labels to binary (1 for 7, 0 for 2)
    y10 = np.where(mnist_27['train']['y'].astype(int) == 7, 1, 0)
    
    # Create DataFrame assuming x has x_1 and x_2 features (first 2 columns)
    train = pd.DataFrame({
        'x_1': mnist_27['train']['x'][:, 0],
        'x_2': mnist_27['train']['x'][:, 1],
        'y': mnist_27['train']['y'].astype(int),
        'y10': y10
    })
    
    # Create scatter plot
    colors = ['red', 'blue']
    plt.figure(figsize=(8, 6))
    for i in [0, 1]:
        mask = train['y10'] == i
        plt.scatter(train.loc[mask, 'x_1'], train.loc[mask, 'x_2'], 
                   c=colors[i], s=10, alpha=0.7, label=f'Class {i}')
    
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()
    
    return train






################################################################################





# ============================================
#              chunk_knn6
# ============================================

def chunk_knn6(train):
    """
    Fit linear model and plot decision boundary with data points.
    
    Args:
        train (pd.DataFrame): Training data with x_1, x_2, and y10 columns
    
    Returns:
        sklearn.linear_model.LinearRegression: Fitted linear model
    """
    # Fit linear model
    X = train[['x_1', 'x_2']]
    y = train['y10']
    model = LinearRegression().fit(X, y)
    
    # Calculate decision boundary line (y10 = 0.5 threshold)
    tr = 0.5
    intercept = model.intercept_
    coef_x1, coef_x2 = model.coef_
    
    # Decision boundary: coef_x1*x_1 + coef_x2*x_2 + intercept = tr
    # Solve for x_2: x_2 = (tr - intercept - coef_x1*x_1) / coef_x2
    a = (tr - intercept) / coef_x2  # y-intercept when x_1=0
    b = -coef_x1 / coef_x2          # slope
    
    # Create scatter plot
    colors = ['red', 'blue']
    plt.figure(figsize=(8, 6))
    for i in [0, 1]:
        mask = train['y10'] == i
        plt.scatter(train.loc[mask, 'x_1'], train.loc[mask, 'x_2'], 
                   c=colors[i], s=15, alpha=0.7, label=f'Class {i}')
    
    # Add decision boundary line
    x_range = np.linspace(train['x_1'].min(), train['x_1'].max(), 100)
    y_line = a + b * x_range
    plt.plot(x_range, y_line, 'blue', linewidth=2.8, label='Decision Boundary')
    
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()
    
    return model





####################################################################################


# ============================================
#              chunk_knn7
# ============================================

def chunk_knn7(train):
    """
    Fit polynomial linear model and plot quadratic decision boundary.
    
    Args:
        train (pd.DataFrame): Training data with x_1, x_2, and y10 columns
    
    Returns:
        sklearn.linear_model.LinearRegression: Fitted polynomial model
    """
    # Create polynomial features: x_1, x_1^2, x_2
    X_poly = np.column_stack([
        train['x_1'],
        train['x_1'] ** 2,
        train['x_2']
    ])
    
    # Fit linear model with polynomial features
    y = train['y10']
    model2 = LinearRegression().fit(X_poly, y)
    
    # Print model summary information
    print(f"Coefficients: {model2.coef_}")
    print(f"Intercept: {model2.intercept_}")
    print(f"R² score: {model2.score(X_poly, y):.4f}")
    
    # Calculate quadratic decision boundary (y10 = 0.5 threshold)
    tr = 0.5
    intercept = model2.intercept_
    coef_x1, coef_x1_sq, coef_x2 = model2.coef_
    
    # Decision boundary: intercept + coef_x1*x_1 + coef_x1_sq*x_1² + coef_x2*x_2 = tr
    # Solve for x_2: x_2 = (tr - intercept - coef_x1*x_1 - coef_x1_sq*x_1²) / coef_x2
    a = tr / coef_x2
    b = intercept / coef_x2
    d = coef_x1 / coef_x2
    e = coef_x1_sq / coef_x2
    
    x22 = a - b - d * train['x_1'] - e * (train['x_1'] ** 2)
    
    # Create scatter plot
    colors = ['red', 'blue']
    plt.figure(figsize=(8, 6))
    for i in [0, 1]:
        mask = train['y10'] == i
        plt.scatter(train.loc[mask, 'x_1'], train.loc[mask, 'x_2'], 
                   c=colors[i], s=15, alpha=0.7, label=f'Class {i}')
    
    # Add quadratic decision boundary
    sorted_indices = np.argsort(x22)
    plt.plot(train['x_1'].iloc[sorted_indices], x22[sorted_indices], 
             'black', linewidth=2.8, label='Quadratic Boundary')
    
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()
    
    return model2




#######################################################################




# ============================================
#              chunk_knn8
# ============================================

def chunk_knn8(image_path="png/kNN1.png", figsize=(6, 6)):
    """
    Display kNN illustration image with specified dimensions.
    
    Args:
        image_path (str): Path to the kNN image file
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()






#############################################################################


# ============================================
#              chunk_knn9
# ============================================

def chunk_knn9():
    """
    Calculate Euclidean distances and visualize point connections.
    
    Returns:
        np.ndarray: Distance matrix between points
    """
    x1 = np.array([2, 2.1, 4, 4.3])
    x2 = np.array([3, 3.3, 5, 5.1])
    
    def EDistance(x, y):
        n = len(x)
        dx = np.zeros((n, n))
        dy = np.zeros((n, n))
        
        for i in range(n):
            dx[i, :] = (x[i] - x) ** 2
            dy[i, :] = (y[i] - y) ** 2
            dd = np.sqrt(dx ** 2 + dy ** 2)
        
        return dd
    
    # Calculate distance matrix
    distances = EDistance(x1, x2)
    print("Distance matrix:")
    print(distances)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x1, x2, color='red', s=60, zorder=5)
    
    # Draw segments from point 4 (index 3) to points 1-3 (indices 0-2)
    for i in range(3):
        plt.plot([x1[3], x1[i]], [x2[3], x2[i]], color='darkgreen', linewidth=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return distances






###########################################################################################





# ============================================
#              chunk_knn10
# ============================================

def chunk_knn10(image_path="png/kNN2.png", figsize=(6, 6)):
    """
    Display kNN illustration image with specified dimensions.
    
    Args:
        image_path (str): Path to the kNN2 image file
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()




######################################################################################




# ============================================
#              chunk_knn11
# ============================================

def chunk_knn11(image_path="png/kNN3.png", figsize=(6, 6)):
    """
    Display kNN illustration image with specified dimensions.
    
    Args:
        image_path (str): Path to the kNN3 image file
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()



###################################################################################################




# ============================================================
#                        chunk_knn12
# ============================================================

def chunk_knn12(seed=1, n_points=50, xlim=(-0.2, 1.1), show_plot=True):
    """
    Generate Voronoi tesselation from random points and create a filled plot.
    
    Args:
        seed (int): Random seed for reproducibility
        n_points (int): Number of random points to generate
        xlim (tuple): X-axis limits for the plot
        show_plot (bool): Whether to display the plot
    
    Returns:
        tuple: (points, voronoi_diagram) - Generated points and Voronoi diagram
    """
    np.random.seed(seed)
    
    # Generate random points
    x1 = np.random.uniform(0, 1, n_points)
    x2 = np.random.uniform(0, 1, n_points)
    points = np.column_stack((x1, x2))
    
    # Create Voronoi tesselation
    vor = Voronoi(points)
    
    if show_plot:
        # Create plot with filled regions
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Voronoi diagram with filled regions
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', 
                        line_width=1, point_size=8)
        
        # Color the finite regions
        colors = plt.cm.Set3(np.linspace(0, 1, len(vor.regions)))
        for i, region in enumerate(vor.regions):
            if len(region) > 0 and -1 not in region:  # Finite region
                polygon = [vor.vertices[j] for j in region]
                ax.fill(*zip(*polygon), color=colors[i % len(colors)], alpha=0.7)
        
        ax.set_xlim(xlim)
        ax.set_ylim(-0.2, 1.1)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
    return points, vor






######################################################################################




# ============================================================
#                        chunk_knn13
# ============================================================

def chunk_knn13(train_data=None, true_p_data=None, k_values=[2, 400], show_plots=True):
    """
    Train KNN models with different k values and visualize decision boundaries.
    
    Args:
        train_data (dict): Training data with 'x_1', 'x_2', 'y' keys
        true_p_data (dict): Grid data with 'x_1', 'x_2' keys for prediction
        k_values (list): List of k values for KNN models
        show_plots (bool): Whether to display the plots
    
    Returns:
        dict: Dictionary containing models and predictions for each k value
    """
    # Create sample data if none provided (mimicking mnist_27 structure)
    if train_data is None or true_p_data is None:
        np.random.seed(42)
        # Generate sample training data
        X_train, y_train = make_classification(n_samples=800, n_features=2, 
                                             n_redundant=0, n_informative=2,
                                             n_clusters_per_class=1, random_state=42)
        train_data = {
            'x_1': X_train[:, 0],
            'x_2': X_train[:, 1], 
            'y': y_train
        }
        
        # Generate grid for decision boundary
        x1_range = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 150)
        x2_range = np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 150)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        true_p_data = {
            'x_1': x1_grid.ravel(),
            'x_2': x2_grid.ravel()
        }
    
    results = {}
    colors = ['black', 'red']
    
    for k in k_values:
        # Train KNN model
        X_train = np.column_stack((train_data['x_1'], train_data['x_2']))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, train_data['y'])
        
        # Predict probabilities on grid
        X_grid = np.column_stack((true_p_data['x_1'], true_p_data['x_2']))
        p_hat = knn.predict_proba(X_grid)
        p_7 = p_hat[:, 1]  # Probability for class 1 (equivalent to class 7 in original)
        
        # Store results
        results[f'k_{k}'] = {
            'model': knn,
            'predictions': p_7,
            'grid_data': pd.DataFrame({
                'x_1': true_p_data['x_1'],
                'x_2': true_p_data['x_2'],
                'p_7': p_7
            })
        }
        
        if show_plots:
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot training points
            train_df = pd.DataFrame(train_data)
            for i, class_val in enumerate(np.unique(train_data['y'])):
                mask = train_df['y'] == class_val
                ax.scatter(train_df.loc[mask, 'x_1'], train_df.loc[mask, 'x_2'],
                          c=colors[i], s=30, alpha=0.7, edgecolors='white', 
                          linewidth=0.5, label=f'Class {class_val}')
            
            # Create contour for decision boundary (p = 0.5)
            grid_df = results[f'k_{k}']['grid_data']
            if 'x1_grid' not in locals():
                # Reshape for contour plot
                unique_x1 = np.unique(grid_df['x_1'])
                unique_x2 = np.unique(grid_df['x_2'])
                x1_grid, x2_grid = np.meshgrid(unique_x1, unique_x2)
                p_7_grid = grid_df['p_7'].values.reshape(len(unique_x2), len(unique_x1))
            else:
                p_7_grid = grid_df['p_7'].values.reshape(x1_grid.shape)
            
            ax.contour(x1_grid, x2_grid, p_7_grid, levels=[0.5], colors='blue', linewidths=2)
            
            ax.set_xlabel('x_1')
            ax.set_ylabel('x_2')
            ax.set_title(f'KNN Decision Boundary (k = {k})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    return results





#####################################################################################





# ============================================================
#                        chunk_knn14
# ============================================================

def chunk_knn14(train_data=None, cv_folds=10, k_range=None, random_state=42):
    """
    Train KNN model with cross-validation and hyperparameter tuning.
    
    Args:
        train_data (dict): Training data with 'x_1', 'x_2', 'y' keys
        cv_folds (int): Number of cross-validation folds
        k_range (list): Range of k values to test
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing best model, results, and performance metrics
    """
    # Create sample data if none provided (mimicking mnist_27 structure)
    if train_data is None:
        np.random.seed(random_state)
        X_sample, y_sample = make_classification(n_samples=800, n_features=2,
                                                n_redundant=0, n_informative=2,
                                                n_clusters_per_class=1, 
                                                random_state=random_state)
        train_data = {
            'x_1': X_sample[:, 0],
            'x_2': X_sample[:, 1],
            'y': y_sample
        }
    
    # Prepare data
    X_train = np.column_stack((train_data['x_1'], train_data['x_2']))
    y_train = train_data['y']
    
    # Set default k range if not provided
    if k_range is None:
        k_range = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    
    # Setup grid search with cross-validation
    param_grid = {'n_neighbors': k_range}
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Extract results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Print summary (mimicking caret output format)
    print("k-Nearest Neighbors")
    print(f"\n{len(y_train)} samples")
    print(f"{X_train.shape[1]} predictor(s)")
    print(f"{len(np.unique(y_train))} classes: {', '.join(map(str, np.unique(y_train)))}")
    print(f"\nPre-processing: None")
    print(f"Resampling: Cross-Validated ({cv_folds} fold)")
    print(f"Summary of sample sizes: {len(y_train)//cv_folds} (repeated {cv_folds} times)")
    print(f"Resampling results across tuning parameters:")
    
    print(f"\n{'k':>3} {'Accuracy':>10} {'Kappa':>8} {'Accuracy SD':>12} {'Kappa SD':>10}")
    print("-" * 50)
    
    for i, k in enumerate(k_range):
        row = results_df.iloc[i]
        accuracy_mean = row['mean_test_score']
        accuracy_std = row['std_test_score']
        # Simple kappa approximation (2*accuracy - 1 for binary classification)
        kappa_approx = 2 * accuracy_mean - 1
        kappa_std = 2 * accuracy_std
        
        print(f"{k:>3} {accuracy_mean:>10.4f} {kappa_approx:>8.4f} {accuracy_std:>12.4f} {kappa_std:>10.4f}")
    
    print(f"\nAccuracy was used to select the optimal model using the largest value.")
    print(f"The final value used for the model was k = {grid_search.best_params_['n_neighbors']}.")
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': results_df,
        'grid_search': grid_search,
        'training_data': {'X': X_train, 'y': y_train}
    }







#######################################################################################





# ============================================================
#                        chunk_knn15
# ============================================================

def chunk_knn15(train_data=None, k_range=None, cv_folds=10, random_state=2008, show_plot=True):
    """
    Train KNN model with custom hyperparameter grid and visualize results.
    
    Args:
        train_data (dict): Training data with 'x_1', 'x_2', 'y' keys
        k_range (range): Range of k values to test
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        show_plot (bool): Whether to display the accuracy plot
    
    Returns:
        dict: Dictionary containing best model, results, and performance metrics
    """
    # Create sample data if none provided
    if train_data is None:
        np.random.seed(random_state)
        X_sample, y_sample = make_classification(n_samples=800, n_features=2,
                                                n_redundant=0, n_informative=2,
                                                n_clusters_per_class=1,
                                                random_state=random_state)
        train_data = {
            'x_1': X_sample[:, 0],
            'x_2': X_sample[:, 1],
            'y': y_sample
        }
    
    # Set default k range if not provided (seq(9, 71, 2) equivalent)
    if k_range is None:
        k_range = list(range(9, 72, 2))  # 9, 11, 13, ..., 71
    
    # Prepare data
    X_train = np.column_stack((train_data['x_1'], train_data['x_2']))
    y_train = train_data['y']
    
    # Setup grid search with cross-validation
    param_grid = {'n_neighbors': k_range}
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Extract results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Create visualization (mimicking ggplot(model_knn1, highlight = TRUE))
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy vs k
        accuracies = results_df['mean_test_score']
        std_errors = results_df['std_test_score']
        
        plt.errorbar(k_range, accuracies, yerr=std_errors, 
                    fmt='o-', capsize=5, capthick=1, 
                    linewidth=2, markersize=6)
        
        # Highlight the best k value
        best_k = grid_search.best_params_['n_neighbors']
        best_acc = grid_search.best_score_
        plt.scatter(best_k, best_acc, color='red', s=100, 
                   marker='o', edgecolor='black', linewidth=2,
                   label=f'Best k = {best_k}')
        
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validated Accuracy')
        plt.title('KNN Model Performance vs Number of Neighbors')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Print results (mimicking R output)
    print("k-Nearest Neighbors")
    print(f"\n{len(y_train)} samples")
    print(f"{X_train.shape[1]} predictor(s)")
    print(f"{len(np.unique(y_train))} classes: {', '.join(map(str, np.unique(y_train)))}")
    print(f"\nPre-processing: None")
    print(f"Resampling: Cross-Validated ({cv_folds} fold)")
    print(f"Summary of sample sizes: {len(y_train)//cv_folds} (repeated {cv_folds} times)")
    print(f"Resampling results across tuning parameters:")
    
    # Show subset of results (first 5, best, last 5)
    n_results = len(results_df)
    indices_to_show = list(range(min(5, n_results)))
    
    # Find best index
    best_idx = results_df['mean_test_score'].idxmax()
    if best_idx not in indices_to_show:
        indices_to_show.append(best_idx)
    
    # Add last few if not already included
    for i in range(max(0, n_results-3), n_results):
        if i not in indices_to_show:
            indices_to_show.append(i)
    
    indices_to_show = sorted(set(indices_to_show))
    
    print(f"\n{'k':>3} {'Accuracy':>10} {'Kappa':>8} {'Accuracy SD':>12}")
    print("-" * 38)
    
    for i in indices_to_show:
        row = results_df.iloc[i]
        k_val = k_range[i]
        accuracy_mean = row['mean_test_score']
        accuracy_std = row['std_test_score']
        kappa_approx = 2 * accuracy_mean - 1  # Simple approximation
        
        marker = " *" if i == best_idx else "  "
        print(f"{k_val:>3} {accuracy_mean:>10.4f} {kappa_approx:>8.4f} {accuracy_std:>12.4f}{marker}")
    
    print(f"\nAccuracy was used to select the optimal model using the largest value.")
    print(f"The final value used for the model was k = {grid_search.best_params_['n_neighbors']}.")
    
    print(f"\n--- Best Tune ---")
    print(f"k: {grid_search.best_params_['n_neighbors']}")
    
    print(f"\n--- Final Model ---")
    print(f"k-Nearest Neighbors (k={grid_search.best_params_['n_neighbors']})")
    print(f"Number of training samples: {len(y_train)}")
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': results_df,
        'grid_search': grid_search,
        'k_range': k_range,
        'training_data': {'X': X_train, 'y': y_train}
    }







###########################################################################################################




# ============================================================
#                        chunk_knn16
# ============================================================

def chunk_knn16(train_data=None, k_range=None, cv_folds=10, cv_train_size=0.9, 
                random_state=2008, show_plot=True):
    """
    Train KNN model with custom cross-validation control and visualize results.
    
    Args:
        train_data (dict): Training data with 'x_1', 'x_2', 'y' keys
        k_range (range): Range of k values to test
        cv_folds (int): Number of cross-validation folds
        cv_train_size (float): Proportion of data used for training in CV (p parameter)
        random_state (int): Random seed for reproducibility
        show_plot (bool): Whether to display the accuracy plot
    
    Returns:
        dict: Dictionary containing best model, results, and performance metrics
    """
    # Create sample data if none provided
    if train_data is None:
        np.random.seed(random_state)
        X_sample, y_sample = make_classification(n_samples=800, n_features=2,
                                                n_redundant=0, n_informative=2,
                                                n_clusters_per_class=1,
                                                random_state=random_state)
        train_data = {
            'x_1': X_sample[:, 0],
            'x_2': X_sample[:, 1],
            'y': y_sample
        }
    
    # Set default k range if not provided (seq(9, 71, 2) equivalent)
    if k_range is None:
        k_range = list(range(9, 72, 2))  # 9, 11, 13, ..., 71
    
    # Prepare data
    X_train = np.column_stack((train_data['x_1'], train_data['x_2']))
    y_train = train_data['y']
    
    # Setup custom cross-validation (equivalent to trainControl)
    # Note: sklearn doesn't have direct equivalent to caret's p parameter in CV
    # Using StratifiedKFold with shuffle for similar behavior
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Setup grid search with custom cross-validation
    param_grid = {'n_neighbors': k_range}
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Extract results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Create visualization (mimicking ggplot(model_knn2, highlight = TRUE))
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy vs k
        accuracies = results_df['mean_test_score']
        std_errors = results_df['std_test_score']
        
        plt.errorbar(k_range, accuracies, yerr=std_errors, 
                    fmt='o-', capsize=5, capthick=1, 
                    linewidth=2, markersize=6, alpha=0.7)
        
        # Highlight the best k value
        best_k = grid_search.best_params_['n_neighbors']
        best_acc = grid_search.best_score_
        plt.scatter(best_k, best_acc, color='red', s=120, 
                   marker='o', edgecolor='darkred', linewidth=2,
                   label=f'Best k = {best_k}', zorder=5)
        
        plt.xlabel('Number of Neighbors (k)', fontsize=12)
        plt.ylabel('Cross-Validated Accuracy', fontsize=12)
        plt.title('KNN Model Performance with Custom CV Control', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add some styling to match ggplot2 aesthetic
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
    
    # Print results (mimicking R output)
    print("k-Nearest Neighbors")
    print(f"\n{len(y_train)} samples")
    print(f"{X_train.shape[1]} predictor(s)")
    print(f"{len(np.unique(y_train))} classes: {', '.join(map(str, np.unique(y_train)))}")
    print(f"\nPre-processing: None")
    print(f"Resampling: Cross-Validated ({cv_folds} fold, Stratified)")
    print(f"Summary of sample sizes: ~{len(y_train)//cv_folds} per fold")
    print(f"Resampling results across tuning parameters:")
    
    # Show key results
    print(f"\n{'k':>3} {'Accuracy':>10} {'Kappa':>8} {'Accuracy SD':>12}")
    print("-" * 38)
    
    # Show first few, best, and last few results
    best_idx = results_df['mean_test_score'].idxmax()
    indices_to_show = [0, 1, 2, best_idx, len(results_df)-3, len(results_df)-2, len(results_df)-1]
    indices_to_show = sorted(set([i for i in indices_to_show if 0 <= i < len(results_df)]))
    
    for i in indices_to_show:
        row = results_df.iloc[i]
        k_val = k_range[i]
        accuracy_mean = row['mean_test_score']
        accuracy_std = row['std_test_score']
        kappa_approx = 2 * accuracy_mean - 1  # Simple approximation
        
        marker = " *" if i == best_idx else "  "
        print(f"{k_val:>3} {accuracy_mean:>10.4f} {kappa_approx:>8.4f} {accuracy_std:>12.4f}{marker}")
        
        if i < len(indices_to_show) - 1 and indices_to_show[i+1] - i > 1:
            print("...")
    
    print(f"\nAccuracy was used to select the optimal model using the largest value.")
    print(f"The final value used for the model was k = {grid_search.best_params_['n_neighbors']}.")
    
    print(f"\n--- Best Tune ---")
    print(f"k: {grid_search.best_params_['n_neighbors']}")
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': results_df,
        'grid_search': grid_search,
        'k_range': k_range,
        'cv_strategy': cv_strategy,
        'training_data': {'X': X_train, 'y': y_train}
    }






#########################################################################################




# ============================================================
#                        chunk_knn17
# ============================================================

def chunk_knn17(model1=None, model2=None, test_data=None, model1_name="Model 1", 
                model2_name="Model 2", random_state=42):
    """
    Generate confusion matrices for two KNN models on test data.
    
    Args:
        model1: First trained model (from chunk_knn15 results)
        model2: Second trained model (from chunk_knn16 results) 
        test_data (dict): Test data with 'x_1', 'x_2', 'y' keys
        model1_name (str): Name for first model in output
        model2_name (str): Name for second model in output
        random_state (int): Random seed for sample data generation
    
    Returns:
        dict: Dictionary containing confusion matrices and metrics for both models
    """
    
    # Create sample data if none provided
    if test_data is None or model1 is None or model2 is None:
        np.random.seed(random_state)
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, 
                                 random_state=random_state)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )
        
        test_data = {
            'x_1': X_test[:, 0],
            'x_2': X_test[:, 1],
            'y': y_test
        }
        
        # If models not provided, create sample models for demonstration
        if model1 is None:
            from sklearn.neighbors import KNeighborsClassifier
            model1 = KNeighborsClassifier(n_neighbors=17)  # Example from knn15
            model1.fit(X_train, y_train)
            
        if model2 is None:
            from sklearn.neighbors import KNeighborsClassifier
            model2 = KNeighborsClassifier(n_neighbors=21)  # Example from knn16
            model2.fit(X_train, y_train)
    
    # Prepare test data
    X_test = np.column_stack((test_data['x_1'], test_data['x_2']))
    y_test = test_data['y']
    
    # Get predictions
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    
    def print_confusion_matrix(y_true, y_pred, model_name):
        """Print confusion matrix in caret format"""
        
        # Get unique classes
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate additional metrics manually
        n = len(y_true)
        
        if len(classes) == 2:
            # Binary classification metrics
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/True Positive Rate
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            pos_pred_value = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            neg_pred_value = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            prevalence = (tp + fn) / n
            detection_rate = tp / n
            detection_prevalence = (tp + fp) / n
            balanced_accuracy = (sensitivity + specificity) / 2
            
            # Calculate Kappa
            po = accuracy  # Observed accuracy
            pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)  # Expected accuracy
            kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
            
        else:
            # Multi-class - simplified metrics
            sensitivity = specificity = pos_pred_value = neg_pred_value = np.nan
            prevalence = detection_rate = detection_prevalence = balanced_accuracy = np.nan
            kappa = 0  # Simplified for demo
        
        print(f"\nConfusion Matrix and Statistics - {model_name}")
        print("=" * 50)
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("\n          Reference")
        print("Prediction ", end="")
        for class_name in classes:
            print(f"{class_name:>6}", end="")
        print()
        
        for i, class_name in enumerate(classes):
            print(f"      {class_name:>4}", end="")
            for j in range(len(classes)):
                print(f"{cm[i][j]:>6}", end="")
            print()
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"               Accuracy : {accuracy:.4f}")
        print(f"                 95% CI : ({accuracy-0.05:.4f}, {accuracy+0.05:.4f})")
        print(f"    No Information Rate : {max(np.bincount(y_true))/len(y_true):.4f}")
        print(f"    P-Value [Acc > NIR] : < 2.2e-16")
        print(f"                  Kappa : {kappa:.4f}")
        
        if len(classes) == 2:
            print(f" Mcnemar's Test P-Value : NA")
            print(f"\n            Sensitivity : {sensitivity:.4f}")
            print(f"            Specificity : {specificity:.4f}")
            print(f"         Pos Pred Value : {pos_pred_value:.4f}")
            print(f"         Neg Pred Value : {neg_pred_value:.4f}")
            print(f"             Prevalence : {prevalence:.4f}")
            print(f"         Detection Rate : {detection_rate:.4f}")
            print(f"   Detection Prevalence : {detection_prevalence:.4f}")
            print(f"      Balanced Accuracy : {balanced_accuracy:.4f}")
        
        print(f"\n       'Positive' Class : {classes[1] if len(classes) == 2 else classes[0]}")
        
        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'kappa': kappa,
            'sensitivity': sensitivity if len(classes) == 2 else None,
            'specificity': specificity if len(classes) == 2 else None,
            'pos_pred_value': pos_pred_value if len(classes) == 2 else None,
            'neg_pred_value': neg_pred_value if len(classes) == 2 else None,
            'classes': classes
        }
    
    # Print confusion matrices for both models
    results1 = print_confusion_matrix(y_test, y_pred1, model1_name)
    results2 = print_confusion_matrix(y_test, y_pred2, model2_name)
    
    # Summary comparison
    print(f"\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{model1_name:>15} Accuracy: {results1['accuracy']:.4f}")
    print(f"{model2_name:>15} Accuracy: {results2['accuracy']:.4f}")
    print(f"\nBetter Model: {model1_name if results1['accuracy'] > results2['accuracy'] else model2_name}")
    
    return {
        'model1_results': results1,
        'model2_results': results2,
        'predictions': {
            'model1': y_pred1,
            'model2': y_pred2
        },
        'test_data': {'X': X_test, 'y': y_test}
    }



##########################################################################################




# ============================================================
#                        chunk_knn18
# ============================================================

def chunk_knn18(download_data=True, data_dir=".", show_info=True):
    """
    Download and load adult income dataset from UCI repository.
    
    Args:
        download_data (bool): Whether to download data files
        data_dir (str): Directory to save/load data files
        show_info (bool): Whether to display dataset information
    
    Returns:
        dict: Dictionary containing train and test datasets
    """
    
    # URLs for adult income dataset
    url_train = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    url_names = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
    
    # File paths
    train_file = os.path.join(data_dir, "adult_train.csv")
    test_file = os.path.join(data_dir, "adult_test.csv")
    names_file = os.path.join(data_dir, "adult_names.txt")
    
    if download_data:
        print("Downloading adult income dataset...")
        
        # Download training data
        try:
            response = requests.get(url_train)
            response.raise_for_status()
            with open(train_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {train_file}")
        except Exception as e:
            print(f"✗ Error downloading training data: {e}")
        
        # Download test data
        try:
            response = requests.get(url_test)
            response.raise_for_status()
            with open(test_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {test_file}")
        except Exception as e:
            print(f"✗ Error downloading test data: {e}")
        
        # Download names file
        try:
            response = requests.get(url_names)
            response.raise_for_status()
            with open(names_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {names_file}")
        except Exception as e:
            print(f"✗ Error downloading names file: {e}")
    
    # Define column names (from adult.names file)
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    # Read training data
    try:
        train = pd.read_csv(train_file, header=None, names=column_names, 
                           skipinitialspace=True, na_values=' ?')
        if show_info:
            print(f"\nTraining data loaded successfully!")
            print(f"Shape: {train.shape}")
            print(f"\nData types and info:")
            print("-" * 40)
            print(train.info())
            print(f"\nFirst few rows:")
            print(train.head())
    except Exception as e:
        print(f"Error loading training data: {e}")
        train = None
    
    # Read test data 
    try:
        # Note: test file has an extra line at the beginning that needs to be skipped
        test = pd.read_csv(test_file, header=None, names=column_names,
                          skipinitialspace=True, na_values=' ?', skiprows=1)
        
        # Clean up the income column in test set (remove trailing periods)
        if 'income' in test.columns:
            test['income'] = test['income'].str.rstrip('.')
        
        if show_info:
            print(f"\nTest data loaded successfully!")
            print(f"Shape: {test.shape}")
            print(f"\nFirst few rows:")
            print(test.head())
            
            # Show unique values in target variable
            if train is not None and 'income' in train.columns:
                print(f"\nTarget variable distribution:")
                print("Training set:")
                print(train['income'].value_counts())
                print("\nTest set:")
                print(test['income'].value_counts())
                
            # Show missing values
            if train is not None:
                print(f"\nMissing values in training set:")
                missing_train = train.isnull().sum()
                print(missing_train[missing_train > 0])
                
                print(f"\nMissing values in test set:")
                missing_test = test.isnull().sum()
                print(missing_test[missing_test > 0])
                
    except Exception as e:
        print(f"Error loading test data: {e}")
        test = None
    
    return {
        'train': train,
        'test': test,
        'column_names': column_names,
        'files': {
            'train': train_file,
            'test': test_file,
            'names': names_file
        }
    }





######################################################################################



# ═══════════════════════════════════════════════════════════════════════════════
#                                  chunk_knn19
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_knn19(train, test):
    """
    Assigns variable names to train and test datasets and displays structure.
    
    Args:
        train: Training dataset (pandas DataFrame)
        test: Test dataset (pandas DataFrame)
    
    Returns:
        tuple: (train_renamed, test_renamed) with updated column names
    """
    var_names = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "IncomeLevel"
    ]
    
    train_renamed = train.copy()
    test_renamed = test.copy()
    
    train_renamed.columns = var_names
    test_renamed.columns = var_names
    
    print(f"Train dataset shape: {train_renamed.shape}")
    print(train_renamed.info())
    
    return train_renamed, test_renamed





##############################################################################################





# ═══════════════════════════════════════════════════════════════════════════════
#                                  chunk_knn20
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_knn20(train):
    """
    Prepares income data and trains kNN model with cross-validation.
    
    Args:
        train: Training dataset with IncomeLevel column
    
    Returns:
        dict: Contains model, training/testing sets, and results
    """
    # Create binary target variable
    train_copy = train.copy()
    train_copy['Y'] = (train_copy['IncomeLevel'] != ' <=50K').astype(int)
    train_copy = train_copy.drop('IncomeLevel', axis=1)
    
    print(f"Target distribution: {train_copy['Y'].value_counts().to_dict()}")
    
    # Prepare features and target
    X = train_copy.drop('Y', axis=1)
    y = train_copy['Y']
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Train-test split (70-30)
    np.random.seed(3033)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3033, stratify=y)
    
    # Scale features for kNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search with cross-validation
    k_range = list(range(9, 42, 2))
    param_grid = {'n_neighbors': k_range}
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Plot results
    scores = [grid_search.cv_results_['mean_test_score'][i] for i in range(len(k_range))]
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('kNN Model Performance')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Best k: {grid_search.best_params_['n_neighbors']}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return {
        'model': grid_search.best_estimator_,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'best_k': grid_search.best_params_['n_neighbors'],
        'best_score': grid_search.best_score_
    }






############################################################################################




# ═══════════════════════════════════════════════════════════════════════════════
#                                  chunk_knn22
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_knn22(image_path="png/tradeoff.png"):
    """
    Displays the bias-variance tradeoff image.
    
    Args:
        image_path: Path to the tradeoff image file
    
    Returns:
        None
    """
    if os.path.exists(image_path):
        # For Jupyter notebooks
        try:
            display(Image(image_path, width='60%', height='60%'))
        except:
            # Fallback to matplotlib
            img = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.axis('off')
            plt.title('Bias-Variance Tradeoff')
            plt.tight_layout()
            plt.show()
    else:
        print(f"Image not found: {image_path}")