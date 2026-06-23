import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.inspection import permutation_importance


# ============================================
#              chunk_sv1
# ============================================

def chunk_sv1():
    """Convert R script sv1 to Python: create scatter plot with color-coded points"""
    
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    x1 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.72])
    
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'green' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.7)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return data





########################################################################################################################


# ============================================
#              chunk_sv2
# ============================================

def chunk_sv2():
    """Convert R script sv2 to Python: scatter plot with added regression line"""
    
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    x1 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.72])
    
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'green' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.7)
    
    # Add line: y = a + b*x (where a=0.29, b=0.6)
    x_line = np.linspace(data['x1'].min(), data['x1'].max(), 100)
    y_line = 0.29 + 0.6 * x_line
    plt.plot(x_line, y_line, color='orange', linewidth=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return data




########################################################################################################################


# ============================================
#              chunk_sv3
# ============================================

def chunk_sv3():
    """Convert R script sv3 to Python: scatter plot with multiple regression lines"""
    
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    x1 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.72])
    
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'green' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.7)
    
    # Add multiple lines with different parameters
    x_line = np.linspace(data['x1'].min(), data['x1'].max(), 100)
    
    # Blue line: y = 0.29 + 0.6*x
    plt.plot(x_line, 0.29 + 0.6 * x_line, color='blue', linewidth=2)
    
    # Orange line: y = 0.20 + 0.8*x
    plt.plot(x_line, 0.20 + 0.8 * x_line, color='orange', linewidth=2)
    
    # Green line: y = 0.10 + 1.05*x
    plt.plot(x_line, 0.10 + 1.05 * x_line, color='green', linewidth=2)
    
    # Brown line: y = 0.38 + 0.47*x
    plt.plot(x_line, 0.38 + 0.47 * x_line, color='brown', linewidth=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return data





########################################################################################################################


# ============================================
#              chunk_sv4
# ============================================

def chunk_sv4():
    """Convert R script sv4 to Python: scatter plot with decision boundaries and margin annotation"""
    
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    x1 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.72])
    
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'green' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.7)
    
    # Add lines with different styles
    x_line = np.linspace(data['x1'].min(), data['x1'].max(), 100)
    
    # Red dashed line (H1): y = 0.30 + 0.8*x
    plt.plot(x_line, 0.30 + 0.8 * x_line, color='red', linewidth=2, linestyle='--')
    plt.text(0.4, 0.65, "H1", color='red', fontsize=12)
    
    # Gray solid line (Boundary): y = 0.20 + 0.8*x
    plt.plot(x_line, 0.20 + 0.8 * x_line, color='gray', linewidth=2)
    plt.text(0.3, 0.45, "Boundary", color='gray', fontsize=12)
    
    # Black dashed line (H0): y = 0.10 + 0.8*x
    plt.plot(x_line, 0.10 + 0.8 * x_line, color='black', linewidth=2, linestyle='--')
    plt.text(0.4, 0.45, "H0", color='black', fontsize=12)
    
    # Arrow for margin annotation
    plt.annotate('', xy=(0.47, 0.66), xytext=(0.483, 0.587),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    plt.text(0.55, 0.64, "Margin - 'm'", color='blue', fontsize=12)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return data




########################################################################################################################


# ============================================
#              chunk_sv5
# ============================================

def chunk_sv5(image_path="png/triangle.png"):
    """Convert R script sv5 to Python: display external image file"""
    
    try:
        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(4, 4))  # Equivalent to 40% size
        ax.imshow(img)
        ax.axis('off')  # Remove axes for clean image display
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found")
    except Exception as e:
        print(f"Error loading image: {e}")
    
    return None




########################################################################################################################


# ============================================
#              chunk_sv6
# ============================================

def chunk_sv6():
    """Convert R script sv6 to Python: scatter plot with decision boundaries and vector annotations"""
    
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    x1 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.72])
    
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    
    # Create scatter plot with colors based on y values
    colors = ['red' if val == 0 else 'green' for val in data['y']]
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x1'], data['x2'], c=colors, s=100, alpha=0.7)
    
    # Add lines with different styles
    x_line = np.linspace(data['x1'].min(), data['x1'].max(), 100)
    
    # Red dashed line (H1): y = 0.30 + 0.8*x
    plt.plot(x_line, 0.30 + 0.8 * x_line, color='red', linewidth=2, linestyle='--')
    plt.text(0.4, 0.65, "H1", color='red', fontsize=12)
    
    # Gray solid line (Boundary): y = 0.20 + 0.8*x
    plt.plot(x_line, 0.20 + 0.8 * x_line, color='gray', linewidth=2)
    plt.text(0.3, 0.45, "Boundary", color='gray', fontsize=12)
    
    # Black dashed line (H0): y = 0.10 + 0.8*x
    plt.plot(x_line, 0.10 + 0.8 * x_line, color='black', linewidth=2, linestyle='--')
    plt.text(0.4, 0.45, "H0", color='black', fontsize=12)
    
    # Blue arrow for margin annotation
    plt.annotate('', xy=(0.47, 0.66), xytext=(0.481, 0.587),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    plt.text(0.55, 0.64, "Margin - 'm'", color='blue', fontsize=12)
    
    # Dark green arrow for weight vector
    plt.annotate('', xy=(0.54, 0.90), xytext=(0.565, 0.76),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    plt.text(0.53, 0.80, "w", color='darkgreen', fontsize=12)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return data





########################################################################################################################


# ============================================
#              chunk_sv7
# ============================================

def chunk_sv7():
    """Convert R script sv7 to Python: SVM with linear kernel on perfectly separated data"""
    
    # Sample data - Perfectly separated
    np.random.seed(1)
    x = np.random.randn(20, 2)
    y = np.array([-1] * 10 + [1] * 10)
    x[y == 1, :] = x[y == 1, :] + 2
    
    # Create DataFrame
    dat = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # Support Vector Machine model
    mfit = SVC(kernel='linear', C=1e10)  # Large C for hard margin
    mfit.fit(x, y)
    
    print("SVM Model Summary:")
    print(f"Support vectors: {mfit.n_support_}")
    print(f"Number of support vectors: {len(mfit.support_)}")
    
    # Create decision boundary plot
    plt.figure(figsize=(8, 6))
    
    # Create a mesh for plotting decision boundary
    h = 0.05
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = mfit.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linestyles='--', linewidths=1)
    
    # Plot data points
    colors = ['lightcoral' if label == -1 else 'lightblue' for label in y]
    plt.scatter(x[:, 0], x[:, 1], c=colors, s=100, edgecolor='black', alpha=0.8)
    
    # Highlight support vectors
    plt.scatter(x[mfit.support_, 0], x[mfit.support_, 1], 
               s=200, facecolors='none', edgecolors='red', linewidths=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM with Linear Kernel')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dat, mfit





########################################################################################################################


# ============================================
#              chunk_sv8
# ============================================

def chunk_sv8():
    """Convert R script sv8 to Python: side-by-side comparison of separable vs non-separable data"""
    
    # First dataset
    y1 = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    x1_1 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2_1 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.52])
    data1 = pd.DataFrame({"y": y1, "x1": x1_1, "x2": x2_1})
    
    # Second dataset
    y2 = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    x1_2 = np.array([0.09, 0.11, 0.17, 0.23, 0.33, 0.5, 0.54, 0.65, 0.83, 0.78])
    x2_2 = np.array([0.5, 0.82, 0.24, 0.09, 0.56, 0.40, 0.93, 0.82, 0.3, 0.72])
    data2 = pd.DataFrame({"y": y2, "x1": x1_2, "x2": x2_2})
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot - Separable with tight margin
    colors1 = ['red' if val == 0 else 'green' for val in data1['y']]
    ax1.scatter(data1['x1'], data1['x2'], c=colors1, s=80, alpha=0.7)
    
    # Add lines for first plot
    x_line = np.linspace(data1['x1'].min(), data1['x1'].max(), 100)
    ax1.plot(x_line, 0.29 + 0.45 * x_line, color='blue', linewidth=2)
    ax1.plot(x_line, 0.41 + 0.08 * x_line, color='orange', linewidth=2)
    
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Separable with tight margin', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot - Non-separable without error
    colors2 = ['red' if val == 0 else 'green' for val in data2['y']]
    ax2.scatter(data2['x1'], data2['x2'], c=colors2, s=80, alpha=0.7)
    
    # Add line for second plot
    x_line = np.linspace(data2['x1'].min(), data2['x1'].max(), 100)
    ax2.plot(x_line, 0.21 + 0.77 * x_line, color='blue', linewidth=2)
    
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Non-Separable without error', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return data1, data2





########################################################################################################################


# ============================================
#              chunk_sv9
# ============================================

def chunk_sv9():
    """Convert R script sv9 to Python: SVM with C parameter tuning and model comparison"""
    
    # Generate data
    np.random.seed(1)
    x = np.random.randn(20, 2)
    y = np.array([-1] * 10 + [1] * 10)
    x[y == 1, :] = x[y == 1, :] + 1
    dt = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # SVM with C = 10
    mfit10 = SVC(kernel='linear', C=10)
    mfit10.fit(x, y)
    
    # Plot SVM with C = 10
    plt.figure(figsize=(8, 6))
    h = 0.05
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = mfit10.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linestyles='--', linewidths=1)
    
    colors = ['lightcoral' if label == -1 else 'lightblue' for label in y]
    plt.scatter(x[:, 0], x[:, 1], c=colors, s=100, edgecolor='black', alpha=0.8)
    plt.scatter(x[mfit10.support_, 0], x[mfit10.support_, 1], 
               s=200, facecolors='none', edgecolors='red', linewidths=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('C = 10')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Tuning C parameter
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}
    tuned = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, scoring='accuracy')
    tuned.fit(x, y)
    
    best = tuned.best_estimator_
    print(f"Best C parameter: {tuned.best_params_['C']}")
    print(f"Best cross-validation score: {tuned.best_score_:.4f}")
    
    # Predictions using best model
    yhat = best.predict(x)
    misclass = confusion_matrix(dt['y'], yhat)
    
    print("\nConfusion Matrix:")
    print("Truth\\Predict  -1   1")
    print(f"        -1     {misclass[0,0]:2d}  {misclass[0,1]:2d}")
    print(f"         1     {misclass[1,0]:2d}  {misclass[1,1]:2d}")
    
    return dt, mfit10, best




########################################################################################################################


# ============================================
#              chunk_sv10
# ============================================

def chunk_sv10():
    """Convert R script sv10 to Python: generate and plot non-linear separable data"""
    
    # Generate data
    np.random.seed(1)
    x = np.random.randn(200, 2)
    x[:100, :] = x[:100, :] + 2
    x[100:150, :] = x[100:150, :] - 2
    y = np.array([1] * 150 + [2] * 50)
    
    dt = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # Create plot with colors matching R's y*2 scheme
    colors = ['red' if label == 1 else 'green' for label in y]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=colors, s=50, alpha=0.7)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dt




########################################################################################################################


# ============================================
#              chunk_sv11
# ============================================

def chunk_sv11():
    """Convert R script sv11 to Python: SVM with polynomial kernel on non-linear data"""
    
    # Generate the same data as sv10
    np.random.seed(1)
    x = np.random.randn(200, 2)
    x[:100, :] = x[:100, :] + 2
    x[100:150, :] = x[100:150, :] - 2
    y = np.array([1] * 150 + [2] * 50)
    
    dt = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # SVM with polynomial kernel
    svmfit = SVC(kernel='poly', C=1, degree=2, gamma='scale', coef0=1)
    svmfit.fit(x, y)
    
    # Create decision boundary plot
    plt.figure(figsize=(8, 6))
    
    # Create mesh for plotting decision boundary
    h = 0.1
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = svmfit.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, levels=[0.5, 1.5, 2.5], colors=['pink', 'lightblue'], alpha=0.6)
    
    # Plot data points
    colors = ['red' if label == 1 else 'blue' for label in y]
    plt.scatter(x[:, 0], x[:, 1], c=colors, s=50, edgecolor='black', alpha=0.8)
    
    # Highlight support vectors
    plt.scatter(x[svmfit.support_, 0], x[svmfit.support_, 1], 
               s=150, facecolors='none', edgecolors='green', linewidths=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM with Polynomial Kernel (degree=2)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Number of support vectors: {len(svmfit.support_)}")
    
    return dt, svmfit





########################################################################################################################


# ============================================
#              chunk_sv12
# ============================================

def chunk_sv12():
    """Convert R script sv12 to Python: SVM with RBF kernel hyperparameter tuning"""
    
    # Generate the same data as sv10
    np.random.seed(1)
    x = np.random.randn(200, 2)
    x[:100, :] = x[:100, :] + 2
    x[100:150, :] = x[100:150, :] - 2
    y = np.array([1] * 150 + [2] * 50)
    
    dt = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # Hyperparameter tuning for RBF kernel
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.5, 1, 2, 3, 4]
    }
    
    tune_out = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
    tune_out.fit(x, y)
    
    best_model = tune_out.best_estimator_
    print(f"Best parameters: C={tune_out.best_params_['C']}, gamma={tune_out.best_params_['gamma']}")
    print(f"Best cross-validation score: {tune_out.best_score_:.4f}")
    
    # Create decision boundary plot with best model
    plt.figure(figsize=(8, 6))
    
    # Create mesh for plotting decision boundary
    h = 0.1
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, levels=[0.5, 1.5, 2.5], colors=['pink', 'lightblue'], alpha=0.6)
    
    # Plot data points
    colors = ['red' if label == 1 else 'blue' for label in y]
    plt.scatter(x[:, 0], x[:, 1], c=colors, s=50, edgecolor='black', alpha=0.8)
    
    # Highlight support vectors
    plt.scatter(x[best_model.support_, 0], x[best_model.support_, 1], 
               s=150, facecolors='none', edgecolors='green', linewidths=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'SVM with RBF Kernel (C={tune_out.best_params_["C"]}, γ={tune_out.best_params_["gamma"]})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dt, tune_out, best_model





########################################################################################################################


# ============================================
#              chunk_sv13
# ============================================

def chunk_sv13():
    """Convert R script sv13 to Python: ROC curve comparison between SVM and Logistic Regression"""
    
    # Generate the same data as previous scripts
    np.random.seed(1)
    x = np.random.randn(200, 2)
    x[:100, :] = x[:100, :] + 2
    x[100:150, :] = x[100:150, :] - 2
    y = np.array([1] * 150 + [2] * 50)
    
    dt = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # Train-test split
    np.random.seed(1)
    train_indices = np.sort(np.random.choice(200, 100, replace=False))[::-1]
    test_indices = np.setdiff1d(np.arange(200), train_indices)
    
    X_train, X_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Convert labels to binary (1 vs 2 -> 0 vs 1)
    y_train_binary = (y_train == 2).astype(int)
    y_test_binary = (y_test == 2).astype(int)
    
    # SVM with RBF kernel
    svm_model = SVC(kernel='rbf', C=1, gamma=0.5, probability=True)
    svm_model.fit(X_train, y_train_binary)
    svm_scores = svm_model.decision_function(X_test)
    
    # Logistic Regression
    logit_model = LogisticRegression(max_iter=1000)
    logit_model.fit(X_train, y_train_binary)
    logit_scores = logit_model.predict_proba(X_test)[:, 1]
    
    # ROC curves
    fpr_svm, tpr_svm, _ = roc_curve(y_test_binary, svm_scores)
    fpr_logit, tpr_logit, _ = roc_curve(y_test_binary, logit_scores)
    
    # AUC scores
    auc_svm = auc(fpr_svm, tpr_svm)
    auc_logit = auc(fpr_logit, tpr_logit)
    
    print(f"SVM AUC: {auc_svm:.4f}")
    print(f"Logistic Regression AUC: {auc_logit:.4f}")
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f})', linewidth=2)
    plt.plot(fpr_logit, tpr_logit, label=f'Logistic Regression (AUC = {auc_logit:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: SVM vs Logistic Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dt, svm_model, logit_model, auc_svm, auc_logit





########################################################################################################################



# ============================================
#              chunk_sv14
# ============================================

def chunk_sv14(file_path="adult_train.csv"):
    """Convert R script sv14 to Python: load and preprocess adult income dataset"""
    
    # Define variable names
    varNames = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
                "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
                "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "IncomeLevel"]
    
    # Read CSV file
    try:
        train = pd.read_csv(file_path, header=None, names=varNames)
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Please ensure the file exists.")
        return None, None
    
    data = train.copy()
    
    # Check income level distribution
    tbl = data['IncomeLevel'].value_counts()
    print("Income Level Distribution:")
    print(tbl)
    
    # Remove outliers - Holand-Netherlands entries
    ind = data[data['NativeCountry'] == " Holand-Netherlands"].index
    if len(ind) > 0:
        data = data.drop(ind).reset_index(drop=True)
        print(f"Removed {len(ind)} outlier(s) with NativeCountry = ' Holand-Netherlands'")
    
    # Convert string columns to categorical
    df = data.copy()
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype('category')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Converted {len(string_cols)} string columns to categorical")
    
    return data, df




########################################################################################################################


# ============================================
#              chunk_sv15
# ============================================

def chunk_sv15(df):
    """Convert R script sv15 to Python: train-test split and SVM hyperparameter tuning on adult dataset"""
    
    if df is None:
        print("No data provided. Please run chunk_sv14 first.")
        return None, None, None, None
    
    # Prepare data for sklearn
    df_encoded = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for column in df_encoded.select_dtypes(include=['category', 'object']).columns:
        if column != 'IncomeLevel':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            label_encoders[column] = le
    
    # Separate features and target
    X = df_encoded.drop('IncomeLevel', axis=1)
    y = df_encoded['IncomeLevel']
    
    # Initial 90-10% split
    np.random.seed(123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=123)
    
    # Use 10% of training data for hyperparameter tuning
    np.random.seed(321)
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, test_size=0.90, random_state=321)
    
    print(f"Original dataset size: {len(df_encoded)}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Tuning subset size: {len(X_tune)}")
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.05, 0.5, 1, 2, 3, 4]
    }
    
    print("Performing hyperparameter tuning...")
    tuning = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    tuning.fit(X_tune, y_tune)
    
    best_model = tuning.best_estimator_
    print(f"Best parameters: C={tuning.best_params_['C']}, gamma={tuning.best_params_['gamma']}")
    print(f"Best cross-validation score: {tuning.best_score_:.4f}")
    
    return X_train, X_test, y_train, y_test, X_tune, y_tune, tuning, best_model





########################################################################################################################


# ============================================
#              chunk_sv16
# ============================================

def chunk_sv16(X_tune, y_tune, X_test, y_test):
    """Convert R script sv16 to Python: train SVM with fixed parameters and evaluate on test set"""
    
    if any(var is None for var in [X_tune, y_tune, X_test, y_test]):
        print("Required data not provided. Please run chunk_sv15 first.")
        return None, None
    
    # Train SVM with fixed parameters (C=1, radial kernel)
    tuned = SVC(kernel='rbf', C=1, gamma='scale')
    tuned.fit(X_tune, y_tune)
    
    # Make predictions on test set
    y_pred = tuned.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique labels for proper labeling
    labels = sorted(y_test.unique())
    
    print("Confusion Matrix:")
    print("Reference")
    print(f"Prediction   {labels[0]:>8} {labels[1]:>8}")
    for i, label in enumerate(labels):
        print(f"{label:>10}   {cm[i,0]:>8} {cm[i,1]:>8}")
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Additional metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(label) for label in labels]))
    
    return tuned, y_pred





########################################################################################################################


# ============================================
#              chunk_sv17
# ============================================

def chunk_sv17(X_tune, y_tune, X_test, y_test):
    """Convert R script sv17 to Python: SVM with probability estimates and ROC analysis"""
    
    if any(var is None for var in [X_tune, y_tune, X_test, y_test]):
        print("Required data not provided. Please run chunk_sv15 first.")
        return None, None, None
    
    # Train SVM with probability estimates enabled
    tuned2 = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    tuned2.fit(X_tune, y_tune)
    
    # Get probability predictions
    svm_probs = tuned2.predict_proba(X_test)
    
    # Extract probabilities for the positive class (assuming second class is positive)
    unique_labels = sorted(y_test.unique())
    if len(unique_labels) == 2:
        # Find which column corresponds to the positive class
        pos_class_idx = 1 if unique_labels[1] > unique_labels[0] else 0
        phat = svm_probs[:, pos_class_idx]
        
        # Convert labels to binary (0/1) for ROC calculation
        y_test_binary = (y_test == unique_labels[pos_class_idx]).astype(int)
    else:
        print("Error: This function is designed for binary classification")
        return None, None, None
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test_binary, phat)
    auc_score = auc(fpr, tpr)
    
    print(f"AUC: {auc_score:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    
    # Create color map for the curve
    n_points = len(fpr)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    # Plot ROC curve with color gradient
    for i in range(len(fpr)-1):
        plt.plot(fpr[i:i+2], tpr[i:i+2], color=colors[i], linewidth=2)
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc_score:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return tuned2, phat, auc_score






########################################################################################################################


# ============================================
#              chunk_sv18
# ============================================

def chunk_sv18(X_tune, y_tune, X_train, y_train, feature_names=None):
    """Convert R script sv18 to Python: SVM feature importance analysis"""
    
    if any(var is None for var in [X_tune, y_tune, X_train, y_train]):
        print("Required data not provided. Please run chunk_sv15 first.")
        return None, None
    
    # Train SVM model (equivalent to rminer's fit function)
    M = SVC(kernel='rbf', C=1, gamma='scale')
    M.fit(X_tune, y_tune)
    
    # Calculate feature importance using permutation importance
    # This is equivalent to rminer's Importance function
    print("Calculating feature importance (this may take a moment)...")
    svm_imp = permutation_importance(M, X_train, y_train, 
                                   n_repeats=10, random_state=42, 
                                   scoring='accuracy')
    
    # Get feature names
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': svm_imp.importances_mean,
        'Std': svm_imp.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance (top 10):")
    print(importance_df.head(10).to_string(index=False, float_format='%.4f'))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], 
             xerr=top_features['Std'], alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Permutation Importance')
    plt.title('SVM Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return M, importance_df