import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, classification_report





# ============================================
#               chunk_nn1
# ============================================

def chunk_nn1(n=200, seed=1):
    """Generate simulated data and create a scatter plot."""
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sin(12 * (x + 0.2)) / (x + 0.2) + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({'y': y, 'x': x})
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='grey', alpha=0.7)
    plt.title("Simulated data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    return df





####################################################################################################################


# ============================================
#               chunk_nn2
# ============================================

def chunk_nn2(x, y):
    """Fit polynomial regression (degree 3) and plot results."""
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))
    
    # Fit OLS model
    ols = LinearRegression()
    ols.fit(X_poly, y)
    y_pred = ols.predict(X_poly)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='grey', alpha=0.7)
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.title("Polynomial: M = 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    return ols, y_pred





####################################################################################################################



# ============================================
#               chunk_nn3
# ============================================

def chunk_nn3(x, y, ols_model):
    """Decompose polynomial regression into components and create visualization."""
    # Create polynomial features to get coefficients
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))
    
    # Extract coefficients (intercept + 3 polynomial terms)
    intercept = ols_model.intercept_
    coefs = ols_model.coef_
    
    # Calculate components
    first = coefs[0] * x
    second = coefs[1] * x**2
    third = coefs[2] * x**3
    yhat = intercept + first + second + third
    
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Fixed Components", fontsize=16, color="olivedrab", y=1.02)
    
    # Plot components
    axes[0].plot(x, first, color='pink')
    axes[0].set_title('x')
    axes[0].set_ylabel('y')
    
    axes[1].plot(x, second, color='orange')
    axes[1].set_title('x²')
    axes[1].set_ylabel('y')
    
    axes[2].plot(x, third, color='green')
    axes[2].set_title('x³')
    axes[2].set_ylabel('y')
    
    axes[3].scatter(x, y, color='grey', alpha=0.7)
    axes[3].plot(x, yhat, color='red', linewidth=3)
    axes[3].set_title('y = α + β₁x + β₂x² + β₃x³')
    axes[3].set_ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    return first, second, third, yhat





####################################################################################################################



# ============================================
#          chunk_ann_function
# ============================================

def chunk_ann_function(x, y=None, a=[1.5, 9, 3], b=[-20, -14, -8], beta=[15, 25, -40], intercept=3):
    """Artificial Neural Network function with 3 sigmoid units"""
    
    def ann(x_val, a, b, beta, intercept):
        # First sigmoid
        z1 = a[0] + b[0] * x_val
        sig1 = 1 / (1 + np.exp(-z1))
        f1 = sig1
        
        # Second sigmoid  
        z2 = a[1] + b[1] * x_val
        sig2 = 1 / (1 + np.exp(-z2))
        f2 = sig2
        
        # Third sigmoid
        z3 = a[2] + b[2] * x_val
        sig3 = 1 / (1 + np.exp(-z3))
        f3 = sig3
        
        # Final output
        yhat = intercept + beta[0] * f1 + beta[1] * f2 + beta[2] * f3
        return yhat
    
    # Generate predictions
    yhat = ann(x, a, b, beta, intercept)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot data points if y is provided
    if y is not None:
        plt.scatter(x, y, alpha=0.6, label='Data points')
    
    # Plot ANN predictions
    plt.plot(x, yhat, color='red', linewidth=3, label='ANN (M=3)')
    
    plt.title("ANN: M = 3", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-5, 15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return yhat





####################################################################################################################




# ============================================
#               chunk_nn4
# ============================================

def chunk_nn4(x, y, a=[1.5, 9, 3], b=[-20, -14, -8], beta=[15, 25, -40], intercept=3):
    """Artificial Neural Network with 3 sigmoid units."""
    
    def ann(x, a, b, beta, intercept):
        # 1st sigmoid
        z1 = a[0] + b[0] * x
        sig1 = 1 / (1 + np.exp(-z1))
        f1 = sig1
        
        # 2nd sigmoid
        z2 = a[1] + b[1] * x
        sig2 = 1 / (1 + np.exp(-z2))
        f2 = sig2
        
        # 3rd sigmoid
        z3 = a[2] + b[2] * x
        sig3 = 1 / (1 + np.exp(-z3))
        f3 = sig3
        
        yhat = intercept + beta[0] * f1 + beta[1] * f2 + beta[2] * f3
        return yhat
    
    yhat = ann(x, a, b, beta, intercept)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='grey', alpha=0.7)
    plt.plot(x, yhat, color='red', linewidth=3)
    plt.title("ANN: M = 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-5, 15)
    plt.show()
    
    return yhat





####################################################################################################################



# ============================================
#               chunk_nn5
# ============================================

def chunk_nn5(df, seed=2):
    """Train neural network with 3 hidden units and plot results."""
    np.random.seed(seed)
    
    # Extract x and y from dataframe
    x = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    
    # Create and train neural network
    nn = MLPRegressor(
        hidden_layer_sizes=(3,),
        activation='logistic',  # equivalent to sigmoid
        solver='lbfgs',
        max_iter=1000,
        random_state=seed,
        tol=0.05
    )
    
    nn.fit(x, y)
    yhat = nn.predict(x)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], color='grey', alpha=0.7)
    plt.plot(df['x'], yhat, color='red', linewidth=3)
    plt.title("Neural Networks: M = 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    return nn, yhat





####################################################################################################################



# ============================================
#               chunk_nn6
# ============================================

def chunk_nn6(nn_model):
    """Display neural network weights and architecture visualization."""
    
    # Display weights
    print("Neural Network Weights:")
    print("Input to Hidden Layer:")
    print(f"Weights: {nn_model.coefs_[0].flatten()}")
    print(f"Biases: {nn_model.intercepts_[0]}")
    print("\nHidden to Output Layer:")
    print(f"Weights: {nn_model.coefs_[1].flatten()}")
    print(f"Bias: {nn_model.intercepts_[1]}")
    
    # Create network diagram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Node positions
    input_pos = [(1, 2)]
    hidden_pos = [(3, i) for i in [1, 2, 3]]
    output_pos = [(5, 2)]
    
    # Draw nodes
    # Input node
    circle = patches.Circle(input_pos[0], 0.2, color='lightblue', ec='black')
    ax.add_patch(circle)
    ax.text(input_pos[0][0], input_pos[0][1], 'x', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Hidden nodes
    for i, pos in enumerate(hidden_pos):
        circle = patches.Circle(pos, 0.2, color='lightgreen', ec='black')
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f'H{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output node
    circle = patches.Circle(output_pos[0], 0.2, color='lightcoral', ec='black')
    ax.add_patch(circle)
    ax.text(output_pos[0][0], output_pos[0][1], 'y', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw connections with weight labels
    # Input to hidden
    for i, h_pos in enumerate(hidden_pos):
        ax.plot([input_pos[0][0], h_pos[0]], [input_pos[0][1], h_pos[1]], 'k-', alpha=0.7)
        # Weight label
        mid_x = (input_pos[0][0] + h_pos[0]) / 2
        mid_y = (input_pos[0][1] + h_pos[1]) / 2
        weight = nn_model.coefs_[0][0][i]
        ax.text(mid_x, mid_y + 0.1, f'{weight:.2f}', ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Hidden to output
    for i, h_pos in enumerate(hidden_pos):
        ax.plot([h_pos[0], output_pos[0][0]], [h_pos[1], output_pos[0][1]], 'k-', alpha=0.7)
        # Weight label
        mid_x = (h_pos[0] + output_pos[0][0]) / 2
        mid_y = (h_pos[1] + output_pos[0][1]) / 2
        weight = nn_model.coefs_[1][i][0]
        ax.text(mid_x, mid_y + 0.1, f'{weight:.2f}', ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Labels
    ax.text(1, 0.5, 'Input Layer', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3, 0.5, 'Hidden Layer (3 units)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 0.5, 'Output Layer', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return nn_model.coefs_, nn_model.intercepts_






####################################################################################################################


# ============================================
#               chunk_nn7
# ============================================

def chunk_nn7(x, nn_model):
    """Visualize individual hidden unit activations from neural network."""
    
    # Create design matrix with intercept
    X = np.column_stack([np.ones(len(x)), x])
    
    # Get weights from input to hidden layer
    input_to_hidden = nn_model.coefs_[0]  # shape: (1, 3)
    hidden_biases = nn_model.intercepts_[0]  # shape: (3,)
    
    # Calculate activations for each hidden unit
    # Node 1
    linear1 = X[:, 0] * hidden_biases[0] + X[:, 1] * input_to_hidden[0, 0]
    f1 = 1 / (1 + np.exp(-linear1))  # sigmoid activation
    
    # Node 2  
    linear2 = X[:, 0] * hidden_biases[1] + X[:, 1] * input_to_hidden[0, 1]
    f2 = 1 / (1 + np.exp(-linear2))  # sigmoid activation
    
    # Node 3
    linear3 = X[:, 0] * hidden_biases[2] + X[:, 1] * input_to_hidden[0, 2]
    f3 = 1 / (1 + np.exp(-linear3))  # sigmoid activation
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Flexible Components", fontsize=16, color="olivedrab", y=1.02)
    
    # Plot each activation function
    axes[0].plot(x, f1, color='pink', linewidth=2)
    axes[0].set_title('f(α₁ + β₁x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, f2, color='orange', linewidth=2)
    axes[1].set_title('f(α₂ + β₂x)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x, f3, color='green', linewidth=2)
    axes[2].set_title('f(α₃ + β₃x)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('f')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return f1, f2, f3






####################################################################################################################


# ============================================
#               chunk_nn8
# ============================================

def chunk_nn8(x, y, f1, f2, f3, nn_model):
    """Combine hidden unit outputs with final layer weights to create predictions."""
    
    # Get weights from hidden to output layer
    hidden_to_output = nn_model.coefs_[1]  # shape: (3, 1)
    output_bias = nn_model.intercepts_[1]  # shape: (1,)
    
    # Weight each hidden unit output
    f12 = f1 * hidden_to_output[0, 0]
    f22 = f2 * hidden_to_output[1, 0] 
    f32 = f3 * hidden_to_output[2, 0]
    
    # Final prediction: bias + weighted sum of hidden units
    yhat = output_bias[0] + f12 + f22 + f32
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='grey', alpha=0.7)
    plt.plot(x, yhat, color='red', linewidth=3)
    plt.title("ANN: M = 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    return yhat, f12, f22, f32







####################################################################################################################


# ============================================
#              chunk_nn7a
# ============================================

def chunk_nn7a(n=200, learning_rate=0.005, max_iterations=2000000, tolerance=1e-12):
    """Train neural network with gradient descent from scratch"""
    
    # Generate simulated data
    np.random.seed(1)
    x = np.sort(np.random.uniform(0, 1, n))
    Y = np.sin(12 * (x + 0.2)) / (x + 0.2) + np.random.normal(0, 0.5, n)
    
    # Plot initial data
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, Y, color='grey', alpha=0.7)
    plt.title("Simulated data")
    plt.xlabel("x")
    plt.ylabel("Y")
    
    # Initialize parameters
    np.random.seed(234)
    alpha = np.random.uniform(0, 1)
    beta1, beta2, beta3 = np.random.uniform(0, 1, 3)
    a1, a2, a3 = np.random.uniform(0, 1, 3)
    b1, b2, b3 = np.random.uniform(0, 1, 3)
    
    # Initial forward pass
    z1, z2, z3 = a1 + b1*x, a2 + b2*x, a3 + b3*x
    sig1 = 1/(1 + np.exp(-z1))
    sig2 = 1/(1 + np.exp(-z2))
    sig3 = 1/(1 + np.exp(-z3))
    yhat = alpha + beta1*sig1 + beta2*sig2 + beta3*sig3
    
    MSE = [np.sum((Y - yhat)**2) / n]
    converged = False
    iterations = 0
    
    # Gradient descent loop
    while not converged:
        # Compute gradients
        grad_alpha = -np.mean(Y - yhat)
        grad_beta1 = -np.mean((Y - yhat) * sig1)
        grad_beta2 = -np.mean((Y - yhat) * sig2)
        grad_beta3 = -np.mean((Y - yhat) * sig3)
        
        sig1_deriv = np.exp(-z1) / ((1 + np.exp(-z1))**2)
        sig2_deriv = np.exp(-z2) / ((1 + np.exp(-z2))**2)
        sig3_deriv = np.exp(-z3) / ((1 + np.exp(-z3))**2)
        
        grad_a1 = -np.mean((Y - yhat) * beta1 * sig1_deriv)
        grad_a2 = -np.mean((Y - yhat) * beta2 * sig2_deriv)
        grad_a3 = -np.mean((Y - yhat) * beta3 * sig3_deriv)
        grad_b1 = -np.mean((Y - yhat) * beta1 * x * sig1_deriv)
        grad_b2 = -np.mean((Y - yhat) * beta2 * x * sig2_deriv)
        grad_b3 = -np.mean((Y - yhat) * beta3 * x * sig3_deriv)
        
        # Update parameters
        alpha -= learning_rate * grad_alpha
        beta1 -= learning_rate * grad_beta1
        beta2 -= learning_rate * grad_beta2
        beta3 -= learning_rate * grad_beta3
        a1 -= learning_rate * grad_a1
        a2 -= learning_rate * grad_a2
        a3 -= learning_rate * grad_a3
        b1 -= learning_rate * grad_b1
        b2 -= learning_rate * grad_b2
        b3 -= learning_rate * grad_b3
        
        # Forward pass
        z1, z2, z3 = a1 + b1*x, a2 + b2*x, a3 + b3*x
        sig1 = 1/(1 + np.exp(-z1))
        sig2 = 1/(1 + np.exp(-z2))
        sig3 = 1/(1 + np.exp(-z3))
        yhat = alpha + beta1*sig1 + beta2*sig2 + beta3*sig3
        
        # Check convergence
        MSE_new = np.sum((Y - yhat)**2) / n
        MSE.append(MSE_new)
        
        if len(MSE) > 1:
            diff = abs(MSE[-1] - MSE[-2])
            if round(diff, 12) == 0 or iterations > max_iterations:
                converged = True
        
        iterations += 1
    
    print(f"Converged after {iterations} iterations")
    print(f"Final parameters:")
    print(f"Alpha, Beta: [{alpha:.4f}, {beta1:.4f}, {beta2:.4f}, {beta3:.4f}]")
    print(f"a values: [{a1:.4f}, {a2:.4f}, {a3:.4f}]")
    print(f"b values: [{b1:.4f}, {b2:.4f}, {b3:.4f}]")
    
    # Final plot
    plt.subplot(1, 2, 2)
    plt.scatter(x, Y, color='grey', alpha=0.7, label='Data')
    plt.plot(x, yhat, color='red', linewidth=3, label='Neural Network')
    plt.title("Fitted Neural Network")
    plt.xlabel("x")
    plt.ylabel("Y")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return x, Y, yhat, (alpha, beta1, beta2, beta3), (a1, a2, a3), (b1, b2, b3)




####################################################################################################################


# ============================================
#               chunk_nn9
# ============================================

def chunk_nn9(n=200, seed1=1, seed2=234, learning_rate=0.005, max_iterations=2000000):
    """Manual gradient descent training of neural network with 3 sigmoid units."""
    
    # Generate data
    np.random.seed(seed1)
    x = np.sort(np.random.uniform(0, 1, n))
    Y = np.sin(12 * (x + 0.2)) / (x + 0.2) + np.random.normal(0, 0.5, n)
    df = {'Y': Y, 'x': x}
    
    # Plot initial data
    plt.figure(figsize=(8, 6))
    plt.scatter(x, Y, color='grey', alpha=0.7)
    plt.title("Simulated data")
    plt.xlabel("x")
    plt.ylabel("Y")
    plt.show()
    
    # Initialize parameters
    np.random.seed(seed2)
    alpha = np.random.uniform(0, 1)
    beta1 = np.random.uniform(0, 1)
    beta2 = np.random.uniform(0, 1)
    beta3 = np.random.uniform(0, 1)
    a1 = np.random.uniform(0, 1)
    a2 = np.random.uniform(0, 1)
    a3 = np.random.uniform(0, 1)
    b1 = np.random.uniform(0, 1)
    b2 = np.random.uniform(0, 1)
    b3 = np.random.uniform(0, 1)
    
    # Initial forward pass
    z1 = a1 + b1 * x
    z2 = a2 + b2 * x
    z3 = a3 + b3 * x
    sig1 = 1 / (1 + np.exp(-z1))
    sig2 = 1 / (1 + np.exp(-z2))
    sig3 = 1 / (1 + np.exp(-z3))
    yhat = alpha + beta1 * sig1 + beta2 * sig2 + beta3 * sig3
    
    # Initialize MSE tracking
    MSE = [np.sum((Y - yhat) ** 2) / n]
    converged = False
    iterations = 0
    
    # Gradient descent loop
    while not converged:
        # Calculate gradients
        alpha_new = alpha - (learning_rate * (1/n) * np.sum((Y - yhat) * (-1)))
        beta1_new = beta1 - (learning_rate * (1/n) * np.sum((Y - yhat) * sig1 * (-1)))
        beta2_new = beta2 - (learning_rate * (1/n) * np.sum((Y - yhat) * sig2 * (-1)))
        beta3_new = beta3 - (learning_rate * (1/n) * np.sum((Y - yhat) * sig3 * (-1)))
        
        a1_new = a1 - (learning_rate * (1/n) * np.sum((Y - yhat) * (-beta1 * np.exp(-z1) / ((1 + np.exp(-z1))**2))))
        a2_new = a2 - (learning_rate * (1/n) * np.sum((Y - yhat) * (-beta2 * np.exp(-z2) / ((1 + np.exp(-z2))**2))))
        a3_new = a3 - (learning_rate * (1/n) * np.sum((Y - yhat) * (-beta3 * np.exp(-z3) / ((1 + np.exp(-z3))**2))))
        
        b1_new = b1 - (learning_rate * (1/n) * np.sum((Y - yhat) * (-beta1 * x * np.exp(-z1) / ((1 + np.exp(-z1))**2))))
        b2_new = b2 - (learning_rate * (1/n) * np.sum((Y - yhat) * (-beta2 * x * np.exp(-z2) / ((1 + np.exp(-z2))**2))))
        b3_new = b3 - (learning_rate * (1/n) * np.sum((Y - yhat) * (-beta3 * x * np.exp(-z3) / ((1 + np.exp(-z3))**2))))
        
        # Update parameters
        alpha, beta1, beta2, beta3 = alpha_new, beta1_new, beta2_new, beta3_new
        a1, a2, a3 = a1_new, a2_new, a3_new
        b1, b2, b3 = b1_new, b2_new, b3_new
        
        # Forward pass with new parameters
        z1 = a1 + b1 * x
        z2 = a2 + b2 * x
        z3 = a3 + b3 * x
        sig1 = 1 / (1 + np.exp(-z1))
        sig2 = 1 / (1 + np.exp(-z2))
        sig3 = 1 / (1 + np.exp(-z3))
        yhat = alpha + beta1 * sig1 + beta2 * sig2 + beta3 * sig3
        
        # Check convergence
        MSE_new = np.sum((Y - yhat) ** 2) / n
        MSE.append(MSE_new)
        
        if len(MSE) > 1:
            d = abs(MSE[-1] - MSE[-2])
            if round(d, 12) == 0:
                converged = True
        
        iterations += 1
        if iterations > max_iterations:
            converged = True
    
    # Print final parameters
    print(f"Final parameters after {iterations} iterations:")
    print(f"Output weights: [α={alpha:.6f}, β1={beta1:.6f}, β2={beta2:.6f}, β3={beta3:.6f}]")
    print(f"Hidden biases: [a1={a1:.6f}, a2={a2:.6f}, a3={a3:.6f}]")
    print(f"Hidden weights: [b1={b1:.6f}, b2={b2:.6f}, b3={b3:.6f}]")
    
    # Final plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, Y, color='grey', alpha=0.7)
    plt.plot(x, yhat, color='red', linewidth=3)
    plt.title("Simulated data")
    plt.xlabel("x")
    plt.ylabel("Y")
    plt.show()
    
    return {'x': x, 'Y': Y, 'yhat': yhat, 'parameters': {
        'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3,
        'a1': a1, 'a2': a2, 'a3': a3, 'b1': b1, 'b2': b2, 'b3': b3
    }, 'MSE_history': MSE, 'iterations': iterations}








####################################################################################################################



# ============================================
#              chunk_nn10
# ============================================

def chunk_nn10(n_iterations=10):
    """Compare regression models on CPS1985 wage data using bootstrap validation"""
    
    # Load CPS1985 dataset or create sample data
    try:
        # Try to load from OpenML or create realistic wage data
        data = fetch_openml(name='CPS1985', version=1, as_frame=True)
        df = data.frame
    except:
        # Create sample CPS1985-like wage data
        np.random.seed(42)
        n_samples = 534
        df = pd.DataFrame({
            'wage': np.random.exponential(7, n_samples) + np.random.normal(0, 2, n_samples),
            'education': np.random.randint(8, 19, n_samples),
            'experience': np.random.randint(0, 56, n_samples),
            'age': np.random.randint(18, 65, n_samples),
            'ethnicity': np.random.choice(['cauc', 'afam'], n_samples),
            'region': np.random.choice(['northeast', 'midwest', 'south', 'west'], n_samples),
            'gender': np.random.choice(['male', 'female'], n_samples),
            'occupation': np.random.choice(['management', 'sales', 'clerical', 'service', 'professional', 'other'], n_samples),
            'sector': np.random.choice(['manufacturing', 'construction', 'other'], n_samples),
            'union': np.random.choice(['yes', 'no'], n_samples)
        })
    
    print(f"Dataset shape: {df.shape}")
    
    # Scale numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Create dummy variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    ddf = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Prepare feature sets
    wage_col = 'wage'
    feature_cols = [col for col in ddf.columns if col != wage_col]
    
    # Add experience squared for linear model
    if 'experience' in ddf.columns:
        ddf['experience_squared'] = ddf['experience'] ** 2
        feature_cols_lm = feature_cols + ['experience_squared']
    else:
        feature_cols_lm = feature_cols
    
    X = ddf[feature_cols]
    X_lm = ddf[feature_cols_lm] if 'experience_squared' in ddf.columns else X
    y = ddf[wage_col]
    
    # Bootstrap validation
    mse_test = np.zeros((n_iterations, 5))
    
    for i in range(n_iterations):
        np.random.seed(i + 1)
        
        # Bootstrap sampling
        n_samples = len(ddf)
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        test_indices = list(set(range(n_samples)) - set(boot_indices))
        
        if len(test_indices) == 0:  # Fallback if no test samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=i
            )
            X_lm_train, X_lm_test = train_test_split(
                X_lm, test_size=0.3, random_state=i
            )[0:2]
        else:
            X_train, y_train = X.iloc[boot_indices], y.iloc[boot_indices]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
            X_lm_train, X_lm_test = X_lm.iloc[boot_indices], X_lm.iloc[test_indices]
        
        # Models
        # Linear regression with experience^2
        fit_lm = LinearRegression()
        fit_lm.fit(X_lm_train, y_train)
        pred_lm = fit_lm.predict(X_lm_test)
        
        # Neural networks with different architectures
        fit_nn1 = MLPRegressor(hidden_layer_sizes=(1,), max_iter=1000, random_state=i,
                              activation='logistic', solver='lbfgs', alpha=0.001)
        fit_nn1.fit(X_train, y_train)
        pred_nn1 = fit_nn1.predict(X_test)
        
        fit_nn2 = MLPRegressor(hidden_layer_sizes=(2,), max_iter=1000, random_state=i,
                              activation='relu', solver='adam', alpha=0.001)
        fit_nn2.fit(X_train, y_train)
        pred_nn2 = fit_nn2.predict(X_test)
        
        fit_nn3 = MLPRegressor(hidden_layer_sizes=(3,), max_iter=1000, random_state=i,
                              activation='relu', solver='adam', alpha=0.001)
        fit_nn3.fit(X_train, y_train)
        pred_nn3 = fit_nn3.predict(X_test)
        
        fit_nn4 = MLPRegressor(hidden_layer_sizes=(3,), max_iter=1000, random_state=i,
                              activation='tanh', solver='adam', alpha=0.001)
        fit_nn4.fit(X_train, y_train)
        pred_nn4 = fit_nn4.predict(X_test)
        
        # Calculate MSE
        mse_test[i, 0] = mean_squared_error(y_test, pred_lm)
        mse_test[i, 1] = mean_squared_error(y_test, pred_nn1)
        mse_test[i, 2] = mean_squared_error(y_test, pred_nn2)
        mse_test[i, 3] = mean_squared_error(y_test, pred_nn3)
        mse_test[i, 4] = mean_squared_error(y_test, pred_nn4)
    
    # Calculate mean MSE across all iterations
    mean_mse = np.mean(mse_test, axis=0)
    
    model_names = ['Linear Reg', 'NN (1 hidden)', 'NN (2 hidden)', 
                   'NN (3 hidden)', 'NN (3 hidden, tanh)']
    
    print("Mean MSE across models:")
    for name, mse in zip(model_names, mean_mse):
        print(f"{name}: {mse:.4f}")
    
    return mean_mse, mse_test







####################################################################################################################


# ============================================
#               chunk_nn11
# ============================================

def chunk_nn11(n_iterations=10):
    """Compare neural networks vs linear regression on CPS1985 wage data using bootstrap sampling."""
    
    # Load and prepare data (simulating CPS1985 dataset structure)
    # Note: Creating synthetic data that mimics CPS1985 structure
    np.random.seed(42)
    n_obs = 534  # Original CPS1985 size
    
    # Generate synthetic data with similar structure to CPS1985
    data = {
        'wage': np.random.lognormal(2.5, 0.6, n_obs),
        'education': np.random.randint(6, 19, n_obs),
        'experience': np.random.randint(0, 50, n_obs),
        'age': np.random.randint(18, 65, n_obs),
        'ethnicity': np.random.choice(['cauc', 'afam'], n_obs, p=[0.85, 0.15]),
        'region': np.random.choice(['south', 'northeast', 'west', 'midwest'], n_obs),
        'gender': np.random.choice(['male', 'female'], n_obs, p=[0.6, 0.4]),
        'occupation': np.random.choice(['worker', 'technical', 'services', 'office', 'sales', 'management'], n_obs),
        'sector': np.random.choice(['manufacturing', 'construction', 'other'], n_obs),
        'union': np.random.choice(['yes', 'no'], n_obs, p=[0.25, 0.75])
    }
    
    df = pd.DataFrame(data)
    
    # Scale numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Create dummy variables
    ddf = pd.get_dummies(df, drop_first=False)
    
    # Prepare feature matrices
    wage_col = 'wage'
    feature_cols = [col for col in ddf.columns if col != wage_col]
    
    # Add experience squared for linear model
    if 'experience' in ddf.columns:
        ddf['experience_squared'] = ddf['experience'] ** 2
        linear_features = feature_cols + ['experience_squared']
    else:
        linear_features = feature_cols
    
    # Bootstrap sampling and model comparison
    mse_test = np.zeros((n_iterations, 5))
    
    for i in range(n_iterations):
        np.random.seed(i + 1)
        
        # Bootstrap sampling (with replacement)
        train_idx = np.random.choice(len(ddf), size=len(ddf), replace=True)
        test_idx = np.setdiff1d(np.arange(len(ddf)), train_idx)
        
        if len(test_idx) == 0:  # Handle edge case
            test_idx = np.random.choice(len(ddf), size=max(1, len(ddf)//5), replace=False)
        
        train = ddf.iloc[train_idx]
        test = ddf.iloc[test_idx]
        
        X_train = train[feature_cols]
        y_train = train[wage_col]
        X_test = test[feature_cols]
        y_test = test[wage_col]
        
        X_train_linear = train[linear_features] if 'experience_squared' in ddf.columns else X_train
        X_test_linear = test[linear_features] if 'experience_squared' in ddf.columns else X_test
        
        try:
            # Model 1: Linear Regression with experience^2
            fit_lm = LinearRegression()
            fit_lm.fit(X_train_linear, y_train)
            pred_lm = fit_lm.predict(X_test_linear)
            mse_test[i, 0] = mean_squared_error(y_test, pred_lm)
            
            # Model 2: Neural Network - 1 hidden unit, sigmoid
            fit_nn1 = MLPRegressor(hidden_layer_sizes=(1,), activation='logistic', 
                                   solver='lbfgs', max_iter=1000, random_state=i+1, tol=0.05)
            fit_nn1.fit(X_train, y_train)
            pred_nn1 = fit_nn1.predict(X_test)
            mse_test[i, 1] = mean_squared_error(y_test, pred_nn1)
            
            # Model 3: Neural Network - 2 hidden units
            fit_nn2 = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic', 
                                   solver='lbfgs', max_iter=1000, random_state=i+1, tol=0.05)
            fit_nn2.fit(X_train, y_train)
            pred_nn2 = fit_nn2.predict(X_test)
            mse_test[i, 2] = mean_squared_error(y_test, pred_nn2)
            
            # Model 4: Neural Network - 3 hidden units
            fit_nn3 = MLPRegressor(hidden_layer_sizes=(3,), activation='logistic', 
                                   solver='lbfgs', max_iter=1000, random_state=i+1, tol=0.05)
            fit_nn3.fit(X_train, y_train)
            pred_nn3 = fit_nn3.predict(X_test)
            mse_test[i, 3] = mean_squared_error(y_test, pred_nn3)
            
            # Model 5: Neural Network - 3 hidden units, tanh activation
            fit_nn4 = MLPRegressor(hidden_layer_sizes=(3,), activation='tanh', 
                                   solver='lbfgs', max_iter=1000, random_state=i+1, tol=0.05)
            fit_nn4.fit(X_train, y_train)
            pred_nn4 = fit_nn4.predict(X_test)
            mse_test[i, 4] = mean_squared_error(y_test, pred_nn4)
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            # Fill with NaN for failed iterations
            mse_test[i, :] = np.nan
    
    # Calculate mean MSE across iterations (ignoring NaN values)
    mean_mse = np.nanmean(mse_test, axis=0)
    
    print("Model Comparison Results:")
    print("=" * 50)
    print(f"Linear Regression (with experience²): {mean_mse[0]:.6f}")
    print(f"Neural Network (1 hidden unit):       {mean_mse[1]:.6f}")
    print(f"Neural Network (2 hidden units):      {mean_mse[2]:.6f}")
    print(f"Neural Network (3 hidden units):      {mean_mse[3]:.6f}")
    print(f"Neural Network (3 units, tanh):       {mean_mse[4]:.6f}")
    
    return {
        'mse_matrix': mse_test,
        'mean_mse': mean_mse,
        'data': ddf,
        'feature_columns': feature_cols
    }






####################################################################################################################


# ============================================================
#                        chunk_nn12
# ============================================================

def chunk_nn12(fit_nn22):
    """
    Plot neural network with best repetition.
    
    Args:
        fit_nn22: Fitted neural network object with plot capability
    """
    fit_nn22.plot(rep="best")
    plt.show()







####################################################################################################################


# ============================================
#              chunk_nn13
# ============================================

def chunk_nn13(fit_nn22, save_fig=False, filename="plotnet.png"):
    """Plot neural network architecture using network visualization"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if hasattr(fit_nn22, 'coefs_'):
        # For sklearn neural networks
        layer_sizes = [fit_nn22.coefs_[0].shape[0]]
        layer_sizes.extend([layer.shape[1] for layer in fit_nn22.coefs_])
        
        # Calculate positions
        max_neurons = max(layer_sizes)
        layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
        
        node_positions = {}
        
        # Draw nodes
        for i, (layer_pos, layer_size) in enumerate(zip(layer_positions, layer_sizes)):
            y_positions = np.linspace(0.1, 0.9, layer_size)
            for j, y_pos in enumerate(y_positions):
                circle = Circle((layer_pos, y_pos), 0.02, color='lightblue', 
                              ec='black', linewidth=1)
                ax.add_patch(circle)
                node_positions[(i, j)] = (layer_pos, y_pos)
        
        # Draw connections
        for layer_idx in range(len(fit_nn22.coefs_)):
            weights = fit_nn22.coefs_[layer_idx]
            max_weight = np.abs(weights).max()
            
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    if (layer_idx, i) in node_positions and (layer_idx + 1, j) in node_positions:
                        x1, y1 = node_positions[(layer_idx, i)]
                        x2, y2 = node_positions[(layer_idx + 1, j)]
                        
                        alpha = min(0.8, abs(weights[i, j]) / max_weight)
                        color = 'red' if weights[i, j] < 0 else 'blue'
                        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=16, pad=20)
    
    # Add legend
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative weights')
    blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Positive weights')
    ax.legend(handles=[blue_patch, red_patch], loc='upper right')
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()







####################################################################################################################


# ============================================
#              chunk_nn14
# ============================================

def chunk_nn14():
    """Prepare Carseats dataset with scaling and dummy coding for neural network"""
    
    # Load Carseats dataset (using sklearn's fetch or create sample data)
    try:
        # Try to load from OpenML or create sample data
        data = fetch_openml(name='carseats', version=1, as_frame=True)
        df = data.frame
    except:
        # Create sample Carseats-like data if not available
        np.random.seed(42)
        n_samples = 400
        df = pd.DataFrame({
            'Sales': np.random.uniform(1, 16, n_samples),
            'CompPrice': np.random.uniform(77, 175, n_samples),
            'Income': np.random.uniform(21, 120, n_samples),
            'Advertising': np.random.uniform(0, 29, n_samples),
            'Population': np.random.uniform(10, 509, n_samples),
            'Price': np.random.uniform(24, 191, n_samples),
            'ShelveLoc': np.random.choice(['Bad', 'Good', 'Medium'], n_samples),
            'Age': np.random.uniform(25, 80, n_samples),
            'Education': np.random.uniform(10, 18, n_samples),
            'Urban': np.random.choice(['No', 'Yes'], n_samples),
            'US': np.random.choice(['No', 'Yes'], n_samples)
        })
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Change Sales to binary factor variable
    df['Sales'] = (df['Sales'] > 8).astype(int)
    dff = df.drop(columns=['Sales'])
    
    # Scaling numeric variables
    numeric_cols = dff.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    dff[numeric_cols] = scaler.fit_transform(dff[numeric_cols])
    
    # Dummy coding for categorical variables
    categorical_cols = dff.select_dtypes(include=['object', 'category']).columns
    ddf = pd.get_dummies(dff, columns=categorical_cols, drop_first=False)
    
    # Add Sales back as first column
    ddf = pd.concat([df[['Sales']], ddf], axis=1)
    
    # Create feature list for formula (all columns except Sales)
    feature_cols = [col for col in ddf.columns if col != 'Sales']
    
    return ddf, feature_cols







####################################################################################################################



# ============================================
#              chunk_nn15
# ============================================

def chunk_nn15(ddf, feature_cols, n_iterations=10):
    """Compare AUC performance between logistic regression and neural network"""
    
    np.random.seed(42)
    AUC1 = []  # Logistic regression AUC
    AUC2 = []  # Neural network AUC
    
    X = ddf[feature_cols]
    y = ddf['Sales']
    
    for i in range(n_iterations):
        # Bootstrap sampling with replacement
        np.random.seed(i)
        n_samples = len(ddf)
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        test_indices = list(set(range(n_samples)) - set(boot_indices))
        
        if len(test_indices) == 0:  # Fallback if no test samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=i, stratify=y)
        else:
            X_train, y_train = X.iloc[boot_indices], y.iloc[boot_indices]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        
        # Logistic regression model
        fit_ln = LogisticRegression(random_state=i, max_iter=1000)
        fit_ln.fit(X_train, y_train)
        phat_ln = fit_ln.predict_proba(X_test)[:, 1]
        
        # Neural network model (hidden layer with 2 neurons)
        fit_dnn = MLPClassifier(hidden_layer_sizes=(2,), random_state=i,
                               max_iter=1000, solver='lbfgs', alpha=0.01)
        fit_dnn.fit(X_train, y_train)
        phat_dnn = fit_dnn.predict_proba(X_test)[:, 1]
        
        # Calculate AUC scores
        if len(np.unique(y_test)) > 1:  # Ensure both classes present
            AUC1.append(roc_auc_score(y_test, phat_ln))
            AUC2.append(roc_auc_score(y_test, phat_dnn))
    
    mean_auc_lr = np.mean(AUC1) if AUC1 else 0
    mean_auc_nn = np.mean(AUC2) if AUC2 else 0
    
    print(f"Mean AUC - Logistic Regression: {mean_auc_lr:.4f}")
    print(f"Mean AUC - Neural Network: {mean_auc_nn:.4f}")
    
    return mean_auc_lr, mean_auc_nn








####################################################################################################################


# ============================================
#              chunk_nn16
# ============================================

def chunk_nn16(file_path="wineQualityReds.csv"):
    """Load and preprocess wine quality dataset, removing outlier qualities"""
    
    try:
        # Try to load from CSV file
        dfr = pd.read_csv(file_path)
        if dfr.columns[0].lower() in ['unnamed: 0', 'index']:
            dfr = dfr.iloc[:, 1:]  # Remove index column
    except FileNotFoundError:
        # Create sample wine quality data if file not found
        print("CSV file not found. Creating sample wine quality dataset...")
        np.random.seed(42)
        n_samples = 1599
        dfr = pd.DataFrame({
            'fixed.acidity': np.random.uniform(4.6, 15.9, n_samples),
            'volatile.acidity': np.random.uniform(0.12, 1.58, n_samples),
            'citric.acid': np.random.uniform(0, 1, n_samples),
            'residual.sugar': np.random.uniform(0.9, 15.5, n_samples),
            'chlorides': np.random.uniform(0.012, 0.611, n_samples),
            'free.sulfur.dioxide': np.random.uniform(1, 72, n_samples),
            'total.sulfur.dioxide': np.random.uniform(6, 289, n_samples),
            'density': np.random.uniform(0.99007, 1.00369, n_samples),
            'pH': np.random.uniform(2.74, 4.01, n_samples),
            'sulphates': np.random.uniform(0.33, 2.0, n_samples),
            'alcohol': np.random.uniform(8.4, 14.9, n_samples),
            'quality': np.random.choice([3, 4, 5, 6, 7, 8], n_samples, 
                                      p=[0.02, 0.1, 0.43, 0.4, 0.04, 0.01])
        })
    
    print("Original quality distribution:")
    print(dfr['quality'].value_counts().sort_index())
    
    # Remove outlier qualities (3 and 8)
    outlier_mask = (dfr['quality'] == 3) | (dfr['quality'] == 8)
    dfr = dfr[~outlier_mask].reset_index(drop=True)
    
    # Convert quality to categorical
    dfr['quality'] = dfr['quality'].astype('category')
    
    print("\nQuality distribution after removing outliers:")
    print(dfr['quality'].value_counts().sort_index())
    
    return dfr








####################################################################################################################

# ============================================
#              chunk_nn17
# ============================================

def chunk_nn17(dfr):
    """Scale numeric features and create dummy variables for quality"""
    
    # Scale numeric variables (all columns except quality)
    numeric_cols = dfr.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop('quality', errors='ignore')
    
    dfr_scaled = dfr.copy()
    scaler = StandardScaler()
    dfr_scaled[numeric_cols] = scaler.fit_transform(dfr[numeric_cols])
    
    # Create dummy variables for quality (one-hot encoding)
    quality_dummies = pd.get_dummies(dfr_scaled['quality'], prefix='quality', dtype=int)
    
    # Remove quality from original dataframe
    dfr_features = dfr_scaled.drop(columns=['quality'])
    
    # Combine dummy variables with scaled features
    df = pd.concat([quality_dummies, dfr_features], axis=1)
    
    # Create feature lists for formula representation
    target_cols = list(quality_dummies.columns)
    feature_cols = list(dfr_features.columns)
    
    # Display formula representation
    target_str = ' + '.join(target_cols)
    feature_str = ' + '.join(feature_cols)
    formula_str = f"{target_str} ~ {feature_str}"
    
    print("Formula representation:")
    print(formula_str)
    
    return df, target_cols, feature_cols








####################################################################################################################



# ============================================
#              chunk_nn18
# ============================================

def chunk_nn18(df, target_cols, feature_cols, test_size=0.3, random_state=42):
    """Train neural network and visualize architecture"""
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_cols].idxmax(axis=1)  # Convert one-hot to single labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train neural network with hidden layers (3, 2)
    fit_nn = MLPClassifier(
        hidden_layer_sizes=(3, 2),
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs',
        alpha=0.01
    )
    
    fit_nn.fit(X_train, y_train)
    
    # Plot neural network architecture
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define layer sizes including input and output
    layer_sizes = [X_train.shape[1]] + list(fit_nn.hidden_layer_sizes) + [len(target_cols)]
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(fit_nn.hidden_layer_sizes))] + ['Output']
    
    # Calculate positions
    layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
    node_positions = {}
    
    # Draw nodes
    for i, (layer_pos, layer_size, layer_name) in enumerate(zip(layer_positions, layer_sizes, layer_names)):
        y_positions = np.linspace(0.15, 0.85, layer_size)
        for j, y_pos in enumerate(y_positions):
            color = 'lightgreen' if i == 0 else 'lightcoral' if i == len(layer_sizes)-1 else 'lightblue'
            circle = Circle((layer_pos, y_pos), 0.02, color=color, ec='black', linewidth=1)
            ax.add_patch(circle)
            node_positions[(i, j)] = (layer_pos, y_pos)
        
        # Add layer labels
        ax.text(layer_pos, 0.05, f"{layer_name}\n({layer_size})", 
               ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw connections with weights
    for layer_idx in range(len(fit_nn.coefs_)):
        weights = fit_nn.coefs_[layer_idx]
        max_weight = np.abs(weights).max()
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if (layer_idx, i) in node_positions and (layer_idx + 1, j) in node_positions:
                    x1, y1 = node_positions[(layer_idx, i)]
                    x2, y2 = node_positions[(layer_idx + 1, j)]
                    
                    alpha = min(0.8, abs(weights[i, j]) / max_weight)
                    color = 'red' if weights[i, j] < 0 else 'blue'
                    linewidth = max(0.3, alpha * 2)
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture (3, 2 Hidden Layers)', fontsize=16, pad=20)
    
    # Add legend
    green_patch = mpatches.Patch(color='lightgreen', label='Input Layer')
    blue_patch = mpatches.Patch(color='lightblue', label='Hidden Layers')
    red_patch = mpatches.Patch(color='lightcoral', label='Output Layer')
    ax.legend(handles=[green_patch, blue_patch, red_patch], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fit_nn, X_train, X_test, y_train, y_test








####################################################################################################################



# ============================================
#              chunk_nn19
# ============================================

def chunk_nn19(fit_nn, X_test, y_test, target_cols):
    """Generate predictions and confusion matrix for neural network model"""
    
    # Get prediction probabilities
    phat = fit_nn.predict_proba(X_test)
    print("Prediction probabilities (first 6 rows):")
    print(pd.DataFrame(phat[:6], columns=fit_nn.classes_).round(4))
    
    # Create binary label matrix (one-hot style)
    label_hat = np.zeros_like(phat)
    max_indices = np.argmax(phat, axis=1)
    label_hat[np.arange(len(phat)), max_indices] = 1
    
    print("\nBinary label assignments (first 6 rows):")
    print(pd.DataFrame(label_hat[:6], columns=fit_nn.classes_).astype(int))
    
    # Get predicted and actual class labels
    predicted = fit_nn.predict(X_test)
    actual = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted, labels=fit_nn.classes_)
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=fit_nn.classes_, columns=fit_nn.classes_)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    print(cm_df)
    
    # Additional metrics
    print("\nClassification Report:")
    print(classification_report(actual, predicted))
    
    return phat, label_hat, predicted, actual