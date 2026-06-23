import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc



# ============================================
#              chunk_tc1
# ============================================

def chunk_tc1(img_path="png/confusion.png", width_scale=0.8, height_scale=0.8):
    """Display image with specified scaling similar to R knitr::include_graphics"""
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(8*width_scale, 6*height_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()





############################################################################################################################




# ============================================
#              chunk_tc2
# ============================================

def chunk_tc2(img_path="png/ROC1.png", width_scale=0.7, height_scale=0.7):
    """Display ROC image with specified scaling similar to R knitr::include_graphics"""
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(8*width_scale, 6*height_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()






############################################################################################################################




warnings.filterwarnings('ignore')

# ============================================
#              chunk_tc3
# ============================================

def chunk_tc3():
    """Create binary classification model from Boston housing data"""
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    
    # Create binary outcome (medv > 25)
    dummy = (boston.target > 25).astype(int)
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(data, dummy)
    
    # Print model summary
    print(f"Logistic Regression Summary:")
    print(f"Features: {len(boston.feature_names)}")
    print(f"Accuracy: {model.score(data, dummy):.4f}")
    
    return model, data, dummy






############################################################################################################################




# ============================================
#              chunk_tc4
# ============================================

def chunk_tc4(model, data, dummy):
    """Create and reorder confusion matrix from logistic regression predictions"""
    # Get predictions (threshold > 0.5)
    yHat = (model.predict_proba(data)[:, 1] > 0.5).astype(int)
    
    # Create confusion matrix and reorder cells
    cm = confusion_matrix(dummy, yHat)
    ct = np.array([[cm[1,1], cm[1,0]], 
                   [cm[0,1], cm[0,0]]])
    
    # Create labeled DataFrame
    ct_df = pd.DataFrame(ct, 
                        index=["Yhat = 1", "Yhat = 0"],
                        columns=["Y = 1", "Y = 0"])
    
    print(ct_df)
    return ct_df





############################################################################################################################



# ============================================
#              chunk_tc5
# ============================================

def chunk_tc5(model, data, dummy):
    """Create confusion matrix using rotation function equivalent"""
    # Get predictions
    yHat = (model.predict_proba(data)[:, 1] > 0.5).astype(int)
    
    # Create confusion matrix
    conf_table = confusion_matrix(yHat, dummy)
    
    # Apply rotation (equivalent to R rot function)
    def rot(x):
        t = np.flip(x, axis=0)
        tt = np.flip(t, axis=1)
        return tt.T
    
    ct = rot(conf_table)
    
    # Create labeled DataFrame
    ct_df = pd.DataFrame(ct,
                        index=["Yhat = 1", "Yhat = 0"],
                        columns=["Y = 1", "Y = 0"])
    
    print(ct_df)
    return ct_df






############################################################################################################################



# ============================================
#              chunk_tc6
# ============================================

def chunk_tc6(ct):
    """Calculate TPR, FPR, and J-Index from confusion matrix"""
    # TPR (True Positive Rate / Sensitivity)
    TPR = ct.iloc[0,0] / (ct.iloc[0,0] + ct.iloc[1,0])
    print(f"TPR: {TPR:.6f}")
    
    # FPR (False Positive Rate)
    FPR = ct.iloc[0,1] / (ct.iloc[0,1] + ct.iloc[1,1])
    print(f"FPR: {FPR:.6f}")
    
    # J-Index (Youden's Index)
    J_index = TPR - FPR
    print(f"J-Index: {J_index:.6f}")
    
    return TPR, FPR, J_index






############################################################################################################################



# ============================================
#              chunk_tc7
# ============================================

def chunk_tc7(model, data, dummy):
    """Generate ROC curve from model predictions using threshold sweep"""
    # Get fitted values and create ordered grid
    fitted_values = model.predict_proba(data)[:, 1]
    phat = np.sort(fitted_values)
    
    # Containers for TPR and FPR
    TPR = []
    FPR = []
    
    # Loop through thresholds
    for i in range(len(phat)):
        yHat = (fitted_values > phat[i]).astype(int)
        ct = confusion_matrix(dummy, yHat)
        
        # Check if matrix has both classes (sum of dimensions > 3)
        if ct.shape[0] == 2 and ct.shape[1] == 2:
            tpr = ct[1,1] / (ct[1,1] + ct[1,0]) if (ct[1,1] + ct[1,0]) > 0 else 0
            fpr = ct[0,1] / (ct[0,0] + ct[0,1]) if (ct[0,0] + ct[0,1]) > 0 else 0
            TPR.append(tpr)
            FPR.append(fpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPR, TPR, color='blue', linewidth=3, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='-', label='Random')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return np.array(FPR), np.array(TPR)






############################################################################################################################



# ============================================
#              chunk_tc8
# ============================================

def chunk_tc8(model, data, FPR, TPR):
    """Find optimal threshold using Youden's J statistic"""
    # Get sorted fitted values
    fitted_values = model.predict_proba(data)[:, 1]
    phat = np.sort(fitted_values)
    
    # Calculate Youden's J statistics
    J = np.array(TPR) - np.array(FPR)
    
    # Find best discriminating threshold
    best_idx = np.argmax(J)
    best_threshold = phat[best_idx] if best_idx < len(phat) else phat[-1]
    
    print(f"Best threshold: {best_threshold:.6f}")
    print(f"TPR at best threshold: {TPR[best_idx]:.6f}")
    print(f"FPR at best threshold: {FPR[best_idx]:.6f}")
    print(f"J-statistic at best threshold: {J[best_idx]:.6f}")
    
    return best_threshold, TPR[best_idx], FPR[best_idx], J[best_idx]






############################################################################################################################


# ============================================
#              chunk_tc9
# ============================================

def chunk_tc9(img_path="png/AUC.png", width_scale=0.7, height_scale=0.7):
    """Display AUC image with specified scaling similar to R knitr::include_graphics"""
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(8*width_scale, 6*height_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()






############################################################################################################################



warnings.filterwarnings('ignore')

# ============================================
#              chunk_tc10
# ============================================

def chunk_tc10():
    """Create ROC curve with AUC using sklearn metrics (ROCR equivalent)"""
    # Load data and create model
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    dummy = (boston.target > 25).astype(int)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(data, dummy)
    phat = model.predict_proba(data)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(dummy, phat)
    roc_auc = auc(fpr, tpr)
    
    # Create colorized ROC plot
    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fpr)))
    for i in range(len(fpr)-1):
        plt.plot(fpr[i:i+2], tpr[i:i+2], color=colors[i], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"AUC: {roc_auc:.6f}")
    return roc_auc







############################################################################################################################




# ============================================
#              chunk_tc11
# ============================================

def chunk_tc11(img_path="png/AUCs.png", width_scale=0.7, height_scale=0.7):
    """Display AUCs image with specified scaling similar to R knitr::include_graphics"""
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(8*width_scale, 6*height_scale))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()