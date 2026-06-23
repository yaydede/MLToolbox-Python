import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ============================================
#              chunk_tc12
# ============================================

def chunk_tc12(file_path="adult_train.csv"):
    """
    Download and read adult income data with proper column names.
    
    Args:
        file_path (str): Path to the adult training CSV file
        
    Returns:
        pd.DataFrame: Adult income dataset with named columns
    """
    # Read the training set into memory
    df = pd.read_csv(file_path, header=None)
    
    var_names = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
                 "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
                 "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", 
                 "IncomeLevel"]
    
    df.columns = var_names
    
    return df





#####################################################################################################################



# ============================================
#              chunk_tc13
# ============================================

def chunk_tc13(data):
    """
    Analyze income level distribution and calculate ratio.
    
    Args:
        data (pd.DataFrame): DataFrame with IncomeLevel column
        
    Returns:
        tuple: (value_counts, ratio) where ratio is second category / first category
    """
    tbl = data['IncomeLevel'].value_counts().sort_index()
    ratio = tbl.iloc[1] / tbl.iloc[0]
    
    return tbl, ratio





#####################################################################################################################


# ============================================
#              chunk_tc14
# ============================================

def chunk_tc14(data):
    """
    Display data structure and frequency tables for WorkClass and NativeCountry.
    
    Args:
        data (pd.DataFrame): DataFrame to analyze
        
    Returns:
        dict: Dictionary containing data info and frequency tables
    """
    # Get data structure info
    info = {
        'shape': data.shape,
        'dtypes': data.dtypes,
        'work_class_counts': data['WorkClass'].value_counts(),
        'native_country_counts': data['NativeCountry'].value_counts()
    }
    
    return info






#####################################################################################################################



# ============================================
#              chunk_tc15
# ============================================

def chunk_tc15(data):
    """
    Remove rows where NativeCountry is ' Holand-Netherlands'.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with specified rows removed
    """
    filtered_data = data[data['NativeCountry'] != ' Holand-Netherlands'].copy()
    
    return filtered_data






#####################################################################################################################



# ============================================
#              chunk_tc16
# ============================================

def chunk_tc16(data):
    """
    Convert character/object columns to categorical type.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with object columns converted to categorical
    """
    df = data.copy()
    
    # Convert object columns to category
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('category')
    
    return df




#####################################################################################################################



# ============================================
#              chunk_tc17
# ============================================

def chunk_tc17(data):
    """
    Convert income level data preparation for LPM analysis.
    Checks for missing values, creates binary outcome variable, and removes original column.
    """
    # Check for any missing values
    has_na = data.isnull().any().any()
    if has_na:
        print("Data contains missing values")
    
    # Create binary outcome variable for LPM
    data['Y'] = (data['IncomeLevel'] != ' <=50K').astype(int)
    
    # Remove the original IncomeLevel column (assuming it's column 15, index 14)
    data = data.drop(data.columns[14], axis=1)
    
    return data






#####################################################################################################################



# ============================================
#              chunk_tc18
# ============================================

def chunk_tc18(data, t=100, k=5):
    """
    Perform cross-validation with Linear Probability Model (LPM) and calculate AUC scores.
    Loops t times with different random seeds, splits data into k-fold CV, and plots results.
    """
    AUC = []
    
    for i in range(1, t + 1):
        np.random.seed(i)
        
        # Create shuffled indices for k-fold split
        n_samples = len(data)
        shuffled_indices = np.random.permutation(n_samples)
        test_size = n_samples // k
        
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]
        
        # Split data
        trdf = data.iloc[train_indices]
        tsdf = data.iloc[test_indices]
        
        # Prepare features and target
        X_train = trdf.drop('Y', axis=1)
        y_train = trdf['Y']
        X_test = tsdf.drop('Y', axis=1)
        y_test = tsdf['Y']
        
        # Linear Probability Model (LPM) using Linear Regression
        model1 = LinearRegression()
        model1.fit(X_train, y_train)
        phat = model1.predict(X_test)
        
        # Clip predictions to [0, 1]
        phat = np.clip(phat, 0, 1)
        
        # Calculate AUC
        auc_score = roc_auc_score(y_test, phat)
        AUC.append(auc_score)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(AUC, color='grey', alpha=0.7)
    plt.axhline(y=np.mean(AUC), color='red', linestyle='-')
    plt.title('AUC Scores Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.show()
    
    mean_auc = np.mean(AUC)
    std_auc = np.sqrt(np.var(AUC))
    
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Standard Deviation: {std_auc:.4f}")
    
    return AUC, mean_auc, std_auc





#####################################################################################################################



# ============================================
#              chunk_tc19
# ============================================

def chunk_tc19(phat, y_test, plot_both=True):
    """
    Generate ROC curves using both sklearn and manual calculation methods.
    Creates colorized ROC plot and manual ROC calculation for comparison.
    """
    
    if plot_both:
        # ROC using sklearn (equivalent to ROCR)
        fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, phat)
        roc_auc = auc(fpr_sklearn, tpr_sklearn)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: sklearn ROC (colorized equivalent)
        plt.subplot(1, 2, 1)
        plt.plot(fpr_sklearn, tpr_sklearn, color='blue', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (sklearn)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    # Manual ROC calculation
    phat_clipped = np.clip(phat, 0, 1)
    phator = np.unique(np.sort(phat_clipped))
    
    TPR = []
    FPR = []
    
    for threshold in phator:
        yHat = (phat > threshold).astype(int)
        
        # Create confusion matrix
        tn = np.sum((yHat == 0) & (y_test == 0))
        fp = np.sum((yHat == 1) & (y_test == 0))
        fn = np.sum((yHat == 0) & (y_test == 1))
        tp = np.sum((yHat == 1) & (y_test == 1))
        
        # Calculate TPR and FPR if denominators are non-zero
        if (tp + fn) > 0 and (tn + fp) > 0:
            tpr = tp / (tp + fn)
            fpr = fp / (tn + fp)
            TPR.append(tpr)
            FPR.append(fpr)
    
    # Plot manual ROC
    if plot_both:
        plt.subplot(1, 2, 2)
    else:
        plt.figure(figsize=(8, 6))
    
    plt.plot(FPR, TPR, color='blue', linewidth=2, label='Manual ROC')
    plt.plot([0, 1], [0, 1], color='red', linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (Manual Calculation)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return FPR, TPR





#####################################################################################################################



# ============================================
#              chunk_tc20
# ============================================

def chunk_tc20(TPR, FPR, phator):
    """
    Calculate Youden's J statistic and find optimal discrimination threshold.
    Returns optimal threshold and corresponding TPR, FPR, and J values.
    """
    # Calculate Youden's J statistic
    J = np.array(TPR) - np.array(FPR)
    
    # Find the index of maximum J
    max_j_idx = np.argmax(J)
    
    # Get optimal threshold and corresponding metrics
    opt_th = phator[max_j_idx]
    opt_tpr = TPR[max_j_idx]
    opt_fpr = FPR[max_j_idx]
    max_j = J[max_j_idx]
    
    print(f"Optimal threshold: {opt_th:.4f}")
    print(f"TPR at optimal threshold: {opt_tpr:.4f}")
    print(f"FPR at optimal threshold: {opt_fpr:.4f}")
    print(f"Youden's J statistic: {max_j:.4f}")
    
    return opt_th, opt_tpr, opt_fpr, max_j





#####################################################################################################################


# ============================================
#              chunk_tc21
# ============================================

def chunk_tc21(phat, y_test, opt_th):
    """
    Generate confusion matrix using optimal threshold with proper formatting.
    Creates a rotated confusion matrix with descriptive labels.
    """
    # Generate predictions using optimal threshold
    yHat = (phat > opt_th).astype(int)
    
    # Create confusion matrix manually
    tn = np.sum((yHat == 0) & (y_test == 0))
    fp = np.sum((yHat == 1) & (y_test == 0))
    fn = np.sum((yHat == 0) & (y_test == 1))
    tp = np.sum((yHat == 1) & (y_test == 1))
    
    # Create rotated confusion matrix (equivalent to R's rot function)
    ct = np.array([[tp, fp],
                   [fn, tn]])
    
    # Convert to DataFrame with proper labels
    ct_df = pd.DataFrame(ct, 
                        index=["Yhat = 1", "Yhat = 0"],
                        columns=["Y = 1", "Y = 0"])
    
    print("Confusion Matrix:")
    print(ct_df)
    
    return ct_df





#####################################################################################################################




# ============================================
#              chunk_tc22
# ============================================

def chunk_tc22(data, t=100, k=5):
    """
    Perform cross-validation with Logistic Regression and calculate AUC scores.
    Loops t times with different random seeds, splits data into k-fold CV, and plots results.
    """
    AUC = []
    
    for i in range(1, t + 1):
        np.random.seed(i)
        
        # Create shuffled indices for k-fold split
        n_samples = len(data)
        shuffled_indices = np.random.permutation(n_samples)
        test_size = n_samples // k
        
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]
        
        # Split data
        trdf = data.iloc[train_indices]
        tsdf = data.iloc[test_indices]
        
        # Prepare features and target
        X_train = trdf.drop('Y', axis=1)
        y_train = trdf['Y']
        X_test = tsdf.drop('Y', axis=1)
        y_test = tsdf['Y']
        
        # Logistic Regression
        model2 = LogisticRegression(max_iter=1000, random_state=i)
        model2.fit(X_train, y_train)
        
        # Predict probabilities (equivalent to type="response" in R)
        phat = model2.predict_proba(X_test)[:, 1]
        
        # Clip predictions to [0, 1] (though logistic already outputs in this range)
        phat = np.clip(phat, 0, 1)
        
        # Calculate AUC
        auc_score = roc_auc_score(y_test, phat)
        AUC.append(auc_score)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(AUC, color='grey', alpha=0.7)
    plt.axhline(y=np.mean(AUC), color='red', linestyle='-')
    plt.title('AUC Scores Across Iterations (Logistic Regression)')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.show()
    
    mean_auc = np.mean(AUC)
    std_auc = np.sqrt(np.var(AUC))
    
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Standard Deviation: {std_auc:.4f}")
    
    return AUC, mean_auc, std_auc




#####################################################################################################################


# ============================================
#              chunk_tc23
# ============================================

def chunk_tc23(csv_path="adult_train.csv"):
    """
    Load and preprocess adult dataset with variable naming, scaling, and type conversion.
    Removes specific observations and standardizes numerical variables.
    """
    # Load data without header
    data = pd.read_csv(csv_path, header=None)
    
    # Define variable names
    varNames = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
                "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
                "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry",
                "IncomeLevel"]
    
    data.columns = varNames
    df = data.copy()
    
    # Drop specific observation
    holand_mask = df['NativeCountry'] == " Holand-Netherlands"
    df = df[~holand_mask].reset_index(drop=True)
    
    # Scale numerical (integer) variables
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        df[col] = scaler.fit_transform(df[[col]]).flatten()
    
    # Convert object/string columns to categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    return df





#####################################################################################################################



# ============================================
#              chunk_tc24
# ============================================

def chunk_tc24(df, h=10, random_state=123):
    """
    Perform k-fold cross-validation with KNN to find optimal k values.
    Uses h-fold CV with hyperparameter tuning for k-nearest neighbors.
    """
    np.random.seed(random_state)
    
    # Create shuffled indices
    n_samples = len(df)
    shuffled_indices = np.random.permutation(n_samples)
    
    # Split into train/test (90%/10%)
    test_size = n_samples // h
    ind_test = shuffled_indices[:test_size]
    ind_train = shuffled_indices[test_size:]
    
    trdf = df.iloc[ind_train].reset_index(drop=True)
    tsdf = df.iloc[ind_test].reset_index(drop=True)
    
    # Prepare data for KNN (encode categorical variables)
    X_full = trdf.drop('IncomeLevel', axis=1)
    y_full = trdf['IncomeLevel']
    
    # Encode categorical variables
    le_dict = {}
    X_encoded = X_full.copy()
    
    for col in X_encoded.select_dtypes(include=['category', 'object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        le_dict[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_full.astype(str))
    
    # h-fold CV setup
    nval = len(trdf) // h
    k_values = list(range(3, 51, 2))  # k from 3 to 50 by 2
    
    MAUC2 = []
    k_opt = []
    
    for i in range(h):
        # Define validation indices for fold i
        if i < h - 1:
            ind_val = list(range(i * nval, (i + 1) * nval))
        else:
            ind_val = list(range(i * nval, len(trdf)))
        
        # Create train/validation split for this fold
        ind_train_fold = [idx for idx in range(len(trdf)) if idx not in ind_val]
        
        X_train_fold = X_encoded.iloc[ind_train_fold]
        y_train_fold = y_encoded[ind_train_fold]
        X_val_fold = X_encoded.iloc[ind_val]
        y_val_fold = y_encoded[ind_val]
        
        AUC = []
        
        # Test different k values
        for k_val in k_values:
            knn = KNeighborsClassifier(n_neighbors=k_val)
            knn.fit(X_train_fold, y_train_fold)
            
            # Get probability predictions
            phat = knn.predict_proba(X_val_fold)
            
            # Calculate AUC (use class 1 probabilities)
            if phat.shape[1] > 1:
                auc_score = roc_auc_score(y_val_fold, phat[:, 1])
            else:
                auc_score = 0.5  # Default if only one class present
            
            AUC.append(auc_score)
        
        # Store best AUC and corresponding k for this fold
        best_idx = np.argmax(AUC)
        MAUC2.append(AUC[best_idx])
        k_opt.append(k_values[best_idx])
    
    print(f"Mean optimal AUC: {np.mean(MAUC2):.4f}")
    print(f"Mean optimal k: {np.mean(k_opt):.1f}")
    
    return MAUC2, k_opt, trdf, tsdf, le_dict, le_target




#####################################################################################################################


# ============================================
#              chunk_tc25
# ============================================

def chunk_tc25(k_opt, MAUC2):
    """
    Display optimal k values and corresponding AUC scores from cross-validation.
    Shows fold-wise results and calculates mean values.
    """
    # Create combined table (equivalent to cbind in R)
    results_df = pd.DataFrame({
        'k_opt': k_opt,
        'MAUC2': MAUC2
    })
    
    print("Optimal k and AUC by fold:")
    print(results_df)
    print()
    
    # Calculate and display means
    mean_k = np.mean(k_opt)
    mean_auc = np.mean(MAUC2)
    
    print(f"Mean optimal k: {mean_k:.1f}")
    print(f"Mean AUC: {mean_auc:.4f}")
    
    return results_df, mean_k, mean_auc





#####################################################################################################################




# ============================================
#              chunk_tc26
# ============================================

def chunk_tc26(df, h=10, m=20, random_state=123):
    """
    Bootstrap hyperparameter tuning for KNN using out-of-bag validation.
    Uses bootstrap sampling to find optimal k values across multiple iterations.
    """
    np.random.seed(random_state)
    
    # Train/test split (same as before)
    n_samples = len(df)
    shuffled_indices = np.random.permutation(n_samples)
    
    test_size = n_samples // h
    ind_test = shuffled_indices[:test_size]
    ind_train = shuffled_indices[test_size:]
    
    trdf = df.iloc[ind_train].reset_index(drop=True)
    tsdf = df.iloc[ind_test].reset_index(drop=True)
    
    # Encode categorical variables for the full dataset
    df_encoded = df.copy()
    le_dict = {}
    
    for col in df_encoded.select_dtypes(include=['category', 'object']).columns:
        if col != 'IncomeLevel':  # Don't encode target yet
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df_encoded['IncomeLevel'] = le_target.fit_transform(df_encoded['IncomeLevel'].astype(str))
    
    # Bootstrap hyperparameter tuning
    k_values = list(range(3, 51, 2))  # k from 3 to 50 by 2
    MAUC = []
    
    for i, k_val in enumerate(k_values):
        AUC = []
        
        for l in range(1, m + 1):
            np.random.seed(l)
            
            # Bootstrap sampling with replacement
            n_train = len(trdf)
            bootstrap_indices = np.random.choice(n_train, size=n_train, replace=True)
            unique_indices = np.unique(bootstrap_indices)
            
            # Out-of-bag indices for validation
            oob_indices = [idx for idx in range(n_train) if idx not in unique_indices]
            
            if len(oob_indices) == 0:
                continue  # Skip if no out-of-bag samples
            
            # Prepare bootstrap train and validation sets
            train_indices_global = ind_train[unique_indices]
            val_indices_global = ind_train[oob_indices]
            
            df_train_boot = df_encoded.iloc[train_indices_global]
            df_val_boot = df_encoded.iloc[val_indices_global]
            
            # Separate features and target
            X_train = df_train_boot.drop('IncomeLevel', axis=1)
            y_train = df_train_boot['IncomeLevel']
            X_val = df_val_boot.drop('IncomeLevel', axis=1)
            y_val = df_val_boot['IncomeLevel']
            
            # Train KNN model
            knn = KNeighborsClassifier(n_neighbors=k_val)
            knn.fit(X_train, y_train)
            
            # Get probability predictions
            phat = knn.predict_proba(X_val)
            
            # Calculate AUC
            if phat.shape[1] > 1 and len(np.unique(y_val)) > 1:
                auc_score = roc_auc_score(y_val, phat[:, 1])
                AUC.append(auc_score)
        
        # Calculate mean AUC for this k value
        if AUC:  # Only if we have valid AUC scores
            MAUC.append(np.mean(AUC))
        else:
            MAUC.append(0.5)  # Default AUC
    
    # Find optimal k
    best_k_idx = np.argmax(MAUC)
    optimal_k = k_values[best_k_idx]
    best_auc = MAUC[best_k_idx]
    
    print(f"Optimal k: {optimal_k}")
    print(f"Best mean AUC: {best_auc:.4f}")
    
    return MAUC, k_values, optimal_k, trdf, tsdf, le_dict, le_target





#####################################################################################################################


# ============================================
#              chunk_tc27
# ============================================

def chunk_tc27(k, MAUC):
    """
    Plot k vs MAUC and return max MAUC value and corresponding k.
    
    Args:
        k: array-like, x-axis values
        MAUC: array-like, y-axis values
        
    Returns:
        tuple: (max_MAUC_value, k_at_max_MAUC)
    """
    plt.plot(k, MAUC, color='red', marker='o', linestyle='-')
    plt.xlabel('k')
    plt.ylabel('MAUC')
    plt.show()
    
    max_idx = np.argmax(MAUC)
    return MAUC[max_idx], k[max_idx]




#####################################################################################################################


# ============================================
#              chunk_tc28
# ============================================

def chunk_tc28(df):
    """
    Separate categorical and numeric columns, apply dummy coding, and combine.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        pandas DataFrame: Combined dataframe with Y, numeric columns, and dummy variables
    """
    dftmp = df.iloc[:, :-1]  # Remove last column (index 15)
    
    # Separate categorical and numeric columns
    categorical_cols = dftmp.select_dtypes(include=['object', 'category']).columns
    numeric_cols = dftmp.select_dtypes(exclude=['object', 'category']).columns
    
    fctdf = dftmp[categorical_cols]
    numdf = dftmp[numeric_cols]
    
    # Dummy coding
    fctdum = pd.get_dummies(fctdf, drop_first=True)
    
    # Binding
    df_dum = pd.concat([
        df['IncomeLevel'].rename('Y'),
        numdf,
        fctdum
    ], axis=1)
    
    return df_dum




#####################################################################################################################



# ============================================
#              chunk_tc29
# ============================================

def chunk_tc29(df):
    """
    Prepare data, train kNN model with cross-validation, and find optimal k.
    
    Args:
        df: pandas DataFrame with IncomeLevel column
        
    Returns:
        dict: Contains best model, results, and optimal k
    """
    # Relabel IncomeLevel
    df_copy = df.copy()
    df_copy['IncomeLevel'] = df_copy['IncomeLevel'].replace({' <=50K': 'Less', ' >50K': 'More'})
    
    # Test/Train split (10% test, 90% train)
    np.random.seed(123)
    X = df_copy.drop('IncomeLevel', axis=1)
    y = df_copy['IncomeLevel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123, stratify=y)
    
    # Scale features for kNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation setup (10-fold)
    cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    
    # Grid search for optimal k
    k_range = range(3, 51, 2)
    cv_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv_folds, scoring='roc_auc')
        cv_scores.append(scores.mean())
    
    # Find optimal k
    optimal_idx = np.argmax(cv_scores)
    optimal_k = k_range[optimal_idx]
    
    # Train final model with optimal k
    best_model = KNeighborsClassifier(n_neighbors=optimal_k)
    best_model.fit(X_train_scaled, y_train)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, cv_scores, 'bo-')
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Cross-Validation ROC AUC')
    plt.title(f'kNN Performance vs k (Optimal k = {optimal_k})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    results = pd.DataFrame({'k': k_range, 'ROC_AUC': cv_scores})
    
    return {
        'model': best_model,
        'optimal_k': optimal_k,
        'results': results,
        'scaler': scaler,
        'train_data': (X_train_scaled, y_train),
        'test_data': (X_test_scaled, y_test)
    }




#####################################################################################################################


# ============================================
#              chunk_tc30
# ============================================

def chunk_tc30(model_results):
    """
    Generate performance metrics and confusion matrices for kNN model.
    
    Args:
        model_results: dict from chunk_tc29 containing model and test data
        
    Returns:
        dict: Contains confusion matrices and performance metrics
    """
    model = model_results['model']
    X_test, y_test = model_results['test_data']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Confusion matrix with default positive class (alphabetical first = "Less")
    cm_default = confusion_matrix(y_test, y_pred)
    
    # Confusion matrix with "More" as positive class
    cm_more_positive = confusion_matrix(y_test, y_pred, labels=["More", "Less"])
    
    # Classification reports
    report_default = classification_report(y_test, y_pred, output_dict=True)
    report_more_positive = classification_report(y_test, y_pred, labels=["More", "Less"], output_dict=True)
    
    # Display confusion matrices
    print("Confusion Matrix (Default - 'Less' as positive):")
    print(pd.DataFrame(cm_default, 
                      index=['True Less', 'True More'], 
                      columns=['Pred Less', 'Pred More']))
    
    print("\nConfusion Matrix ('More' as positive):")
    print(pd.DataFrame(cm_more_positive, 
                      index=['True More', 'True Less'], 
                      columns=['Pred More', 'Pred Less']))
    
    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return {
        'confusion_matrix_default': cm_default,
        'confusion_matrix_more_positive': cm_more_positive,
        'classification_report_default': report_default,
        'classification_report_more_positive': report_more_positive,
        'accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred
    }