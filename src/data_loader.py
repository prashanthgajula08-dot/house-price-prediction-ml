import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset successfully loaded from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        raise

def prepare_features_target(
    df: pd.DataFrame, 
    features: List[str], 
    target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts features and target variable from the dataframe.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        features (List[str]): List of column names to use as features.
        target (str): Column name to use as target variable.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame (X) and Target Series (y).
    """
    # Verify columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from the DataFrame: {missing_cols}")
        
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle missing values if any exist in feature columns
    # In the baseline notebook, X.isnull().sum() was checked and no fillna was needed for the selected columns,
    # but let's add a robust check and log if any missing values are found.
    null_counts = X.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"Warning: Feature '{col}' contains {count} missing values. Filling with column mean.")
            X[col] = X[col].fillna(X[col].mean())
            
    return X, y

def split_dataset(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test
