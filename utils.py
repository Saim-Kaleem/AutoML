import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate uploaded dataframe meets basic requirements.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) < 2:
        return False, "DataFrame must have at least 2 columns (features + target)"
    
    if len(df) < 10:
        return False, "DataFrame must have at least 10 rows"
    
    return True, ""


def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features into numeric and categorical.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with 'numeric' and 'categorical' feature lists
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calculate dataframe memory usage in MB.
    
    Args:
        df: Input dataframe
        
    Returns:
        Memory usage in megabytes
    """
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic statistics about the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with basic statistics
    """
    feature_types = get_feature_types(df)
    
    return {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_numeric': len(feature_types['numeric']),
        'n_categorical': len(feature_types['categorical']),
        'memory_mb': memory_usage_mb(df),
        'duplicate_rows': df.duplicated().sum(),
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }


def safe_column_name(col: str) -> str:
    """
    Convert column name to safe format (no special characters).
    
    Args:
        col: Column name
        
    Returns:
        Safe column name
    """
    return col.replace(' ', '_').replace('-', '_').replace('.', '_')


def detect_constant_features(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Detect features with near-constant values.
    
    Args:
        df: Input dataframe
        threshold: Threshold for considering a feature constant (default 0.95)
        
    Returns:
        List of near-constant feature names
    """
    constant_features = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # For numeric, check if one value dominates
            if len(df[col].unique()) == 1:
                constant_features.append(col)
            elif df[col].value_counts(normalize=True).iloc[0] > threshold:
                constant_features.append(col)
        else:
            # For categorical, check if one category dominates
            if df[col].value_counts(normalize=True).iloc[0] > threshold:
                constant_features.append(col)
    
    return constant_features


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def calculate_class_weights(y: pd.Series) -> Dict[Any, float]:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def get_cardinality(series: pd.Series) -> int:
    """
    Get the number of unique values in a series.
    
    Args:
        series: Input series
        
    Returns:
        Number of unique values
    """
    return series.nunique()


def is_binary_classification(y: pd.Series) -> bool:
    """
    Check if target is binary classification.
    
    Args:
        y: Target variable
        
    Returns:
        True if binary classification
    """
    return len(y.unique()) == 2


def prepare_for_json(obj: Any) -> Any:
    """
    Convert numpy/pandas types to JSON-serializable types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [prepare_for_json(item) for item in obj]
    else:
        return obj
