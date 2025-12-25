import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats


def detect_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect missing values globally and per feature.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with missing value information
    """
    total_cells = len(df) * len(df.columns)
    total_missing = df.isnull().sum().sum()
    
    missing_per_column = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_per_column[col] = {
                'count': int(missing_count),
                'percentage': float((missing_count / len(df)) * 100)
            }
    
    return {
        'total_missing': int(total_missing),
        'total_cells': int(total_cells),
        'global_percentage': float((total_missing / total_cells) * 100),
        'columns_with_missing': missing_per_column,
        'has_missing': total_missing > 0
    }


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Tuple[List[int], int, float, float]:
    """
    Detect outliers using IQR method.
    
    Args:
        series: Input series (numeric)
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Tuple of (outlier_indices, outlier_count, lower_bound, upper_bound)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_indices = outliers.index.tolist()
    
    return outlier_indices, len(outliers), float(lower_bound), float(upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> Tuple[List[int], int]:
    """
    Detect outliers using Z-score method.
    
    Args:
        series: Input series (numeric)
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Tuple of (outlier_indices, outlier_count)
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    outlier_mask = z_scores > threshold
    
    outlier_indices = series.dropna().index[outlier_mask].tolist()
    
    return outlier_indices, len(outlier_indices)


def detect_all_outliers(df: pd.DataFrame, numeric_cols: List[str], method: str = 'iqr') -> Dict[str, Any]:
    """
    Detect outliers in all numeric columns.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        method: Detection method ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier information per column
    """
    outliers_info = {}
    
    for col in numeric_cols:
        if df[col].isnull().all():
            continue
        
        if method == 'iqr':
            indices, count, lower, upper = detect_outliers_iqr(df[col].dropna())
            outliers_info[col] = {
                'count': count,
                'percentage': float((count / len(df)) * 100),
                'lower_bound': lower,
                'upper_bound': upper,
                'method': 'IQR'
            }
        else:  # zscore
            indices, count = detect_outliers_zscore(df[col].dropna())
            outliers_info[col] = {
                'count': count,
                'percentage': float((count / len(df)) * 100),
                'method': 'Z-score'
            }
    
    total_outliers = sum(info['count'] for info in outliers_info.values())
    
    return {
        'outliers_per_column': outliers_info,
        'total_outlier_instances': total_outliers,
        'columns_with_outliers': [col for col, info in outliers_info.items() if info['count'] > 0]
    }


def detect_high_cardinality(df: pd.DataFrame, categorical_cols: List[str], threshold: int = 20) -> List[Dict[str, Any]]:
    """
    Detect categorical columns with high cardinality.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        threshold: Cardinality threshold (default 20)
        
    Returns:
        List of high-cardinality columns with details
    """
    high_card_cols = []
    
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > threshold:
            high_card_cols.append({
                'column': col,
                'unique_values': int(n_unique),
                'percentage_unique': float((n_unique / len(df)) * 100)
            })
    
    return high_card_cols


def detect_constant_features(df: pd.DataFrame, threshold: float = 0.95) -> List[Dict[str, Any]]:
    """
    Detect features with near-constant values.
    
    Args:
        df: Input dataframe
        threshold: Threshold for considering a feature constant
        
    Returns:
        List of near-constant features with details
    """
    constant_features = []
    
    for col in df.columns:
        n_unique = df[col].nunique()
        
        if n_unique == 1:
            constant_features.append({
                'column': col,
                'unique_values': 1,
                'issue': 'All values are identical'
            })
        elif n_unique > 1:
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_freq > threshold:
                constant_features.append({
                    'column': col,
                    'unique_values': int(n_unique),
                    'dominant_value_percentage': float(top_freq * 100),
                    'issue': f'One value dominates ({top_freq*100:.1f}%)'
                })
    
    return constant_features


def detect_class_imbalance(y: pd.Series, threshold: float = None) -> Dict[str, Any]:
    """
    Detect class imbalance in target variable.
    
    Args:
        y: Target variable
        threshold: Imbalance threshold (minority class percentage). If None, automatically calculated based on number of classes.
        
    Returns:
        Dictionary with imbalance information
    """
    class_counts = y.value_counts()
    class_percentages = y.value_counts(normalize=True)
    
    n_classes = len(class_counts)
    
    # Auto-calculate threshold based on number of classes if not provided
    if threshold is None:
        # Expected balanced percentage = 100% / n_classes
        # Threshold = 50% of expected balanced percentage
        # For binary: 50% of 50% = 25%
        # For 3 classes: 50% of 33.3% = 16.7%
        # For 5 classes: 50% of 20% = 10%
        expected_balanced = 1.0 / n_classes
        threshold = expected_balanced * 0.5
    
    min_class = class_percentages.idxmin()
    min_percentage = class_percentages.min()
    max_class = class_percentages.idxmax()
    max_percentage = class_percentages.max()
    
    is_imbalanced = min_percentage < threshold
    
    imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
    
    return {
        'is_imbalanced': is_imbalanced,
        'n_classes': n_classes,
        'class_distribution': {str(k): int(v) for k, v in class_counts.items()},
        'class_percentages': {str(k): float(v * 100) for k, v in class_percentages.items()},
        'minority_class': str(min_class),
        'minority_percentage': float(min_percentage * 100),
        'majority_class': str(max_class),
        'majority_percentage': float(max_percentage * 100),
        'imbalance_ratio': float(imbalance_ratio),
        'threshold_used': float(threshold * 100)
    }


def detect_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect duplicate rows in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with duplicate row information
    """
    n_duplicates = df.duplicated().sum()
    
    return {
        'has_duplicates': n_duplicates > 0,
        'n_duplicates': int(n_duplicates),
        'percentage': float((n_duplicates / len(df)) * 100)
    }


def detect_mixed_datatypes(df: pd.DataFrame) -> List[str]:
    """
    Detect columns with mixed data types.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of columns with mixed types
    """
    mixed_type_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                # Check if it's truly mixed
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    types = non_null.apply(type).unique()
                    if len(types) > 1:
                        mixed_type_cols.append(col)
    
    return mixed_type_cols


def run_comprehensive_diagnostics(df: pd.DataFrame, numeric_cols: List[str], 
                                  categorical_cols: List[str], 
                                  target_col: str = None,
                                  outlier_method: str = 'iqr',
                                  cardinality_threshold: int = 20) -> Dict[str, Any]:
    """
    Run comprehensive data quality diagnostics.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        target_col: Target column name (optional)
        outlier_method: Method for outlier detection ('iqr' or 'zscore')
        cardinality_threshold: Threshold for high cardinality detection (default 20)
        
    Returns:
        Dictionary with all diagnostic results
    """
    diagnostics = {
        'missing_values': detect_missing_values(df),
        'outliers': detect_all_outliers(df, numeric_cols, method=outlier_method),
        'high_cardinality': detect_high_cardinality(df, categorical_cols, threshold=cardinality_threshold),
        'constant_features': detect_constant_features(df),
        'duplicates': detect_duplicate_rows(df),
        'mixed_types': detect_mixed_datatypes(df)
    }
    
    # Add class imbalance if target is specified
    if target_col and target_col in df.columns:
        diagnostics['class_imbalance'] = detect_class_imbalance(df[target_col])
    
    return diagnostics


def get_issues_summary(diagnostics: Dict[str, Any]) -> Dict[str, int]:
    """
    Get summary count of all detected issues.
    
    Args:
        diagnostics: Diagnostics dictionary from run_comprehensive_diagnostics
        
    Returns:
        Dictionary with issue counts
    """
    summary = {
        'missing_values': len(diagnostics['missing_values']['columns_with_missing']),
        'outliers': len(diagnostics['outliers']['columns_with_outliers']),
        'high_cardinality': len(diagnostics['high_cardinality']),
        'constant_features': len(diagnostics['constant_features']),
        'duplicates': 1 if diagnostics['duplicates']['has_duplicates'] else 0,
        'mixed_types': len(diagnostics['mixed_types']),
        'class_imbalance': 1 if diagnostics.get('class_imbalance', {}).get('is_imbalanced', False) else 0
    }
    
    summary['total_issues'] = sum(summary.values())
    
    return summary
