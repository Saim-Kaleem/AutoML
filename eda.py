import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import io


def generate_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    """
    Generate histograms for all numeric features.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        
    Returns:
        Matplotlib figure
    """
    if not numeric_cols:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Handle single subplot case properly
    if n_rows == 1 and n_cols == 1:
        axes = [axes]  # Make it a list for consistent indexing
    elif len(numeric_cols) == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else [axes.item()]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_categorical_barplots(df: pd.DataFrame, categorical_cols: List[str], max_categories: int = 10) -> plt.Figure:
    """
    Generate bar plots for categorical features.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        max_categories: Maximum number of categories to display
        
    Returns:
        Matplotlib figure
    """
    if not categorical_cols:
        return None
    
    n_cols = min(2, len(categorical_cols))
    n_rows = (len(categorical_cols) - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    # Handle single subplot case properly
    if n_rows == 1 and n_cols == 1:
        axes = [axes]  # Make it a list for consistent indexing
    elif len(categorical_cols) == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else [axes.item()]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        value_counts = df[col].value_counts().head(max_categories)
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        if df[col].nunique() > max_categories:
            ax.text(0.5, 0.95, f'(Showing top {max_categories} of {df[col].nunique()} categories)',
                   transform=ax.transAxes, ha='center', va='top', fontsize=8, style='italic')
    
    # Hide unused subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_boxplots(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    """
    Generate boxplots for numeric features to show outliers.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        
    Returns:
        Matplotlib figure
    """
    if not numeric_cols:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Handle single subplot case properly
    if n_rows == 1 and n_cols == 1:
        axes = [axes]  # Make it a list for consistent indexing
    elif len(numeric_cols) == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else [axes.item()]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df.boxplot(column=col, ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', edgecolor='black'),
                   medianprops=dict(color='red', linewidth=2),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
        ax.set_title(f'Boxplot of {col}', fontsize=10, fontweight='bold')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    """
    Generate correlation heatmap for numeric features.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        
    Returns:
        Matplotlib figure
    """
    if not numeric_cols or len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def get_strong_correlations(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """
    Find pairs of features with strong correlations.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        threshold: Correlation threshold (absolute value)
        
    Returns:
        List of tuples (feature1, feature2, correlation)
    """
    if not numeric_cols or len(numeric_cols) < 2:
        return []
    
    corr_matrix = df[numeric_cols].corr()
    strong_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    # Sort by absolute correlation value
    strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return strong_corr


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values per column.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with missing value statistics
    """
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    return missing_data.reset_index(drop=True)


def generate_missing_value_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Generate visualization of missing values.
    
    Args:
        df: Input dataframe
        
    Returns:
        Matplotlib figure
    """
    missing_data = get_missing_value_summary(df)
    
    if missing_data.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(missing_data) * 0.4)))
    
    ax.barh(missing_data['Column'], missing_data['Missing_Percentage'], color='coral', edgecolor='black')
    ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Column', fontsize=12, fontweight='bold')
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (col, pct) in enumerate(zip(missing_data['Column'], missing_data['Missing_Percentage'])):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def generate_target_distribution(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """
    Generate bar plot showing target variable distribution.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        
    Returns:
        Matplotlib figure
    """
    if target_col not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    value_counts = df[target_col].value_counts().sort_index()
    colors = sns.color_palette('Set2', len(value_counts))
    
    bars = ax.bar(value_counts.index.astype(str), value_counts.values, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Target Variable Distribution: {target_col}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({height/len(df)*100:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def compute_eda_summary(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
    """
    Compute comprehensive EDA summary statistics.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary with EDA summary
    """
    summary = {
        'numeric_summary': df[numeric_cols].describe().to_dict() if numeric_cols else {},
        'categorical_summary': {},
        'skewness': {},
        'kurtosis': {}
    }
    
    # Categorical summaries
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_values': int(df[col].nunique()),
            'top_value': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else 'N/A',
            'top_value_freq': int(df[col].value_counts().iloc[0]) if len(df[col]) > 0 else 0
        }
    
    # Skewness and kurtosis for numeric features
    for col in numeric_cols:
        summary['skewness'][col] = float(df[col].skew())
        summary['kurtosis'][col] = float(df[col].kurtosis())
    
    return summary
