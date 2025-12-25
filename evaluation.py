import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)


def calculate_metrics(y_true, y_pred, average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multiclass
        
    Returns:
        Dictionary with metric values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def evaluate_model(model, X_train, y_train, X_test, y_test,
                  training_time: float = 0.0) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a single model.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        training_time: Model training time
        
    Returns:
        Dictionary with evaluation results
    """
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Get prediction probabilities
    y_test_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_test_proba = model.predict_proba(X_test)
        except Exception:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Classification report
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'confusion_matrix': cm,
        'classification_report': report,
        'training_time': training_time
    }


def evaluate_all_models(model_results: Dict[str, Dict[str, Any]],
                       X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all trained models.
    
    Args:
        model_results: Dictionary of model results from training
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation results for all models
    """
    evaluation_results = {}
    
    for model_name, result in model_results.items():
        if result.get('trained', False):
            try:
                evaluation = evaluate_model(
                    result['model'], X_train, y_train, X_test, y_test,
                    training_time=result.get('training_time', 0.0)
                )
                evaluation_results[model_name] = evaluation
            except Exception as e:
                evaluation_results[model_name] = {
                    'error': str(e)
                }
        else:
            evaluation_results[model_name] = {
                'error': 'Model was not trained'
            }
    
    return evaluation_results


def create_comparison_table(evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comparison table of all models.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, result in evaluation_results.items():
        if 'test_metrics' in result:
            test_metrics = result['test_metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{test_metrics['accuracy']:.4f}",
                'Precision': f"{test_metrics['precision']:.4f}",
                'Recall': f"{test_metrics['recall']:.4f}",
                'F1-Score': f"{test_metrics['f1_score']:.4f}",
                'Training Time (s)': f"{result['training_time']:.4f}"
            })
        else:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A',
                'F1-Score': 'N/A',
                'Training Time (s)': 'N/A'
            })
    
    return pd.DataFrame(comparison_data)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None,
                         title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='#1f2937')
    
    plt.tight_layout()
    return fig


def plot_all_confusion_matrices(evaluation_results: Dict[str, Dict[str, Any]],
                                class_names: List[str] = None) -> plt.Figure:
    """
    Plot confusion matrices for all models in a grid.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    # Filter models with valid results
    valid_models = {name: result for name, result in evaluation_results.items()
                   if 'confusion_matrix' in result}
    
    if not valid_models:
        return None
    
    n_models = len(valid_models)
    n_cols = min(3, n_models)
    n_rows = (n_models - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, result) in enumerate(valid_models.items()):
        cm = result['confusion_matrix']
        ax = axes[idx]
        
        if class_names is None:
            class_names_local = [str(i) for i in range(cm.shape[0])]
        else:
            class_names_local = class_names
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
                   xticklabels=class_names_local, yticklabels=class_names_local,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=10, color='#1f2937')
        ax.set_ylabel('True', fontsize=10, color='#1f2937')
        ax.set_title(model_name, fontsize=11, fontweight='bold', color='#1f2937')
    
    # Hide unused subplots
    for idx in range(len(valid_models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_test_proba, model_name: str = 'Model') -> plt.Figure:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_test: True labels
        y_test_proba: Predicted probabilities
        model_name: Model name for title
        
    Returns:
        Matplotlib figure
    """
    # Check if binary classification
    n_classes = len(np.unique(y_test))
    
    if n_classes != 2:
        return None
    
    # Get positive class probabilities
    if y_test_proba.shape[1] == 2:
        y_score = y_test_proba[:, 1]
    else:
        y_score = y_test_proba[:, 0]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#f59e0b', lw=2,
           label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='#1f2937', lw=2, linestyle='--',
           label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=20, color='#1f2937')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_all_roc_curves(evaluation_results: Dict[str, Dict[str, Any]],
                       y_test) -> plt.Figure:
    """
    Plot ROC curves for all models on the same plot.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        y_test: True test labels
        
    Returns:
        Matplotlib figure
    """
    # Check if binary classification
    n_classes = len(np.unique(y_test))
    
    if n_classes != 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use app theme colors
    theme_colors = ['#f59e0b', '#10b981', '#ef4444', '#f97316', '#14b8a6', '#fb923c', '#22c55e']
    colors = (theme_colors * (len(evaluation_results) // len(theme_colors) + 1))[:len(evaluation_results)]
    
    for idx, (model_name, result) in enumerate(evaluation_results.items()):
        if 'y_test_proba' in result and result['y_test_proba'] is not None:
            y_test_proba = result['y_test_proba']
            
            # Get positive class probabilities
            if y_test_proba.shape[1] == 2:
                y_score = y_test_proba[:, 1]
            else:
                y_score = y_test_proba[:, 0]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], color='#1f2937', lw=2, linestyle='--',
           label='Random Classifier', alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold', pad=20, color='#1f2937')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(evaluation_results: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        
    Returns:
        Matplotlib figure
    """
    # Prepare data
    models = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for model_name, result in evaluation_results.items():
        if 'test_metrics' in result:
            models.append(model_name)
            metrics = result['test_metrics']
            accuracy.append(metrics['accuracy'])
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1.append(metrics['f1_score'])
    
    if not models:
        return None
    
    # Create plot
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#f59e0b', edgecolor='#1f2937')
    ax.bar(x - 0.5*width, precision, width, label='Precision', color='#10b981', edgecolor='#1f2937')
    ax.bar(x + 0.5*width, recall, width, label='Recall', color='#f97316', edgecolor='#1f2937')
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#14b8a6', edgecolor='#1f2937')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold', color='#1f2937')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20, color='#1f2937')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig


def get_best_model(evaluation_results: Dict[str, Dict[str, Any]],
                  metric: str = 'f1_score') -> Tuple[str, Dict[str, Any]]:
    """
    Get the best performing model based on a metric.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        metric: Metric to use for comparison
        
    Returns:
        Tuple of (best_model_name, best_model_results)
    """
    best_model_name = None
    best_score = -1
    best_result = None
    
    for model_name, result in evaluation_results.items():
        if 'test_metrics' in result:
            score = result['test_metrics'].get(metric, -1)
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_result = result
    
    return best_model_name, best_result


def get_detailed_classification_report(evaluation_results: Dict[str, Dict[str, Any]],
                                      model_name: str) -> pd.DataFrame:
    """
    Get detailed classification report for a specific model.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        model_name: Name of the model
        
    Returns:
        DataFrame with classification report
    """
    if model_name not in evaluation_results:
        return None
    
    result = evaluation_results[model_name]
    
    if 'classification_report' not in result:
        return None
    
    report = result['classification_report']
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    return df


def export_comparison_table(evaluation_results: Dict[str, Dict[str, Any]],
                           filepath: str):
    """
    Export comparison table to CSV.
    
    Args:
        evaluation_results: Dictionary of evaluation results
        filepath: Path to save the CSV file
    """
    comparison_df = create_comparison_table(evaluation_results)
    comparison_df.to_csv(filepath, index=False)
