import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple rule-based classifier using decision rules.
    Uses majority class for prediction by default.
    """
    
    def __init__(self):
        self.majority_class_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit the classifier."""
        self.classes_ = np.unique(y)
        # Find majority class
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class_ = unique[np.argmax(counts)]
        return self
    
    def predict(self, X):
        """Predict using majority class rule."""
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.full(n_samples, self.majority_class_)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        # Assign probability 1.0 to majority class
        majority_idx = np.where(self.classes_ == self.majority_class_)[0][0]
        proba[:, majority_idx] = 1.0
        
        return proba


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all models.
    
    Returns:
        Dictionary mapping model names to their configurations
    """
    return {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            },
            'description': 'Linear model for binary and multiclass classification'
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'description': 'Instance-based learning using k nearest neighbors'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'description': 'Tree-based model using recursive partitioning'
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'description': 'Probabilistic classifier based on Bayes theorem'
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            },
            'description': 'Ensemble of decision trees with bagging'
        },
        'Support Vector Machine': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'description': 'Maximum margin classifier with kernel trick'
        },
        'Rule-Based Classifier': {
            'model': RuleBasedClassifier(),
            'params': {},
            'description': 'Simple rule-based baseline (majority class)'
        }
    }


def train_single_model(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Train a single model and return results.
    
    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with training results
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
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get prediction probabilities (if available)
    try:
        y_test_proba = model.predict_proba(X_test)
    except AttributeError:
        y_test_proba = None
    
    return {
        'model': model,
        'training_time': training_time,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'trained': True
    }


def train_all_models(X_train, y_train, X_test, y_test, 
                     class_weights: Dict[Any, float] = None) -> Dict[str, Dict[str, Any]]:
    """
    Train all models and return results.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        class_weights: Class weights for imbalanced data (optional)
        
    Returns:
        Dictionary mapping model names to their results
    """
    model_configs = get_model_configs()
    results = {}
    
    for model_name, config in model_configs.items():
        try:
            model = config['model']
            
            # Apply class weights if supported and provided
            if class_weights and hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weights)
            
            result = train_single_model(model, X_train, y_train, X_test, y_test)
            result['description'] = config['description']
            results[model_name] = result
            
        except Exception as e:
            results[model_name] = {
                'trained': False,
                'error': str(e),
                'description': config['description']
            }
    
    return results


def get_model_by_name(model_name: str, **kwargs):
    """
    Get a model instance by name with optional parameters.
    
    Args:
        model_name: Name of the model
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    model_configs = get_model_configs()
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = type(model_configs[model_name]['model'])
    return model_class(**kwargs)


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    importance_df = None
    
    try:
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                # Multiclass
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                # Binary
                importances = np.abs(model.coef_[0])
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
    
    except Exception:
        pass
    
    return importance_df


def create_model_summary_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary table of all model results.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        DataFrame with model summary
    """
    summary_data = []
    
    for model_name, result in results.items():
        if result.get('trained', False):
            summary_data.append({
                'Model': model_name,
                'Training Time (s)': f"{result['training_time']:.4f}",
                'Status': 'Trained Successfully',
                'Description': result.get('description', '')
            })
        else:
            summary_data.append({
                'Model': model_name,
                'Training Time (s)': 'N/A',
                'Status': f"Failed: {result.get('error', 'Unknown error')}",
                'Description': result.get('description', '')
            })
    
    return pd.DataFrame(summary_data)


def get_model_parameters(model) -> Dict[str, Any]:
    """
    Get parameters of a trained model.
    
    Args:
        model: Trained model
        
    Returns:
        Dictionary of model parameters
    """
    try:
        return model.get_params()
    except Exception:
        return {}


def save_model(model, filepath: str):
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    import joblib
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """
    Load trained model from file.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    import joblib
    return joblib.load(filepath)
