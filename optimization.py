import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score


def optimize_model_grid(model, param_grid: Dict[str, List], X_train, y_train,
                        scoring: str = 'accuracy', cv: int = 5,
                        n_jobs: int = -1) -> Dict[str, Any]:
    """
    Optimize model using GridSearchCV.
    
    Args:
        model: Model instance to optimize
        param_grid: Parameter grid to search
        X_train: Training features
        y_train: Training target
        scoring: Scoring metric
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with optimization results
    """
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    
    # Extract results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'optimization_time': optimization_time,
        'n_combinations': len(grid_search.cv_results_['params']),
        'method': 'GridSearchCV'
    }
    
    return results


def optimize_model_random(model, param_distributions: Dict[str, List], X_train, y_train,
                         n_iter: int = 50, scoring: str = 'accuracy', cv: int = 5,
                         n_jobs: int = -1, random_state: int = 42) -> Dict[str, Any]:
    """
    Optimize model using RandomizedSearchCV.
    
    Args:
        model: Model instance to optimize
        param_distributions: Parameter distributions to sample from
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings to sample
        scoring: Scoring metric
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        random_state: Random seed
        
    Returns:
        Dictionary with optimization results
    """
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        random_state=random_state,
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    
    # Extract results
    results = {
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_),
        'best_model': random_search.best_estimator_,
        'cv_results': random_search.cv_results_,
        'optimization_time': optimization_time,
        'n_combinations': n_iter,
        'method': 'RandomizedSearchCV'
    }
    
    return results


def optimize_all_models(model_results: Dict[str, Dict[str, Any]], 
                       model_configs: Dict[str, Dict[str, Any]],
                       X_train, y_train,
                       method: str = 'grid',
                       n_iter: int = 50,
                       scoring: str = 'accuracy',
                       cv: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Optimize all trained models.
    
    Args:
        model_results: Dictionary of initial model results
        model_configs: Dictionary of model configurations
        X_train: Training features
        y_train: Training target
        method: Optimization method ('grid' or 'random')
        n_iter: Number of iterations for random search
        scoring: Scoring metric
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with optimization results for all models
    """
    optimization_results = {}
    
    for model_name, result in model_results.items():
        if not result.get('trained', False):
            optimization_results[model_name] = {
                'optimized': False,
                'error': 'Model was not trained successfully'
            }
            continue
        
        # Get parameter grid/distribution
        param_grid = model_configs[model_name]['params']
        
        # Skip if no parameters to optimize
        if not param_grid:
            optimization_results[model_name] = {
                'optimized': False,
                'error': 'No parameters to optimize'
            }
            continue
        
        try:
            model = result['model']
            
            if method == 'grid':
                opt_result = optimize_model_grid(
                    model, param_grid, X_train, y_train,
                    scoring=scoring, cv=cv
                )
            else:  # random
                opt_result = optimize_model_random(
                    model, param_grid, X_train, y_train,
                    n_iter=n_iter, scoring=scoring, cv=cv
                )
            
            opt_result['optimized'] = True
            optimization_results[model_name] = opt_result
            
        except Exception as e:
            optimization_results[model_name] = {
                'optimized': False,
                'error': str(e)
            }
    
    return optimization_results


def get_cv_results_dataframe(cv_results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert CV results to a clean DataFrame.
    
    Args:
        cv_results: CV results from GridSearchCV or RandomizedSearchCV
        
    Returns:
        DataFrame with CV results
    """
    results_df = pd.DataFrame(cv_results)
    
    # Select relevant columns
    cols_to_keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    cols_to_keep = [col for col in cols_to_keep if col in results_df.columns]
    
    results_df = results_df[cols_to_keep]
    results_df = results_df.sort_values('rank_test_score')
    
    return results_df


def create_optimization_summary(optimization_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary table of optimization results.
    
    Args:
        optimization_results: Dictionary of optimization results
        
    Returns:
        DataFrame with optimization summary
    """
    summary_data = []
    
    for model_name, result in optimization_results.items():
        if result.get('optimized', False):
            summary_data.append({
                'Model': model_name,
                'Method': result.get('method', 'N/A'),
                'Best CV Score': f"{result['best_score']:.4f}",
                'Combinations Tested': result.get('n_combinations', 0),
                'Optimization Time (s)': f"{result['optimization_time']:.2f}",
                'Status': 'Optimized'
            })
        else:
            summary_data.append({
                'Model': model_name,
                'Method': 'N/A',
                'Best CV Score': 'N/A',
                'Combinations Tested': 0,
                'Optimization Time (s)': 'N/A',
                'Status': f"Failed: {result.get('error', 'Unknown')}"
            })
    
    return pd.DataFrame(summary_data)


def compare_before_after_optimization(model_results: Dict[str, Dict[str, Any]],
                                     optimization_results: Dict[str, Dict[str, Any]],
                                     metric_name: str = 'Accuracy') -> pd.DataFrame:
    """
    Compare model performance before and after optimization.
    
    Args:
        model_results: Original model results with test scores
        optimization_results: Optimization results with CV scores
        metric_name: Name of the metric being compared
        
    Returns:
        DataFrame comparing before and after optimization
    """
    comparison_data = []
    
    for model_name in model_results.keys():
        before_result = model_results[model_name]
        after_result = optimization_results.get(model_name, {})
        
        if before_result.get('trained', False) and after_result.get('optimized', False):
            comparison_data.append({
                'Model': model_name,
                f'Before {metric_name}': 'See Test Results',
                f'After CV {metric_name}': f"{after_result['best_score']:.4f}",
                'Improvement': 'Optimized'
            })
        else:
            comparison_data.append({
                'Model': model_name,
                f'Before {metric_name}': 'N/A',
                f'After CV {metric_name}': 'N/A',
                'Improvement': 'Not Available'
            })
    
    return pd.DataFrame(comparison_data)


def retrain_with_best_params(optimization_results: Dict[str, Dict[str, Any]],
                            X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, Any]]:
    """
    Retrain all models with their best parameters on full training set.
    
    Args:
        optimization_results: Optimization results with best models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with retrained model results
    """
    from models import train_single_model
    
    retrained_results = {}
    
    for model_name, opt_result in optimization_results.items():
        if opt_result.get('optimized', False):
            try:
                best_model = opt_result['best_model']
                result = train_single_model(best_model, X_train, y_train, X_test, y_test)
                result['best_params'] = opt_result['best_params']
                result['cv_score'] = opt_result['best_score']
                retrained_results[model_name] = result
            except Exception as e:
                retrained_results[model_name] = {
                    'trained': False,
                    'error': str(e)
                }
        else:
            retrained_results[model_name] = {
                'trained': False,
                'error': 'Optimization failed'
            }
    
    return retrained_results


def get_best_params_summary(optimization_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract best parameters for all optimized models.
    
    Args:
        optimization_results: Optimization results
        
    Returns:
        Dictionary mapping model names to their best parameters
    """
    best_params = {}
    
    for model_name, result in optimization_results.items():
        if result.get('optimized', False):
            best_params[model_name] = result['best_params']
    
    return best_params


def get_optimization_statistics(optimization_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get overall optimization statistics.
    
    Args:
        optimization_results: Optimization results
        
    Returns:
        Dictionary with optimization statistics
    """
    total_models = len(optimization_results)
    optimized_models = sum(1 for r in optimization_results.values() if r.get('optimized', False))
    total_time = sum(r.get('optimization_time', 0) for r in optimization_results.values() 
                    if r.get('optimized', False))
    total_combinations = sum(r.get('n_combinations', 0) for r in optimization_results.values()
                            if r.get('optimized', False))
    
    return {
        'total_models': total_models,
        'optimized_models': optimized_models,
        'failed_models': total_models - optimized_models,
        'total_optimization_time': total_time,
        'total_combinations_tested': total_combinations,
        'average_time_per_model': total_time / optimized_models if optimized_models > 0 else 0
    }
