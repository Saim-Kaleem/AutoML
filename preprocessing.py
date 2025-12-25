import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class DataPreprocessor:
    """
    Handles all preprocessing operations with fit-transform pattern.
    """
    
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.ordinal_encoder = None
        self.feature_names = None
        self.target_encoder = None
        self.preprocessing_config = {}
    
    def handle_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              numeric_strategy: str = 'mean',
                              categorical_strategy: str = 'most_frequent',
                              constant_value: Any = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values in training and test sets.
        
        Args:
            X_train: Training features
            X_test: Test features
            numeric_strategy: Strategy for numeric features ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical features ('most_frequent', 'constant')
            constant_value: Value to use for constant strategy
            
        Returns:
            Tuple of (X_train_imputed, X_test_imputed)
        """
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric features
        if numeric_cols:
            if numeric_strategy == 'constant':
                self.numeric_imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            else:
                self.numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            
            X_train[numeric_cols] = self.numeric_imputer.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = self.numeric_imputer.transform(X_test[numeric_cols])
        
        # Handle categorical features
        if categorical_cols:
            if categorical_strategy == 'constant':
                self.categorical_imputer = SimpleImputer(strategy='constant', 
                                                         fill_value=str(constant_value))
            else:
                self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            
            X_train[categorical_cols] = self.categorical_imputer.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = self.categorical_imputer.transform(X_test[categorical_cols])
        
        self.preprocessing_config['missing_values'] = {
            'numeric_strategy': numeric_strategy,
            'categorical_strategy': categorical_strategy,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols
        }
        
        return X_train, X_test
    
    def remove_outliers(self, X_train: pd.DataFrame, y_train: pd.Series,
                       outlier_indices: List[int]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outlier rows from training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            outlier_indices: Indices of outliers to remove
            
        Returns:
            Tuple of (X_train_cleaned, y_train_cleaned)
        """
        mask = ~X_train.index.isin(outlier_indices)
        X_train_cleaned = X_train[mask].copy()
        y_train_cleaned = y_train[mask].copy()
        
        self.preprocessing_config['outliers_removed'] = len(outlier_indices)
        
        return X_train_cleaned, y_train_cleaned
    
    def encode_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          encoding_type: str = 'onehot',
                          categorical_cols: List[str] = None,
                          exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features.
        
        Args:
            X_train: Training features
            X_test: Test features
            encoding_type: 'onehot' or 'ordinal'
            categorical_cols: List of categorical columns (if None, auto-detect)
            exclude_cols: List of columns to exclude from encoding (e.g., high cardinality)
            
        Returns:
            Tuple of (X_train_encoded, X_test_encoded)
        """
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        if categorical_cols is None:
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude high-cardinality or specified columns
        if exclude_cols:
            # Drop excluded columns
            X_train = X_train.drop(columns=[col for col in exclude_cols if col in X_train.columns])
            X_test = X_test.drop(columns=[col for col in exclude_cols if col in X_test.columns])
            
            # Remove from categorical_cols list
            categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
            
            self.preprocessing_config['excluded_columns'] = exclude_cols
        
        if not categorical_cols:
            self.feature_names = X_train.columns.tolist()
            return X_train, X_test
        
        if encoding_type == 'onehot':
            # One-hot encoding
            X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
            
            # Align columns between train and test
            missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
            for col in missing_cols:
                X_test_encoded[col] = 0
            
            extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
            for col in extra_cols:
                X_test_encoded = X_test_encoded.drop(columns=[col])
            
            # Reorder columns
            X_test_encoded = X_test_encoded[X_train_encoded.columns]
            
            self.feature_names = X_train_encoded.columns.tolist()
            
        else:  # ordinal
            # Ordinal encoding
            for col in categorical_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                
                # Handle unseen categories in test set
                test_values = X_test[col].astype(str)
                X_test[col] = test_values.map(lambda x: le.transform([x])[0] 
                                               if x in le.classes_ else -1)
                
                self.label_encoders[col] = le
            
            X_train_encoded = X_train
            X_test_encoded = X_test
            self.feature_names = X_train_encoded.columns.tolist()
        
        self.preprocessing_config['encoding'] = {
            'type': encoding_type,
            'categorical_cols': categorical_cols,
            'n_features_after': len(self.feature_names)
        }
        
        return X_train_encoded, X_test_encoded
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      scaling_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features.
        
        Args:
            X_train: Training features
            X_test: Test features
            scaling_type: 'standard' or 'minmax'
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if scaling_type == 'standard':
            self.scaler = StandardScaler()
        else:  # minmax
            self.scaler = MinMaxScaler()
        
        feature_names = X_train.columns.tolist()
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=feature_names,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=feature_names,
            index=X_test.index
        )
        
        self.preprocessing_config['scaling'] = {
            'type': scaling_type
        }
        
        return X_train_scaled, X_test_scaled
    
    def encode_target(self, y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode target variable if it's categorical.
        
        Args:
            y_train: Training target
            y_test: Test target
            
        Returns:
            Tuple of (y_train_encoded, y_test_encoded)
        """
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y_train_encoded = self.target_encoder.fit_transform(y_train)
            y_test_encoded = self.target_encoder.transform(y_test)
            
            self.preprocessing_config['target_encoding'] = {
                'classes': self.target_encoder.classes_.tolist()
            }
        else:
            y_train_encoded = y_train.values
            y_test_encoded = y_test.values
        
        return y_train_encoded, y_test_encoded
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.
        
        Returns:
            Preprocessing configuration dictionary
        """
        return self.preprocessing_config


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
               random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to use stratified split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if stratify and len(y.unique()) > 1:
        stratify_param = y
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test


def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series,
                          method: str = 'none',
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance in training data.
    
    Args:
        X_train: Training features
        y_train: Training target
        method: Resampling method ('none', 'undersample', 'oversample', 'smote', 'adasyn')
        random_state: Random seed
        
    Returns:
        Tuple of (X_train_resampled, y_train_resampled)
    """
    if method == 'none':
        return X_train, y_train
    
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'smote':
        # SMOTE requires numeric features only
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    else:
        return X_train, y_train
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train_array, y_train_array)
        
        # Convert back to DataFrame/Series
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        return X_resampled, y_resampled
    except Exception as e:
        # If resampling fails, return original data
        print(f"Resampling failed: {e}. Returning original data.")
        return X_train, y_train


def get_preprocessing_summary(preprocessor: DataPreprocessor, 
                              X_train_original_shape: Tuple[int, int],
                              X_train_final_shape: Tuple[int, int],
                              y_train_original_dist: Dict[Any, int],
                              y_train_final_dist: Dict[Any, int]) -> Dict[str, Any]:
    """
    Generate preprocessing summary report.
    
    Args:
        preprocessor: DataPreprocessor instance
        X_train_original_shape: Original training data shape
        X_train_final_shape: Final training data shape
        y_train_original_dist: Original target distribution
        y_train_final_dist: Final target distribution
        
    Returns:
        Dictionary with preprocessing summary
    """
    return {
        'original_shape': X_train_original_shape,
        'final_shape': X_train_final_shape,
        'rows_removed': X_train_original_shape[0] - X_train_final_shape[0],
        'features_created': X_train_final_shape[1] - X_train_original_shape[1],
        'original_target_distribution': y_train_original_dist,
        'final_target_distribution': y_train_final_dist,
        'preprocessing_config': preprocessor.get_config()
    }


def prepare_data_pipeline(df: pd.DataFrame, target_col: str,
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute complete preprocessing pipeline.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        config: Configuration dictionary with all preprocessing settings
        
    Returns:
        Dictionary containing all processed data and metadata
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df, target_col,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 42),
        stratify=config.get('stratify', True)
    )
    
    original_X_train_shape = X_train.shape
    original_y_train_dist = y_train.value_counts().to_dict()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Remove constant features (truly constant - automatically removed)
    if config.get('remove_constant_features', False) and config.get('constant_features_to_remove'):
        const_cols = config['constant_features_to_remove']
        X_train = X_train.drop(columns=[col for col in const_cols if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in const_cols if col in X_test.columns])
        preprocessor.preprocessing_config['removed_constant_features'] = const_cols
    
    # Remove near-constant features (user option)
    if config.get('remove_near_constant_features', False) and config.get('near_constant_features_to_remove'):
        near_const_cols = config['near_constant_features_to_remove']
        X_train = X_train.drop(columns=[col for col in near_const_cols if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in near_const_cols if col in X_test.columns])
        preprocessor.preprocessing_config['removed_near_constant_features'] = near_const_cols
    
    # Manual feature selection (drop unwanted columns)
    if config.get('manual_feature_selection', False) and config.get('features_to_drop'):
        features_to_drop = config['features_to_drop']
        X_train = X_train.drop(columns=[col for col in features_to_drop if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in features_to_drop if col in X_test.columns])
        preprocessor.preprocessing_config['manually_dropped_features'] = features_to_drop
    
    # Handle missing values
    if config.get('handle_missing', False):
        X_train, X_test = preprocessor.handle_missing_values(
            X_train, X_test,
            numeric_strategy=config.get('numeric_imputation', 'mean'),
            categorical_strategy=config.get('categorical_imputation', 'most_frequent'),
            constant_value=config.get('constant_value', 0)
        )
    
    # Remove outliers (only from training set)
    if config.get('remove_outliers', False) and config.get('outlier_indices'):
        X_train, y_train = preprocessor.remove_outliers(
            X_train, y_train, config['outlier_indices']
        )
    
    # Encode categorical features
    if config.get('encode_categorical', False):
        X_train, X_test = preprocessor.encode_categorical(
            X_train, X_test,
            encoding_type=config.get('encoding_type', 'onehot'),
            exclude_cols=config.get('exclude_high_cardinality_cols', [])
        )
    
    # Scale features
    if config.get('scale_features', False):
        X_train, X_test = preprocessor.scale_features(
            X_train, X_test,
            scaling_type=config.get('scaling_type', 'standard')
        )
    
    # Encode target
    y_train_encoded, y_test_encoded = preprocessor.encode_target(y_train, y_test)
    
    # Handle class imbalance (only on training set)
    if config.get('handle_imbalance', False):
        X_train, y_train_encoded = handle_class_imbalance(
            X_train, y_train_encoded,
            method=config.get('imbalance_method', 'none'),
            random_state=config.get('random_state', 42)
        )
    
    final_X_train_shape = X_train.shape
    final_y_train_dist = pd.Series(y_train_encoded).value_counts().to_dict()
    
    summary = get_preprocessing_summary(
        preprocessor, original_X_train_shape, final_X_train_shape,
        original_y_train_dist, final_y_train_dist
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'preprocessor': preprocessor,
        'summary': summary
    }
