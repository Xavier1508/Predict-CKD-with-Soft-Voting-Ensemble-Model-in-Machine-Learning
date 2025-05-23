import time
import pandas as pd
import numpy as np
import os
import joblib
import logging
import traceback
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, precision_recall_curve, roc_curve, auc, make_scorer, 
                             brier_score_loss, matthews_corrcoef, balanced_accuracy_score, log_loss, 
                             )
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as imPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV, RFE, SelectFromModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import warnings
import shap
import pickle
import matplotlib.cm as cm
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from imblearn.metrics import geometric_mean_score

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ckd_model_training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class CKDPreprocessor:
    def __init__(self, 
                random_state=42, 
                cv_folds=5, 
                test_size=0.2,
                use_smote=True,
                use_feature_selection=True,
                imputation_strategy='knn',
                scaler_type='standard',
                n_jobs=-1):
        """
        Initialize the CKD Preprocessor with configurable parameters
        
        Args:
            random_state: Seed for reproducibility
            cv_folds: Number of cross-validation folds
            test_size: Proportion of data to use for testing
            use_smote: Whether to use SMOTE for handling class imbalance
            use_feature_selection: Whether to perform feature selection
            imputation_strategy: Strategy for imputing missing values ('median', 'knn', 'iterative')
            scaler_type: Type of scaler to use ('standard', 'robust', 'minmax', 'none')
            n_jobs: Number of CPU cores to use (-1 for all available)
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.use_smote = use_smote
        self.use_feature_selection = use_feature_selection
        self.imputation_strategy = imputation_strategy
        self.scaler_type = scaler_type
        self.n_jobs = n_jobs
        
        # Setup paths
        self.base_dir = Path(__file__).parent.absolute()
        self.data_path = self.base_dir / "integrated_data" / "merged_ckd_data.csv"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.base_dir / "plots"
        
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Medical core features with descriptions for better documentation
        self.core_features = {
            'age': 'Patient age in years',
            'bp': 'Blood pressure (mm/Hg)',
            'sc': 'Serum creatinine (mg/dL)',
            'hemo': 'Hemoglobin (g/dL)',
            'bgr': 'Blood glucose random (mg/dL)',
            'bu': 'Blood urea (mg/dL)',
            'al': 'Albumin (0-5)',
            'sg': 'Specific gravity',
            'dm': 'Diabetes mellitus (yes/no)',
            'htn': 'Hypertension (yes/no)',
            'bmi': 'Body mass index',
            'wc': 'White blood cell count',
            'su': 'Sugar (0-5)',
            'pc': 'Pus cell (normal/abnormal)',
            'pcc': 'Pus cell clumps (present/not present)',
            'ba': 'Bacteria (present/not present)',
            'sod': 'Sodium (mEq/L)',
            'pot': 'Potassium (mEq/L)',
            'pcv': 'Packed cell volume',
            'rc': 'Red blood cell count (millions/cmm)',
            'rbc': 'Red blood cells (normal/abnormal)',
            'appetite': 'Appetite (good/poor)'
        }
        
        self.core_feature_names = list(self.core_features.keys())
        self.target = 'target'
        self.best_model = None
        self.feature_importances = None
        
        # Choose scaler based on configuration
        self._initialize_scaler()
        
        # Initialize imputer based on configuration
        self._initialize_imputer()
        
        # Initialize models with optimized configurations
        self._initialize_models()

    def _initialize_scaler(self):
        """Initialize the appropriate scaler based on configuration"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

    def _initialize_imputer(self):
        """Initialize the appropriate imputer based on configuration"""
        if self.imputation_strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.imputation_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        elif self.imputation_strategy == 'iterative':
            self.imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")

    def _initialize_models(self):
        """Initialize all models with optimized configurations"""
        # Determine class weight based on whether SMOTE will be used
        class_weight = None if self.use_smote else 'balanced'
        scale_pos_weight = 1 if self.use_smote else 2
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                class_weight=class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True
            ),
            'XGBoost': XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                learning_rate=0.1,
                n_estimators=100,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist'  # More efficient for large datasets
            ),
            'LightGBM': LGBMClassifier(
                class_weight=class_weight,
                boosting_type='gbdt',
                objective='binary',
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                learning_rate=0.1,
                num_leaves=31,
                max_depth=-1,  # -1 means no limit
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'Logistic Regression': LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                C=1.0,
                solver='saga',  # Efficient for large datasets
                penalty='elasticnet',
                l1_ratio=0.5  # Balance between L1 and L2
            ),
            'SVM': SVC(
                class_weight=class_weight,
                probability=True,
                random_state=self.random_state,
                cache_size=2000,
                verbose=False,
                max_iter=1000,
                tol=1e-3,
                C=1.0,
                kernel='rbf',
                gamma='scale'
            )
        }
        
        # Add stacking ensemble model
        self.models['Stacking'] = StackingClassifier(
            estimators=[
                ('rf', self.models['Random Forest']),
                ('xgb', self.models['XGBoost']),
                ('lgbm', self.models['LightGBM'])
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=self.cv_folds,
            n_jobs=self.n_jobs,
            stack_method='predict_proba'
        )
        
        # Parameter grids for GridSearchCV
        self.param_grids = {
            'Random Forest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 5, 10, 15],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__gamma': [0, 0.1, 0.2]
            },
            'LightGBM': {
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__num_leaves': [31, 63, 127],
                'classifier__max_depth': [-1, 5, 10],
                'classifier__min_child_samples': [20, 50, 100],
                'classifier__subsample': [0.8, 0.9, 1.0]
            },
            'Gradient Boosting': {
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__subsample': [0.8, 0.9, 1.0]
            },
            'Logistic Regression': [
                {
                    'classifier__solver': ['liblinear'],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                {
                    'classifier__solver': ['saga'],
                    'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__l1_ratio': [0.2, 0.5, 0.8]
                }
            ],
            'SVM': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__kernel': ['linear', 'rbf'],  # Focus on most effective kernels
                'classifier__tol': [1e-3],
                'classifier__class_weight': ['balanced'],
                'classifier__shrinking': [True]
            }
        }
        
        # Parameter distributions for RandomizedSearchCV
        self.param_distributions = {
            'Random Forest': {
                'classifier__n_estimators': randint(50, 500),
                'classifier__max_depth': randint(3, 20),
                'classifier__min_samples_split': randint(2, 20),
                'classifier__min_samples_leaf': randint(1, 10),
                'classifier__max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'classifier__learning_rate': uniform(0.01, 0.3),
                'classifier__max_depth': randint(2, 10),
                'classifier__subsample': uniform(0.6, 0.4),
                'classifier__colsample_bytree': uniform(0.6, 0.4),
                'classifier__min_child_weight': randint(1, 10),
                'classifier__gamma': uniform(0, 0.5)
            },
            'LightGBM': {
                'classifier__learning_rate': uniform(0.01, 0.2),
                'classifier__num_leaves': randint(20, 100),
                'classifier__max_depth': randint(3, 10),
                'classifier__min_child_samples': randint(10, 50),
                'classifier__subsample': uniform(0.6, 0.4),
                'classifier__colsample_bytree': uniform(0.6, 0.4),
                'classifier__reg_alpha': uniform(0, 1),
                'classifier__reg_lambda': uniform(0, 1)
            },
            'Gradient Boosting': {
                'classifier__learning_rate': uniform(0.01, 0.3),
                'classifier__n_estimators': randint(50, 500),
                'classifier__max_depth': randint(2, 10),
                'classifier__min_samples_split': randint(2, 20),
                'classifier__subsample': uniform(0.6, 0.4)
            },
            'Logistic Regression': [
                {
                    'classifier__solver': ['liblinear'],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__C': uniform(0.001, 100)
                },
                {
                    'classifier__solver': ['saga'],
                    'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
                    'classifier__C': uniform(0.001, 100),
                    'classifier__l1_ratio': uniform(0, 1)
                }
            ],
            'SVM': {
                'classifier__C': uniform(0.01, 50),  # Narrower range
                'classifier__gamma': ['scale', 'auto'],
                'classifier__kernel': ['linear', 'rbf'],  # Prioritize these kernels
                'classifier__tol': [1e-3, 1e-4],
                'classifier__shrinking': [True],
                'classifier__cache_size': [2000]
            }
        }
        
    def load_data(self):
        """Load and preprocess data"""
        try:
            # Load the data
            df = pd.read_csv(self.data_path)
            
            # Reset index to ensure consistent indexing throughout processing
            df.reset_index(drop=True, inplace=True)
            
            # Log data overview
            logging.info(f"Loaded dataset with shape: {df.shape}")
            
            # Convert target to binary
            df[self.target] = df[self.target].apply(lambda x: 1 if x >= 0.5 else 0)
            
            # Check which core features are available in dataset
            available_features = [f for f in self.core_features if f in df.columns]
            missing_features = [f for f in self.core_features if f not in df.columns]
            
            if missing_features:
                logging.warning(f"Missing features in dataset: {missing_features}")
            
            # Store initial available features as backup
            self.available_features = available_features.copy()
            
            # Handle missing values based on strategy
            df = self._handle_missing_values(df)
            
            # Update available_features with polynomial features if they exist
            if hasattr(self, 'poly_features'):
                # Add only polynomial features that exist in DataFrame
                existing_poly_features = [f for f in self.poly_features if f in df.columns]
                self.available_features.extend(existing_poly_features)
                logging.info(f"Added {len(existing_poly_features)} polynomial features")
            
            # Ensure all available_features actually exist in the dataframe
            self.available_features = [f for f in self.available_features if f in df.columns]
            
            # Prepare features and target with updated available_features
            X = df[self.available_features].copy()
            y = df[self.target].copy()
            
            # Check and handle duplicate column names early
            if X.columns.duplicated().any():
                logging.warning(f"Detected {X.columns.duplicated().sum()} duplicate column names. Removing duplicates.")
                X = X.loc[:, ~X.columns.duplicated()]
                # Update available_features to match unique columns
                self.available_features = X.columns.tolist()
            
            # Remove constant features
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
                    
            if constant_features:
                logging.info(f"Removing constant features: {constant_features}")
                X = X.drop(columns=constant_features)
                # Update available_features after removing constant features
                self.available_features = [f for f in self.available_features if f not in constant_features]
            
            # Identify categorical and numerical columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            logging.info(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")
            
            # Store column types for later use
            self.categorical_cols = categorical_cols
            self.numerical_cols = numerical_cols
            
            # Handle categorical features
            if categorical_cols:
                categorical_features_info = {}
                categorical_cols_to_drop = []
                new_feature_columns = []
                
                for col in categorical_cols:
                    if col not in X.columns:  # Skip if column was already dropped
                        continue
                        
                    unique_values = X[col].nunique()
                    categorical_features_info[col] = unique_values
                    
                    # Handle categorical features based on cardinality
                    if unique_values > 10:  # High cardinality
                        logging.info(f"High cardinality feature detected: {col} with {unique_values} unique values")
                        
                        # Try label encoding for high cardinality
                        try:
                            # Store the mapping for interpretability
                            label_map = {v: i for i, v in enumerate(X[col].dropna().unique())}
                            X[col] = X[col].map(label_map)
                            X[col] = X[col].fillna(-1)  # Mark missing values with -1
                            
                            # Save mapping for later interpretation
                            if not hasattr(self, 'label_encodings'):
                                self.label_encodings = {}
                            self.label_encodings[col] = label_map
                            
                            logging.info(f"Applied label encoding to {col}")
                        except Exception as e:
                            logging.warning(f"Failed to encode {col}: {str(e)}")
                            categorical_cols_to_drop.append(col)
                    else:
                        # Use one-hot encoding for low cardinality
                        try:
                            # Ensure unique prefix for one-hot encoded features to avoid duplicates
                            prefix = f"{col}_encoded"
                            # Check for potential duplicate column names
                            existing_prefixes = [c for c in X.columns if c.startswith(prefix)]
                            if existing_prefixes:
                                prefix = f"{col}_encoded_{len(existing_prefixes)}"
                            
                            # Get dummies for this column only
                            dummies = pd.get_dummies(X[col], prefix=prefix, drop_first=False, dummy_na=True)
                            
                            # Track new feature names
                            new_feature_columns.extend(dummies.columns.tolist())
                            
                            # Check for duplicate column names before concat
                            overlap_columns = set(X.columns).intersection(set(dummies.columns))
                            if overlap_columns:
                                logging.warning(f"Duplicate columns detected during one-hot encoding: {overlap_columns}")
                                # Rename duplicate columns in dummies
                                for dup_col in overlap_columns:
                                    new_col = f"{dup_col}_v2"
                                    dummies.rename(columns={dup_col: new_col}, inplace=True)
                                    # Update the tracked columns
                                    new_feature_columns.remove(dup_col)
                                    new_feature_columns.append(new_col)
                            
                            # Add the dummies to the dataframe
                            X = pd.concat([X, dummies], axis=1)
                            
                            # Mark the original column for removal
                            categorical_cols_to_drop.append(col)
                            logging.info(f"Applied one-hot encoding to {col}, creating {dummies.shape[1]} new features")
                        except Exception as e:
                            logging.warning(f"Failed to one-hot encode {col}: {str(e)}")
                            categorical_cols_to_drop.append(col)
                
                # Drop original categorical columns after encoding
                if categorical_cols_to_drop:
                    # Only drop columns that exist
                    cols_to_drop = [col for col in categorical_cols_to_drop if col in X.columns]
                    if cols_to_drop:
                        X = X.drop(columns=cols_to_drop)
                        logging.info(f"Dropped {len(cols_to_drop)} original categorical columns after encoding")
                        
                        # Update available_features to remove dropped columns and add new ones
                        self.available_features = [f for f in self.available_features if f not in cols_to_drop]
                        self.available_features.extend(new_feature_columns)
            
            # Reset index after categorical processing to avoid indexing issues
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            # Check for duplicate column names after processing categorical features
            if X.columns.duplicated().any():
                logging.warning(f"Detected {X.columns.duplicated().sum()} duplicate column names after categorical processing. Removing duplicates.")
                X = X.loc[:, ~X.columns.duplicated()]
                # Update available_features to match unique columns
                self.available_features = X.columns.tolist()
            
            # Handle numerical features with potential errors
            object_cols_to_drop = []
            for col in X.select_dtypes(include=['object']).columns:
                if col not in X.columns:  # Skip if column was already dropped
                    continue
                    
                logging.warning(f"Column {col} is still 'object' type after categorical processing. Attempting conversion.")
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col].fillna(X[col].median(), inplace=True)
                    logging.info(f"Successfully converted {col} to numeric")
                except Exception as e:
                    logging.warning(f"Could not convert {col} to numeric: {str(e)}. Dropping column.")
                    object_cols_to_drop.append(col)
            
            # Drop any object columns that couldn't be converted
            if object_cols_to_drop:
                # Only drop columns that exist
                cols_to_drop = [col for col in object_cols_to_drop if col in X.columns]
                if cols_to_drop:
                    X = X.drop(columns=cols_to_drop)
                    # Update available_features again to remove these columns
                    self.available_features = [f for f in self.available_features if f not in cols_to_drop]
            
            # Reset index again after numerical processing
            X = X.reset_index(drop=True)
            
            # Check for duplicate columns after handling numerical features
            if X.columns.duplicated().any():
                logging.warning(f"Detected {X.columns.duplicated().sum()} duplicate column names after numerical conversion. Removing duplicates.")
                X = X.loc[:, ~X.columns.duplicated()]
                # Update available_features to match unique columns
                self.available_features = X.columns.tolist()
            
            # Check for any remaining issues
            if X.isnull().sum().sum() > 0:
                null_counts = X.isnull().sum()
                columns_with_nulls = null_counts[null_counts > 0]
                logging.warning(f"Remaining null values after preprocessing: {columns_with_nulls}")
                
                # Impute remaining nulls
                for col in columns_with_nulls.index:
                    if col not in X.columns:  # Skip if column was dropped
                        continue
                        
                    if X[col].dtype.kind in 'if':  # If numeric
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "UNKNOWN", inplace=True)
            
            # Check for infinite values
            if np.any(np.isinf(X.select_dtypes(include=['float64']).values)):
                logging.warning("Detected infinite values in dataset. Replacing with NaN and then imputing.")
                # Replace infinities with NaN
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Impute NaNs with median for each column
                for col in X.columns:
                    if X[col].isna().any():
                        if X[col].dtype.kind in 'if':  # If numeric
                            X[col].fillna(X[col].median(), inplace=True)
                        else:
                            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "UNKNOWN", inplace=True)
            
            # Save the original data distribution
            if hasattr(self, 'plot_data_distribution'):
                self.plot_data_distribution(y, title="Original Class Distribution", 
                                        filename="original_class_distribution.png")
            
            # Store final feature names
            self.final_feature_names = X.columns.tolist()
            
            # Verify that all columns are numeric for model training
            non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
            if non_numeric_cols:
                logging.warning(f"Found non-numeric columns after preprocessing: {non_numeric_cols}")
                for col in non_numeric_cols:
                    if col not in X.columns:  # Skip if column was dropped
                        continue
                        
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col].fillna(X[col].median(), inplace=True)
                    except Exception as e:
                        logging.error(f"Cannot convert column {col} to numeric. Dropping it: {str(e)}")
                        X = X.drop(columns=[col])
                        if col in self.final_feature_names:
                            self.final_feature_names.remove(col)
            
            # Final reset index to ensure alignment
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            # Check for feature collinearity with better error handling
            if X.shape[1] > 1 and X.shape[0] > X.shape[1]:
                try:
                    # Use correlation matrix rather than spearmanr for better robustness
                    corr_matrix = X.corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                    
                    if high_corr_features:
                        logging.info(f"Detected {len(high_corr_features)} features with high correlation (>0.95). Consider feature selection.")
                        # Save information for later analysis but don't drop automatically
                        self.high_correlation_features = high_corr_features
                        
                        # Identify pairs of highly correlated features
                        high_corr_pairs = []
                        for i in range(len(X.columns)):
                            for j in range(i+1, len(X.columns)):
                                if i < len(corr_matrix.index) and j < len(corr_matrix.columns) and abs(corr_matrix.iloc[i, j]) > 0.95:
                                    high_corr_pairs.append((X.columns[i], X.columns[j], corr_matrix.iloc[i, j]))
                        
                        if high_corr_pairs:
                            logging.info(f"Top highly correlated feature pairs: {high_corr_pairs[:5]}")
                            self.high_correlation_pairs = high_corr_pairs
                except Exception as e:
                    logging.warning(f"Could not check for feature collinearity: {str(e)}")
            
            # Final check to ensure all column names are valid and unique
            if X.columns.duplicated().any():
                logging.warning("Final check: Still found duplicate column names. Ensuring uniqueness.")
                X = X.loc[:, ~X.columns.duplicated()]
                self.final_feature_names = X.columns.tolist()
            
            # Ensure final feature list matches actual DataFrame columns
            if set(self.final_feature_names) != set(X.columns):
                logging.warning("Inconsistency between stored feature names and DataFrame columns. Fixing.")
                self.final_feature_names = X.columns.tolist()
                    
            logging.info(f"Data preprocessing complete. Final dataset shape: {X.shape}")
            
            return X, y
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            traceback.print_exc()  # Print the full traceback for debugging
            raise

    def _handle_missing_values(self, df):
        """Handle missing values in the DataFrame based on strategy and perform feature engineering"""
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # 1. Handle missing values
        missing_data = df[self.available_features].isnull().sum()
        features_with_missing = missing_data[missing_data > 0].index.tolist()
        
        if features_with_missing:
            logging.info(f"Features with missing values: {features_with_missing}")
            logging.info(f"Using {self.imputation_strategy} imputation strategy")
            
            if self.imputation_strategy == 'median':
                for feature in self.available_features:
                    if feature in df.columns and df[feature].isnull().any():
                        # Check if column is numeric before applying median
                        if np.issubdtype(df[feature].dtype, np.number):
                            df[feature].fillna(df[feature].median(), inplace=True)
                        else:
                            # For non-numeric columns, use mode instead
                            df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else "UNKNOWN", inplace=True)
                            logging.info(f"Used mode imputation for non-numeric feature: {feature}")
            
            elif self.imputation_strategy == 'knn':
                try:
                    # Ensure we only use numeric features for KNN imputation
                    numeric_features = [f for f in self.available_features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
                    
                    if numeric_features:
                        # Check if there are enough samples for KNN
                        n_neighbors = min(5, len(df) - 1)
                        if n_neighbors < 1:
                            raise ValueError("Not enough samples for KNN imputation")
                        
                        # KNN imputation for numeric features
                        features_subset = df[numeric_features].copy()
                        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
                        imputed_data = imputer.fit_transform(features_subset)
                        
                        # Replace the original data with imputed values
                        for i, feature in enumerate(numeric_features):
                            df[feature] = imputed_data[:, i]
                        
                        logging.info(f"Applied KNN imputation to {len(numeric_features)} numeric features")
                    
                    # For non-numeric features, use mode imputation
                    non_numeric_features = [f for f in features_with_missing if f in df.columns and f not in numeric_features]
                    for feature in non_numeric_features:
                        df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else "UNKNOWN", inplace=True)
                        logging.info(f"Used mode imputation for non-numeric feature: {feature}")
                except Exception as e:
                    logging.error(f"KNN imputation failed: {str(e)}. Falling back to median imputation.")
                    # Fallback to median imputation if KNN fails
                    for feature in self.available_features:
                        if feature in df.columns and df[feature].isnull().any():
                            if np.issubdtype(df[feature].dtype, np.number):
                                df[feature].fillna(df[feature].median(), inplace=True)
                            else:
                                df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else "UNKNOWN", inplace=True)
                    
            elif self.imputation_strategy == 'iterative':
                try:
                    # Ensure we only use numeric features for iterative imputation
                    numeric_features = [f for f in self.available_features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
                    
                    if numeric_features:
                        # Iterative imputation (MICE) for numeric features
                        features_subset = df[numeric_features].copy()
                        
                        # Check for infinite values and replace them
                        features_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
                        
                        # Check if there are still missing values after replacing infinities
                        if features_subset.isnull().sum().sum() > 0:
                            imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
                            imputed_data = imputer.fit_transform(features_subset)
                            
                            # Replace the original data with imputed values
                            for i, feature in enumerate(numeric_features):
                                df[feature] = imputed_data[:, i]
                            
                            logging.info(f"Applied iterative imputation to {len(numeric_features)} numeric features")
                    
                    # For non-numeric features, use mode imputation
                    non_numeric_features = [f for f in features_with_missing if f in df.columns and f not in numeric_features]
                    for feature in non_numeric_features:
                        df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else "UNKNOWN", inplace=True)
                        logging.info(f"Used mode imputation for non-numeric feature: {feature}")
                except Exception as e:
                    logging.error(f"Iterative imputation failed: {str(e)}. Falling back to median imputation.")
                    # Fallback to median imputation if iterative imputation fails
                    for feature in self.available_features:
                        if feature in df.columns and df[feature].isnull().any():
                            if np.issubdtype(df[feature].dtype, np.number):
                                df[feature].fillna(df[feature].median(), inplace=True)
                            else:
                                df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else "UNKNOWN", inplace=True)
        else:
            logging.info("No missing values found in features")

        # 2. Feature Engineering: Add BUN-to-creatinine ratio
        if 'bu' in df.columns and 'sc' in df.columns:
            try:
                # First, ensure both columns are numeric
                for col in ['bu', 'sc']:
                    if not np.issubdtype(df[col].dtype, np.number):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logging.info(f"Converted {col} to numeric for ratio calculation")
                
                # Replace zeros in denominator to avoid division by zero
                df['sc_safe'] = df['sc'].copy()
                df.loc[df['sc_safe'] <= 0, 'sc_safe'] = np.finfo(float).eps
                
                # Calculate ratio
                df['bun_to_creatinine_ratio'] = df['bu'] / df['sc_safe']
                
                # Handle any infinite or NaN values
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Impute missing values in the new feature
                if df['bun_to_creatinine_ratio'].isnull().any():
                    median_ratio = df['bun_to_creatinine_ratio'].median()
                    df['bun_to_creatinine_ratio'].fillna(median_ratio, inplace=True)
                    logging.info(f"Imputed {df['bun_to_creatinine_ratio'].isnull().sum()} missing values in BUN-to-creatinine ratio with median: {median_ratio}")
                
                # Add validation to ensure values are within reasonable clinical range (typically 10-20 for BUN/Cr ratio)
                upper_limit = df['bun_to_creatinine_ratio'].quantile(0.99)  # Use 99th percentile as upper limit
                lower_limit = max(1, df['bun_to_creatinine_ratio'].quantile(0.01))  # Use 1st percentile or 1, whichever is higher
                
                # Cap extreme values
                extreme_high = (df['bun_to_creatinine_ratio'] > upper_limit).sum()
                extreme_low = (df['bun_to_creatinine_ratio'] < lower_limit).sum()
                
                if extreme_high > 0 or extreme_low > 0:
                    df['bun_to_creatinine_ratio'] = df['bun_to_creatinine_ratio'].clip(lower_limit, upper_limit)
                    logging.info(f"Capped {extreme_high} high and {extreme_low} low extreme values in BUN-to-creatinine ratio")
                
                # Drop the temporary column
                df.drop(columns=['sc_safe'], inplace=True)
                
                # Add to available features but store in a separate list for engineered features
                if 'bun_to_creatinine_ratio' not in self.available_features:
                    self.available_features.append('bun_to_creatinine_ratio')
                    # Create engineered_features attribute if it doesn't exist
                    if not hasattr(self, 'engineered_features'):
                        self.engineered_features = []
                    if 'bun_to_creatinine_ratio' not in self.engineered_features:
                        self.engineered_features.append('bun_to_creatinine_ratio')
                    logging.info("Added BUN-to-creatinine ratio feature")
            except Exception as e:
                logging.error(f"Failed to create BUN-to-creatinine ratio: {str(e)}")
        else:
            logging.warning("Cannot create BUN-to-creatinine ratio: 'bu' or 'sc' columns missing")

        # Check for duplicate columns before doing anything else
        if df.columns.duplicated().any():
            logging.warning(f"Found {df.columns.duplicated().sum()} duplicate column names. Fixing...")
            # Create a mapping of old column names to unique new names
            cols_mapping = {}
            for i, col in enumerate(df.columns):
                if col in cols_mapping:
                    cols_mapping[col].append(i)
                else:
                    cols_mapping[col] = [i]
            
            # Rename duplicate columns
            new_columns = list(df.columns)
            for col, indices in cols_mapping.items():
                if len(indices) > 1:
                    # First occurrence keeps the original name
                    for idx in indices[1:]:
                        suffix = 1
                        new_col = f"{col}_{suffix}"
                        # Ensure the new name is unique
                        while new_col in new_columns:
                            suffix += 1
                            new_col = f"{col}_{suffix}"
                        new_columns[idx] = new_col
                        logging.info(f"Renamed duplicate column '{col}' to '{new_col}'")
            
            # Apply the new column names
            df.columns = new_columns
            logging.info("Fixed duplicate column names")

        # Check for highly correlated features before creating polynomial features
        try:
            # Ensure we have enough numeric features to check correlations
            numeric_features = [f for f in self.available_features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
            
            # First, handle any remaining infinities or NaNs in numeric features
            for feature in numeric_features:
                df[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
                if df[feature].isnull().any():
                    df[feature].fillna(df[feature].median(), inplace=True)
            
            if len(numeric_features) > 1:
                # Calculate correlation matrix
                corr_matrix = df[numeric_features].corr().abs()
                # Get upper triangle of correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find features with correlation > 0.95
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                
                if to_drop:
                    logging.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
                    # Update available_features to exclude dropped features
                    self.available_features = [f for f in self.available_features if f not in to_drop]
                    # Don't actually drop from dataframe as we might need them for other purposes
            else:
                logging.info("Not enough numeric features to check for correlations")
        except Exception as e:
            logging.error(f"Failed to check for correlated features: {str(e)}")

        # 3. Feature Engineering: Add polynomial features with proper feature selection
        try:
            # Initialize target-related variables
            has_target = False
            y = None
            
            # Check if we have the target variable for feature selection
            if hasattr(self, 'y') and self.y is not None:
                has_target = True
                y = self.y
                # Ensure y has the correct length
                if len(y) != len(df):
                    logging.warning(f"Target length ({len(y)}) doesn't match data length ({len(df)}). Target will not be used for feature selection.")
                    has_target = False
            else:
                logging.warning("Target variable not available for polynomial feature selection")
            
            # Ensure all core features exist before creating polynomial features
            if not hasattr(self, 'core_features'):
                self.core_features = [f for f in self.available_features if f in df.columns]
                logging.info("Initialized core_features with available features")
            
            existing_core_features = [col for col in self.core_features if col in df.columns]
            
            if len(existing_core_features) > 1:  # Need at least 2 features for interactions
                # Filter out non-numeric and constant features
                numeric_features = [f for f in existing_core_features if np.issubdtype(df[f].dtype, np.number)]
                
                # Ensure all features are finite
                for feature in numeric_features:
                    df[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
                    if df[feature].isnull().any():
                        df[feature].fillna(df[feature].median(), inplace=True)
                
                if len(numeric_features) > 1:
                    # Check for constant features
                    variances = df[numeric_features].var()
                    non_constant_features = variances[variances > 0].index.tolist()
                    
                    if len(non_constant_features) > 1:
                        # Create polynomial features
                        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                        
                        # Ensure data is finite before polynomial transformation
                        features_subset = df[non_constant_features].copy()
                        
                        # Generate polynomial features
                        poly_features = poly.fit_transform(features_subset)
                        
                        # Only keep the interaction terms (exclude original features)
                        n_orig_features = len(non_constant_features)
                        poly_features = poly_features[:, n_orig_features:]
                        
                        # Get feature names (excluding original features)
                        orig_poly_feature_names = poly.get_feature_names_out(non_constant_features)[n_orig_features:]
                        
                        # Create unique feature names to avoid duplicates
                        poly_feature_names = []
                        used_names = set(df.columns)  # Start with existing column names
                        
                        for name in orig_poly_feature_names:
                            if name in used_names:
                                # Name already exists, create a unique version
                                base_name = name
                                suffix = 1
                                new_name = f"{base_name}_{suffix}"
                                while new_name in used_names:
                                    suffix += 1
                                    new_name = f"{base_name}_{suffix}"
                                poly_feature_names.append(new_name)
                                used_names.add(new_name)
                            else:
                                poly_feature_names.append(name)
                                used_names.add(name)
                        
                        # Convert polynomial features to DataFrame
                        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
                        
                        # Verify no duplicated column names remain
                        assert not poly_df.columns.duplicated().any(), "Duplicate column names still present in polynomial features"
                        
                        # Handle any infinite or NaN values in polynomial features
                        poly_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        
                        # Check if we have any NaN values and handle them
                        if poly_df.isnull().any().any():
                            for col in poly_df.columns:
                                # Use median for each column individually
                                if poly_df[col].isnull().any():
                                    poly_df[col].fillna(poly_df[col].median() if not poly_df[col].dropna().empty else 0, inplace=True)
                        
                        # Remove any columns with constant values or very low variance after transformation
                        variances = poly_df.var()
                        valid_cols = variances[variances > 1e-5].index.tolist()
                        poly_df = poly_df[valid_cols]
                        
                        if not poly_df.empty:
                            # Apply feature selection if target is available
                            if has_target:
                                try:
                                    # Determine maximum number of features to select (min between 20 and available features)
                                    max_k = min(20, poly_df.shape[1])
                                    
                                    # Import necessary libraries here to avoid potential issues
                                    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
                                    
                                    # Determine the appropriate scoring function based on target type
                                    if hasattr(self, 'problem_type') and self.problem_type == 'classification':
                                        score_func = mutual_info_classif
                                    else:
                                        # Default to f_classif for regression or unknown problem type
                                        score_func = f_classif
                                    
                                    # Apply feature selection
                                    selector = SelectKBest(score_func=score_func, k=max_k)
                                    selector.fit(poly_df, y)
                                    selected_poly_features = poly_df.columns[selector.get_support()]
                                    
                                    logging.info(f"Selected {len(selected_poly_features)} out of {len(valid_cols)} polynomial features using SelectKBest")
                                    
                                    # Filter poly_df to keep only selected features
                                    poly_df = poly_df[selected_poly_features]
                                except Exception as e:
                                    logging.error(f"Feature selection failed: {str(e)}. Using correlation filtering instead.")
                                    # Continue with correlation filtering as fallback
                            
                            # Apply correlation filtering (as before or as fallback if feature selection fails)
                            try:
                                full_df = pd.concat([df[non_constant_features], poly_df], axis=1)
                                
                                # Check for duplicate column names before calculating correlation
                                if full_df.columns.duplicated().any():
                                    logging.warning("Duplicate columns found before correlation calculation, fixing...")
                                    
                                    # Rename duplicate columns
                                    new_columns = list(full_df.columns)
                                    seen_cols = set()
                                    for i, col in enumerate(new_columns):
                                        if col in seen_cols:
                                            suffix = 1
                                            new_col = f"{col}_{suffix}"
                                            while new_col in seen_cols:
                                                suffix += 1
                                                new_col = f"{col}_{suffix}"
                                            new_columns[i] = new_col
                                            seen_cols.add(new_col)
                                        else:
                                            seen_cols.add(col)
                                    
                                    full_df.columns = new_columns
                                
                                # Calculate correlation matrix 
                                corr_matrix = full_df.corr().abs()
                                
                                # Filter polynomial features
                                poly_features_to_keep = []
                                for poly_col in poly_df.columns:
                                    if poly_col in corr_matrix.columns:
                                        # Calculate correlation with original features
                                        original_feature_correlations = [
                                            corr_matrix.loc[poly_col, feat] 
                                            for feat in non_constant_features 
                                            if feat in corr_matrix.index and poly_col in corr_matrix.index
                                        ]
                                        
                                        if original_feature_correlations:
                                            max_corr = max(original_feature_correlations)
                                            if max_corr < 0.95:  # Only keep if correlation with original features is less than 0.95
                                                poly_features_to_keep.append(poly_col)
                                
                                if poly_features_to_keep:
                                    # Keep only valuable polynomial features
                                    poly_df = poly_df[poly_features_to_keep]
                                    
                                    # Check for column name conflicts before concatenation
                                    conflicts = set(df.columns).intersection(set(poly_df.columns))
                                    if conflicts:
                                        for col in conflicts:
                                            suffix = 1
                                            new_col = f"{col}_poly{suffix}"
                                            while new_col in df.columns or new_col in poly_df.columns:
                                                suffix += 1
                                                new_col = f"{col}_poly{suffix}"
                                            poly_df.rename(columns={col: new_col}, inplace=True)
                                    
                                    # Final check for duplicate column names in poly_df
                                    assert not poly_df.columns.duplicated().any(), "Duplicate column names in polynomial features"
                                    
                                    # Concatenate with original DataFrame
                                    df = pd.concat([df, poly_df], axis=1)
                                    
                                    # Final check for duplicate column names in combined df
                                    if df.columns.duplicated().any():
                                        raise ValueError(f"Duplicate column names after concatenation: {df.columns[df.columns.duplicated()]}")
                                    
                                    # Store polynomial features in a separate attribute instead of core_features
                                    if not hasattr(self, 'poly_features'):
                                        self.poly_features = []
                                    self.poly_features = list(poly_df.columns)
                                    
                                    # Add polynomial features to available_features but not to core_features
                                    for feat in poly_df.columns:
                                        if feat not in self.available_features:
                                            self.available_features.append(feat)
                                        # Also add to engineered_features for tracking
                                        if not hasattr(self, 'engineered_features'):
                                            self.engineered_features = []
                                        if feat not in self.engineered_features:
                                            self.engineered_features.append(feat)
                                    
                                    logging.info(f"Added {len(poly_features_to_keep)} polynomial features after filtering")
                                else:
                                    logging.warning("All polynomial features were highly correlated with original features")
                            except Exception as e:
                                logging.error(f"Correlation filtering failed: {str(e)}")
                                # If correlation filtering fails, we won't add any polynomial features
                        else:
                            logging.warning("No valid polynomial features created after filtering constant values")
                    else:
                        logging.warning("Not enough non-constant features for polynomial transformation")
                else:
                    logging.warning("Not enough numeric features for polynomial transformation")
            else:
                logging.warning("Not enough core features available for polynomial transformation")
        except Exception as e:
            logging.error(f"Failed to create polynomial features: {str(e)}")
            import traceback
            traceback.print_exc()

        # Final check for any remaining missing values
        if df[self.available_features].isnull().any().any():
            for feature in self.available_features:
                if feature in df.columns and df[feature].isnull().any():
                    if np.issubdtype(df[feature].dtype, np.number):
                        df[feature].fillna(df[feature].median(), inplace=True)
                    else:
                        df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else "UNKNOWN", inplace=True)
            logging.info("Filled remaining missing values in final check")
        
        # Final verification of no duplicate column names
        if df.columns.duplicated().any():
            logging.warning(f"Final dataframe still has duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
            # Create a unique set of column names as a last resort
            df.columns = pd.Index([f"{col}_{i}" if i > 0 else col for i, col in enumerate(df.columns.duplicated(keep=False).cumsum())])
            logging.info("Applied final fix for duplicate column names")
        
        # Return the processed dataframe
        return df

    def handle_imbalance(self, X, y, method='smote'):
        """Apply oversampling/undersampling to handle class imbalance"""
        try:
            # Calculate class distribution before resampling
            class_dist_before = pd.Series(y).value_counts(normalize=True)
            logging.info(f"Class distribution before resampling: {class_dist_before.to_dict()}")
            
            if method == 'none':
                logging.info("Skipping class balancing")
                return X, y
                
            elif method == 'smote':
                sampler = SMOTE(random_state=self.random_state)
            
            elif method == 'adasyn':
                sampler = ADASYN(random_state=self.random_state)
                
            elif method == 'borderline_smote':
                sampler = BorderlineSMOTE(random_state=self.random_state)
                
            elif method == 'undersampling':
                sampler = RandomUnderSampler(random_state=self.random_state)
                
            elif method == 'smote_tomek':
                sampler = SMOTETomek(random_state=self.random_state)
                
            elif method == 'smote_enn':
                sampler = SMOTEENN(random_state=self.random_state)
                
            else:
                logging.warning(f"Unknown sampling method: {method}. Using SMOTE instead.")
                sampler = SMOTE(random_state=self.random_state)
                
            # Handle both DataFrame and numpy array inputs
            is_df = isinstance(X, pd.DataFrame)
            columns = X.columns if is_df else None
            target_name = y.name if hasattr(y, 'name') else self.target
            
            # Perform resampling
            if isinstance(X, pd.DataFrame):
                X_res, y_res = sampler.fit_resample(X.values, y)
                X_res = pd.DataFrame(X_res, columns=X.columns)
            else:
                X_res, y_res = sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series with proper indices if the input was a DataFrame
            if is_df:
                X_res = pd.DataFrame(X_res, columns=columns)
                y_res = pd.Series(y_res, name=target_name)
            else:
                X_res = np.array(X_res)
            
            # Make sure indices are reset and aligned between X_res and y_res
            if hasattr(X_res, 'reset_index'):
                X_res.reset_index(drop=True, inplace=True)
            if hasattr(y_res, 'reset_index'):
                y_res.reset_index(drop=True, inplace=True)
            
            # Calculate and log class distribution after resampling
            class_dist_after = pd.Series(y_res).value_counts(normalize=True)
            logging.info(f"Class distribution after {method}: {class_dist_after.to_dict()}")
            
            # Plot the resampled class distribution
            self.plot_data_distribution(y_res, title=f"Class Distribution After {method.upper()}", 
                                        filename=f"class_distribution_after_{method}.png")
            
            return X_res, y_res
        except Exception as e:
            logging.error(f"Class balancing with {method} failed: {str(e)}")
            logging.info("Returning original data")
            return X, y

    def plot_data_distribution(self, y, title="Class Distribution", filename=None):
        """Plot and save the class distribution"""
        plt.figure(figsize=(8, 6))
        # Ensure y is a pandas Series or numpy array
        ax = sns.countplot(x=y)
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points')
        
        # Ensure plots_dir exists
        if filename:
            # Make sure the plots directory exists
            if not hasattr(self, 'plots_dir'):
                import os
                from pathlib import Path
                self.plots_dir = Path('plots')
                os.makedirs(self.plots_dir, exist_ok=True)
                
            plt.savefig(self.plots_dir / filename, bbox_inches='tight')
            plt.close()  # Close figure after saving to prevent memory leaks
        else:
            plt.show()
            plt.close()

    def select_features(self, X, y, method='select_k_best', k=10, feature_names=None):
        """Perform feature selection"""
        try:
            if not self.use_feature_selection:
                logging.info("Feature selection disabled")
                return X, list(X.columns) if hasattr(X, 'columns') else list(range(X.shape[1])), None
                
            logging.info(f"Performing feature selection using {method}")
            
            # If feature_names is not provided, try to get from X (if DataFrame)
            if feature_names is None:
                if hasattr(X, 'columns'):
                    feature_names = X.columns.tolist()
                else:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Save original index if X is a DataFrame
            original_index = X.index if isinstance(X, pd.DataFrame) else None
            
            if method == 'select_k_best':
                selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
                X_new = selector.fit_transform(X, y)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                
                # Return as DataFrame with original index if input was DataFrame
                if isinstance(X, pd.DataFrame):
                    X_new = pd.DataFrame(X_new, columns=selected_features, index=original_index)
                    
                return X_new, selected_features, selector
                
            elif method == 'rfe':
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]), step=1)
                X_new = selector.fit_transform(X, y)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                
                if isinstance(X, pd.DataFrame):
                    X_new = pd.DataFrame(X_new, columns=selected_features, index=original_index)
                    
                return X_new, selected_features, selector
                
            elif method == 'rfecv':
                estimator = RandomForestClassifier(random_state=self.random_state)
                selector = RFECV(
                    estimator=estimator, 
                    step=1, 
                    cv=StratifiedKFold(n_splits=5),
                    scoring='f1'
                )
                X_new = selector.fit_transform(X, y)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                
                if isinstance(X, pd.DataFrame):
                    X_new = pd.DataFrame(X_new, columns=selected_features, index=original_index)
                    
                return X_new, selected_features, selector
                
            elif method == 'select_from_model':
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                selector = SelectFromModel(estimator, threshold='median')
                X_new = selector.fit_transform(X, y)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                
                if isinstance(X, pd.DataFrame):
                    X_new = pd.DataFrame(X_new, columns=selected_features, index=original_index)
                    
                return X_new, selected_features, selector
                
            else:
                logging.warning(f"Unknown feature selection method: {method}. Using all features.")
                return X, feature_names, None
            
        except Exception as e:
            logging.error(f"Feature selection failed: {str(e)}")
            return X, feature_names, None

    def evaluate_model(self, model, X_test, y_test, feature_names=None, model_name="Model"):
        """Evaluate model performance with comprehensive metrics"""
        
        # Initialize curve_data at the beginning to avoid UnboundLocalError
        curve_data = {
            'fpr': np.array([0, 1]),
            'tpr': np.array([0, 1]), 
            'roc_auc': 0,
            'precision': np.array([0, 1]),
            'recall': np.array([1, 0]),
            'pr_auc': 0
        }
        
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)

        try:
            y_pred = model.predict(X_test)
            
            # Check if model is making constant predictions
            if len(np.unique(y_pred)) == 1:
                logging.warning(f"{model_name} produced constant predictions. Evaluation skipped.")
                return {'constant_predictions': True}, {}, curve_data
            
            # Get prediction probabilities safely
            try:
                y_proba = model.predict_proba(X_test)[:,1]
            except (AttributeError, IndexError) as e:
                logging.warning(f"{model_name} doesn't support predict_proba or has unexpected output format: {str(e)}")
                # Create a dummy probability based on predictions (not ideal but allows evaluation to continue)
                y_proba = y_pred.astype(float)
            
            # Basic classification metrics with error handling
            metrics = {}
            try:
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate confusion matrix values for additional metrics
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # Add specificity and geometric mean
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['specificity'] = specificity
                metrics['gmean'] = np.sqrt(metrics['recall'] * specificity)
                
                # Only calculate ROC AUC if we have both classes in y_test
                if len(np.unique(y_test)) > 1:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                else:
                    metrics['roc_auc'] = 0.5  # Default for single-class cases
                    logging.warning(f"Test set contains only one class. ROC AUC set to 0.5.")
            except Exception as metric_error:
                logging.error(f"Error calculating basic metrics: {str(metric_error)}")
            
            # Generate classification report
            try:
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            except Exception as report_error:
                logging.error(f"Error generating classification report: {str(report_error)}")
                class_report = {}
            
            # Initialize variables for threshold optimization
            precision_curve = np.array([0, 1])
            recall_curve = np.array([1, 0])
            pr_thresholds = np.array([0, 1])
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
            roc_thresholds = np.array([0, 1])
            f1_scores = np.array([0, 0])
            youden_j = np.array([0, 0])
            pr_auc = 0
            
            # Threshold optimization - with error handling
            try:
                # Check if we have enough unique values in y_test for curve calculation
                if len(np.unique(y_test)) > 1 and len(np.unique(y_proba)) > 1:
                    # Calculate precision-recall curve
                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
                    
                    # Calculate ROC curve for Youden's J statistic
                    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
                    
                    # Calculate PR AUC
                    pr_auc = auc(recall_curve, precision_curve)
                    
                    # Calculate Youden's J statistic (sensitivity + specificity - 1)
                    youden_j = tpr + (1 - fpr) - 1
                    optimal_idx_j = np.argmax(youden_j)
                    optimal_threshold_j = roc_thresholds[optimal_idx_j]
                    
                    # Also calculate F1-optimal threshold
                    with np.errstate(divide='ignore', invalid='ignore'):
                        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
                        # Replace any NaN values with 0
                        f1_scores = np.nan_to_num(f1_scores, nan=0.0)
                    
                    # Remove last element which doesn't have a corresponding threshold
                    f1_scores = f1_scores[:-1]
                    
                    if f1_scores.size > 0 and len(f1_scores) > 0:
                        optimal_idx_f1 = np.argmax(f1_scores)
                        optimal_threshold_f1 = pr_thresholds[optimal_idx_f1]
                        
                        # Store both thresholds in metrics
                        metrics['optimal_threshold_youden'] = optimal_threshold_j
                        metrics['optimal_threshold_f1'] = optimal_threshold_f1
                        
                        # Use Youden's J as the primary optimal threshold
                        optimal_threshold = optimal_threshold_j
                        metrics['optimal_threshold'] = optimal_threshold
                        
                        logging.info(f"Optimal threshold (Youden's J): {optimal_threshold_j:.2f}, F1-optimal: {optimal_threshold_f1:.2f}")
                    else:
                        logging.warning("No valid F1 scores found for threshold optimization")
                        metrics['optimal_threshold'] = 0.5  # Default
                        optimal_threshold = 0.5
                else:
                    logging.warning("Not enough unique values for threshold optimization")
                    metrics['optimal_threshold'] = 0.5  # Default
                    optimal_threshold = 0.5

            except Exception as threshold_error:
                logging.error(f"Error in threshold optimization: {str(threshold_error)}")
                optimal_threshold = 0.5
                metrics['optimal_threshold'] = 0.5  # Default
            
            # Update curve_data with calculated values - POSITION FIXED HERE
            curve_data = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': metrics.get('roc_auc', 0),
                'precision': precision_curve,
                'recall': recall_curve,
                'pr_auc': pr_auc
            }
            
            # Store threshold data in metrics - POSITION FIXED HERE
            metrics['threshold_data'] = {
                'thresholds': pr_thresholds,
                'f1_scores': f1_scores,
                'youden_j': youden_j,
                'roc_thresholds': roc_thresholds
            }
            
            # Create figure with 3x2 subplots for visualization
            fig = plt.figure(figsize=(16, 20))
            gs = gridspec.GridSpec(3, 2, figure=fig)
            
            # 1. Confusion Matrix
            try:
                ax1 = fig.add_subplot(gs[0, 0])
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No CKD', 'CKD'],
                        yticklabels=['No CKD', 'CKD'], ax=ax1)
                ax1.set_title('Confusion Matrix')
                ax1.set_ylabel('True Label')
                ax1.set_xlabel('Predicted Label')
                
                # Add text annotations for derived metrics
                specificity_val = metrics.get('specificity', 0)
                recall_val = metrics.get('recall', 0)
                gmean_val = metrics.get('gmean', 0)
                
                ax1.text(0.05, -0.15, 
                        f"Sensitivity: {recall_val:.3f}, Specificity: {specificity_val:.3f}, G-mean: {gmean_val:.3f}",
                        transform=ax1.transAxes, fontsize=10)
            except Exception as cm_error:
                logging.error(f"Error plotting confusion matrix: {str(cm_error)}")
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.text(0.5, 0.5, "Confusion Matrix plot failed", ha='center', va='center')
            
            # 2. ROC Curve
            try:
                ax2 = fig.add_subplot(gs[0, 1])
                if len(np.unique(y_test)) > 1 and len(np.unique(y_proba)) > 1:
                    # We've already calculated fpr, tpr in threshold optimization
                    roc_auc = metrics.get('roc_auc', 0)
                    ax2.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
                    
                    # Plot optimal threshold point on ROC curve
                    if 'optimal_threshold_youden' in metrics:
                        # Find the closest index in thresholds to our optimal threshold
                        idx = np.abs(roc_thresholds - metrics['optimal_threshold_youden']).argmin()
                        ax2.plot(fpr[idx], tpr[idx], 'ro', markersize=8, 
                                label=f"Optimal threshold: {metrics['optimal_threshold_youden']:.2f}")
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('Receiver Operating Characteristic')
                ax2.legend(loc="lower right")
            except Exception as roc_error:
                logging.error(f"Error plotting ROC curve: {str(roc_error)}")
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.text(0.5, 0.5, "ROC curve plot failed", ha='center', va='center')
            
            # 3. Precision-Recall Curve
            try:
                ax3 = fig.add_subplot(gs[1, 0])
                if len(precision_curve) > 1 and len(recall_curve) > 1:
                    # Calculate AUC only if we have valid curves
                    try:
                        ax3.plot(recall_curve, precision_curve, label=f'PR curve (area = {pr_auc:.3f})')
                        
                        # Plot optimal F1 threshold point on PR curve
                        if 'optimal_threshold_f1' in metrics:
                            # Find index in pr_thresholds closest to our F1-optimal threshold
                            if len(pr_thresholds) > 1:
                                idx = np.abs(pr_thresholds - metrics['optimal_threshold_f1']).argmin()
                                if idx < len(precision_curve) - 1:  # Ensure index is within bounds
                                    ax3.plot(recall_curve[idx], precision_curve[idx], 'ro', markersize=8,
                                            label=f"F1-optimal threshold: {metrics['optimal_threshold_f1']:.2f}")
                    except:
                        ax3.plot(recall_curve, precision_curve, label='PR curve')
                ax3.set_xlabel('Recall')
                ax3.set_ylabel('Precision')
                ax3.set_title('Precision-Recall Curve')
                ax3.legend(loc="lower left")
            except Exception as pr_error:
                logging.error(f"Error plotting Precision-Recall curve: {str(pr_error)}")
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.text(0.5, 0.5, "Precision-Recall curve plot failed", ha='center', va='center')
            
            # 4. Feature Importance (if available)
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Try to get feature importance based on model type
            try:
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[-min(15, len(importances)):]  # Top 15 features or fewer
                    
                    if feature_names is not None and len(feature_names) == len(importances):
                        features = [feature_names[i] for i in indices]
                    else:
                        features = [f"Feature {i}" for i in indices]
                        
                    ax4.barh(range(len(indices)), importances[indices], align='center')
                    ax4.set_yticks(range(len(indices)))
                    ax4.set_yticklabels(features)
                    
                elif hasattr(model, 'coef_'):
                    # For linear models
                    coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    
                    if feature_names is not None and len(feature_names) == len(coefs):
                        features = feature_names
                    else:
                        features = [f"Feature {i}" for i in range(len(coefs))]
                        
                    indices = np.argsort(np.abs(coefs))[-min(15, len(coefs)):]  # Top 15 features by magnitude
                    
                    ax4.barh(range(len(indices)), coefs[indices], align='center')
                    ax4.set_yticks(range(len(indices)))
                    ax4.set_yticklabels([features[i] for i in indices])
                    
                elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    # For pipeline models with classifier step
                    clf = model.named_steps['classifier']
                    if hasattr(clf, 'feature_importances_'):
                        importances = clf.feature_importances_
                        indices = np.argsort(importances)[-min(15, len(importances)):]
                        
                        if feature_names is not None and len(feature_names) == len(importances):
                            features = [feature_names[i] for i in indices]
                        else:
                            features = [f"Feature {i}" for i in indices]
                            
                        ax4.barh(range(len(indices)), importances[indices], align='center')
                        ax4.set_yticks(range(len(indices)))
                        ax4.set_yticklabels(features)
                    elif hasattr(clf, 'coef_'):
                        coefs = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
                        
                        if feature_names is not None and len(feature_names) == len(coefs):
                            features = feature_names
                        else:
                            features = [f"Feature {i}" for i in range(len(coefs))]
                            
                        indices = np.argsort(np.abs(coefs))[-min(15, len(coefs)):]
                        
                        ax4.barh(range(len(indices)), coefs[indices], align='center')
                        ax4.set_yticks(range(len(indices)))
                        ax4.set_yticklabels([features[i] for i in indices])
                    else:
                        # If no direct feature importance, try permutation importance
                        raise AttributeError("No direct feature importance found in pipeline classifier")
                else:
                    # If no direct feature importance, use permutation importance
                    if hasattr(model, 'estimators_'):
                        # For ensemble models like VotingClassifier
                        ax4.text(0.5, 0.5, "Feature importance not available\nfor this ensemble model", 
                            ha='center', va='center', transform=ax4.transAxes)
                    else:
                        # Calculate permutation importance if X_test has enough samples
                        if len(X_test) >= 10:  # Only calculate if we have enough samples
                            # Convert X_test to numpy if it's a DataFrame
                            X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
                            
                            perm_importance = permutation_importance(model, X_test_arr, y_test, 
                                                            n_repeats=10,  # Increased for more stability
                                                            random_state=self.random_state)
                            indices = np.argsort(perm_importance.importances_mean)[-min(15, X_test_arr.shape[1]):]
                            
                            if feature_names is not None and len(feature_names) == X_test_arr.shape[1]:
                                features = [feature_names[i] for i in indices]
                            else:
                                features = [f"Feature {i}" for i in indices]
                                
                            ax4.barh(range(len(indices)), perm_importance.importances_mean[indices], align='center',
                                xerr=perm_importance.importances_std[indices])  # Add error bars
                            ax4.set_yticks(range(len(indices)))
                            ax4.set_yticklabels(features)
                        else:
                            ax4.text(0.5, 0.5, "Not enough samples for\npermutation importance", 
                                ha='center', va='center', transform=ax4.transAxes)
            except Exception as fi_error:
                logging.error(f"Error calculating feature importance: {str(fi_error)}")
                ax4.text(0.5, 0.5, "Feature importance calculation failed", 
                    ha='center', va='center', transform=ax4.transAxes)
            
            ax4.set_title('Feature Importance')
            ax4.set_xlabel('Importance')
            
            # 5. Calibration Curve
            try:
                ax5 = fig.add_subplot(gs[2, 0])
                if len(np.unique(y_test)) > 1 and len(np.unique(y_proba)) > 1:
                    # Use more bins for larger test sets
                    n_bins = min(20, max(5, len(y_test)//20 + 2))
                    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
                    ax5.plot(prob_pred, prob_true, marker='o', label="Model calibration")
                    
                    # Calculate and display Brier score
                    brier = brier_score_loss(y_test, y_proba)
                    ax5.text(0.05, 0.95, f"Brier score: {brier:.4f}", transform=ax5.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
                ax5.plot([0, 1], [0, 1], linestyle='--', label="Perfect calibration")
                ax5.set_title('Calibration Curve')
                ax5.set_xlabel('Predicted Probability')
                ax5.set_ylabel('True Probability')
                ax5.legend(loc="lower right")
                
                # Add histogram of prediction distribution at the bottom
                if len(np.unique(y_proba)) > 1:
                    ax5_2 = ax5.twinx()
                    ax5_2.hist(y_proba, bins=n_bins, alpha=0.3, color='gray', range=(0, 1))
                    ax5_2.set_ylabel('Count')
                    ax5_2.set_yticks([])  # Hide y-ticks for cleaner look
            except Exception as calib_error:
                logging.error(f"Error plotting calibration curve: {str(calib_error)}")
                ax5 = fig.add_subplot(gs[2, 0])
                ax5.text(0.5, 0.5, "Calibration curve plot failed", ha='center', va='center')
            
            # 6. Threshold Optimization Plots
            try:
                ax6 = fig.add_subplot(gs[2, 1])
                
                # Plot both metrics for threshold optimization: F1 score and Youden's J
                if len(pr_thresholds) > 1 and len(f1_scores) > 1:
                    # Plot F1 scores vs threshold
                    ax6.plot(pr_thresholds, f1_scores, 'b-', label='F1 Score')
                    if 'optimal_threshold_f1' in metrics:
                        ax6.axvline(x=metrics['optimal_threshold_f1'], color='b', linestyle='--', 
                                label=f'F1-optimal = {metrics["optimal_threshold_f1"]:.2f}')
                
                # Add Youden's J to the same plot with a twin axis if available
                if len(roc_thresholds) > 1 and len(youden_j) > 1:
                    ax6_2 = ax6.twinx()
                    ax6_2.plot(roc_thresholds, youden_j, 'r-', label="Youden's J")
                    ax6_2.set_ylabel("Youden's J (Sensitivity + Specificity - 1)")
                    
                    if 'optimal_threshold_youden' in metrics:
                        ax6_2.axvline(x=metrics['optimal_threshold_youden'], color='r', linestyle=':', 
                                label=f'J-optimal = {metrics["optimal_threshold_youden"]:.2f}')
                    
                    # Add legend for the twin axis
                    lines1, labels1 = ax6.get_legend_handles_labels()
                    lines2, labels2 = ax6_2.get_legend_handles_labels()
                    ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')
                else:
                    ax6.legend(loc='best')
                    
                ax6.set_title('Threshold Optimization')
                ax6.set_xlabel('Threshold')
                ax6.set_ylabel('F1 Score')
                
                # Set appropriate x-axis limits
                ax6.set_xlim([0, 1])
                
            except Exception as threshold_plot_error:
                logging.error(f"Error plotting threshold optimization: {str(threshold_plot_error)}")
                ax6 = fig.add_subplot(gs[2, 1])
                ax6.text(0.5, 0.5, "Threshold optimization plot failed", ha='center', va='center')
            
            # Save the figure with error handling
            try:
                plt.tight_layout()
                # Ensure the plot directory exists
                os.makedirs(self.plots_dir, exist_ok=True)
                
                # Save high-quality image with better DPI
                filename = self.plots_dir / f"{model_name.replace(' ', '_').lower()}_evaluation.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                
                # Also save PDF for vector graphics if needed
                pdf_filename = self.plots_dir / f"{model_name.replace(' ', '_').lower()}_evaluation.pdf"
                plt.savefig(pdf_filename, bbox_inches='tight')
                
                logging.info(f"Evaluation plots saved to {filename} and {pdf_filename}")
            except Exception as save_error:
                logging.error(f"Error saving plot: {str(save_error)}")
            finally:
                plt.close()
            
            # Add additional overall model quality metrics
            try:
                # Add Matthews Correlation Coefficient
                metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
                
                # Add balanced accuracy
                metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
                
                # Calculate log loss if possible
                if len(np.unique(y_test)) > 1 and len(np.unique(y_proba)) > 1:
                    y_proba_clipped = np.clip(y_proba, 1e-15, 1 - 1e-15)
                    metrics['log_loss'] = log_loss(y_test, y_proba_clipped)
                    
                # Add Brier score
                metrics['brier_score'] = brier_score_loss(y_test, y_proba)
            except Exception as add_metrics_error:
                logging.error(f"Error calculating additional metrics: {str(add_metrics_error)}")

            return metrics, class_report, curve_data
            
        except Exception as e:
            logging.error(f"Evaluation failed for {model_name}: {str(e)}")
            traceback.print_exc()  # Print stack trace for debugging
            default_curve_data = {
                'fpr': np.array([0, 1]),
                'tpr': np.array([0, 1]),
                'roc_auc': 0,
                'precision': np.array([0, 1]),
                'recall': np.array([1, 0]),                    
                'pr_auc': 0
            }
            
            return {'evaluation_error': str(e)}, {}, default_curve_data

    def visualize_data(self, X, y, feature_names=None):
        """Visualize data distribution and relationships with added SHAP explanations"""
        try:
            # Use feature names if provided
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            
            # Convert X to dataframe if it's not already
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
            else:
                X_df = X.copy()  # Use copy to avoid modifying original dataframe
            
            # Add target to dataframe for visualization
            X_df['target'] = y
            
            # Create plots directory if it doesn't exist
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # 1. Pairplot for key features (limit to top 5 to avoid clutter)
            if X_df.shape[1] > 2:  # Only if we have multiple features
                # Limit to 5 features plus target to avoid huge plots
                cols_to_plot = list(X_df.columns[:min(5, len(X_df.columns)-1)]) + ['target']
                plt.figure(figsize=(12, 10))
                pairplot = sns.pairplot(X_df[cols_to_plot], hue='target', diag_kind='kde')
                pairplot.fig.suptitle('Pairplot of Key Features', y=1.02)
                plt.savefig(self.plots_dir / "pairplot.png", bbox_inches='tight')
                plt.close()
            
            # 2. Correlation heatmap
            plt.figure(figsize=(12, 10))
            # Remove target column for correlation calculation to focus on feature relationships
            corr_df = X_df.drop(columns=['target']) if 'target' in X_df.columns else X_df.copy()
            corr = corr_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "correlation_heatmap.png", bbox_inches='tight')
            plt.close()
            
            # 3. Dimensionality reduction for visualization
            if X.shape[1] > 2:
                # PCA
                pca = PCA(n_components=2, random_state=self.random_state)
                X_pca = pca.fit_transform(X_df.drop(columns=['target']))
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                                    alpha=0.8, edgecolors='w')
                plt.colorbar(scatter)
                plt.title('PCA: 2-Component Visualization')
                plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2f})')
                plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2f})')
                plt.savefig(self.plots_dir / "pca_visualization.png", bbox_inches='tight')
                plt.close()
                
                # t-SNE (with caution for larger datasets)
                if X.shape[0] < 5000:  # Only for smaller datasets to avoid long computation
                    tsne = TSNE(n_components=2, random_state=self.random_state, n_jobs=-1)  # Use all available cores
                    X_tsne = tsne.fit_transform(X_df.drop(columns=['target']))
                    
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                                        alpha=0.8, edgecolors='w')
                    plt.colorbar(scatter)
                    plt.title('t-SNE: 2-Component Visualization')
                    plt.savefig(self.plots_dir / "tsne_visualization.png", bbox_inches='tight')
                    plt.close()
                
            logging.info("Data visualization complete. Plots saved to plots directory.")
            
        except Exception as e:
            logging.error(f"Data visualization failed: {str(e)}")
            traceback.print_exc()  # Print the full traceback for debugging

    def generate_shap_explanations(self, model, X, feature_names=None):
        """Generate SHAP explanations for the model predictions"""
        try:
            if not isinstance(X_sample, pd.DataFrame):
                X_sample = pd.DataFrame(X_sample, columns=feature_names)

            # Use feature names if provided
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            
            # Convert X to pandas DataFrame if it's not already
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
            else:
                X_df = X.copy()  # Use copy to avoid modifying original
            
            # Sample data if it's too large
            if X_df.shape[0] > 1000:
                X_sample = X_df.sample(1000, random_state=self.random_state)
            else:
                X_sample = X_df
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_sample)
            else:
                explainer = shap.Explainer(model.predict, X_sample)

            # Calculate SHAP values
            shap_values = explainer(X_sample)
            
            # Create plots directory if it doesn't exist
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, show=False)
            plt.title('SHAP Feature Importance Summary')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "shap_summary.png", bbox_inches='tight')
            plt.close()
            
            # Bar plot for feature importance
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, plot_type='bar', show=False)
            plt.title('SHAP Feature Importance (Bar Plot)')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "shap_importance_bar.png", bbox_inches='tight')
            plt.close()
            
            # Dependence plots for top 3 features
            if shap_values.values.ndim > 2:  # For multi-class classification
                shap_values_sum = np.abs(shap_values.values).mean(axis=2)
                feature_importance = shap_values_sum.mean(axis=0)
            else:  # For regression or binary classification
                feature_importance = np.abs(shap_values.values).mean(axis=0)
            
            top_indices = np.argsort(feature_importance)[-3:]
            
            for idx in top_indices:
                feature_name = X_sample.columns[idx]
                plt.figure(figsize=(12, 8))
                shap.dependence_plot(idx, shap_values.values, X_sample, feature_names=X_sample.columns, show=False)
                plt.title(f'SHAP Dependence Plot for {feature_name}')
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"shap_dependence_{feature_name}.png", bbox_inches='tight')
                plt.close()
                
            logging.info("SHAP explanations complete. Plots saved to plots directory.")
            
            return shap_values
            
        except Exception as e:
            logging.error(f"SHAP explanation generation failed: {str(e)}")
            traceback.print_exc()  # Print the full traceback for debugging
            return None

    def _clean_params(self, params):
        """Remove pipeline prefix from parameters"""
        return {k.replace('classifier__', ''): v for k, v in params.items()}
    
    def check_for_data_leakage(self, X_train, X_test, y_train, y_test):
        """Check for potential data leakage between train and test sets"""
        try:
            # 1. Check for duplicate entries between train and test
            if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
                # Convert to string to make hashing faster
                train_hashes = set([hash(tuple(row)) for _, row in X_train.astype(str).iterrows()])
                test_hashes = set([hash(tuple(row)) for _, row in X_test.astype(str).iterrows()])
                
                # Find overlap
                overlap = train_hashes.intersection(test_hashes)
                
                if overlap:
                    logging.warning(f"Potential data leakage: {len(overlap)} identical rows found in both train and test sets")
                    return False
                else:
                    logging.info("No duplicate entries found between train and test sets")
            else:
                # If not DataFrames, convert to arrays and check
                X_train_array = X_train.toarray() if hasattr(X_train, 'toarray') else np.array(X_train)
                X_test_array = X_test.toarray() if hasattr(X_test, 'toarray') else np.array(X_test)
                
                # Check if any test row exactly matches a training row
                for test_row in X_test_array:
                    if any(np.array_equal(test_row, train_row) for train_row in X_train_array):
                        logging.warning("Potential data leakage: identical rows found in both train and test sets")
                        return False
                
                logging.info("No duplicate entries found between train and test sets")
            
            # 2. Check for feature correlation with target
            if isinstance(X_train, pd.DataFrame):
                X_train_with_target = X_train.copy()
                X_train_with_target['target'] = y_train
                
                correlations = X_train_with_target.corr()['target'].drop('target')
                high_corr_features = correlations[abs(correlations) > 0.95].index.tolist()
                
                if high_corr_features:
                    logging.warning(f"Potential data leakage: Features with very high correlation to target: {high_corr_features}")
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Data leakage check failed: {str(e)}")
            return True  # Default to True to avoid disrupting the workflow
    
    def train_models(self, X, y, use_pipeline=True, search_type='random'):
        """Train and optimize multiple models, return the best one"""
        try:
            logging.info("Starting model training")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Reset indices to ensure proper alignment after splitting
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
            if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
            
            logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Check for data leakage - simplified handling
            leakage_found = self.check_for_data_leakage(X_train, X_test, y_train, y_test)
            if leakage_found:
                logging.warning("Data leakage detected! Consider reviewing your data splitting strategy")
            
            # Preserve feature names
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
                original_columns = X_train.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                original_columns = feature_names
            
            # Feature selection if configured (BEFORE any data augmentation like SMOTE)
            if self.use_feature_selection:
                try:
                    # Perform feature selection using training data
                    X_train, selected_features, selector = self.select_features(
                        X_train, 
                        y_train,
                        method='select_k_best',
                        feature_names=feature_names
                    )
                    
                    # If feature selection was successful and we have a selector
                    if selector is not None:
                        # Transform test set using the SAME selector
                        X_test_transformed = selector.transform(X_test)
                        
                        # Update feature names with selected ones
                        selected_indices = selector.get_support(indices=True)
                        feature_names = [original_columns[i] for i in selected_indices]
                        
                        # Convert to DataFrame to preserve feature names
                        if isinstance(X, pd.DataFrame):
                            X_train = pd.DataFrame(X_train, columns=feature_names, index=X_train.index if hasattr(X_train, 'index') else None)
                            X_test = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index if hasattr(X_test, 'index') else None)
                        else:
                            # If original wasn't DataFrame, convert arrays to DataFrame for consistency
                            X_train = pd.DataFrame(X_train, columns=feature_names)
                            X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
                    else:
                        logging.error("Feature selection failed, no selector was returned")
                        return {}, None
                except Exception as e:
                    logging.error(f"Feature selection failed: {str(e)}")
                    return {}, None
            
            # Ensure we have DataFrame for consistent processing
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=feature_names)
                X_test = pd.DataFrame(X_test, columns=feature_names)
            
            # Handle class imbalance AFTER feature selection but BEFORE scaling
            # Only apply to training data to prevent data leakage
            if self.use_smote and not use_pipeline:
                try:
                    # Apply SMOTE while preserving DataFrame structure
                    X_train_values, y_train = self.handle_imbalance(X_train.values, y_train, method='smote')
                    # Convert back to DataFrame with original feature names
                    X_train = pd.DataFrame(X_train_values, columns=X_train.columns)
                except Exception as e:
                    logging.error(f"SMOTE application failed: {str(e)}")
                    logging.warning("Continuing without applying SMOTE")
            elif not self.use_smote and 'XGBoost' in self.models:

                # Calculate class weight for imbalanced data
                if isinstance(y_train, (pd.Series, pd.DataFrame)):
                    y_train_values = y_train.values
                else:
                    y_train_values = y_train
                
                pos_count = sum(y_train_values)
                neg_count = len(y_train_values) - pos_count
                scale_pos = neg_count / pos_count if pos_count > 0 else 1
                self.models['XGBoost'].set_params(scale_pos_weight=scale_pos)
                logging.info(f"Set XGBoost scale_pos_weight to {scale_pos:.2f}")
            
            # Scale features if configured (AFTER feature selection and SMOTE)
            if self.scaler is not None and not use_pipeline:
                try:
                    # Apply scaling while preserving DataFrame structure
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    # Convert back to DataFrame with feature names preserved
                    X_train = pd.DataFrame(
                        X_train_scaled, 
                        columns=X_train.columns, 
                        index=X_train.index
                    )
                    X_test = pd.DataFrame(
                        X_test_scaled, 
                        columns=X_test.columns, 
                        index=X_test.index
                    )
                except Exception as e:
                    logging.error(f"Scaling failed: {str(e)}")
                    # Try to continue without scaling
                    logging.warning("Continuing without scaling")
            
            # Visualize preprocessed data
            try:
                self.visualize_data(X_train, y_train, feature_names)
            except Exception as e:
                logging.warning(f"Data visualization failed: {str(e)}")
            
            # Train all models and keep track of results
            model_results = {}
            best_score = 0
            best_model_name = None
            
            for name, base_model in self.models.items():
                logging.info(f"Training {name}...")
                start_time = time.time()
                
                try:
                    if use_pipeline:
                        # Create a pipeline with proper preprocessing steps
                        steps = []
                        
                        # Add scaler if configured
                        if self.scaler is not None:
                            steps.append(('scaler', clone(self.scaler)))
                        
                        # Add SMOTE if configured
                        if self.use_smote:
                            steps.append(('smote', SMOTE(random_state=self.random_state)))
                        
                        # Add the classifier
                        steps.append(('classifier', clone(base_model)))
                        
                        # Use Pipeline instead of imPipeline if imPipeline doesn't exist
                        try:
                            pipe = imPipeline(steps)
                        except NameError:
                            from sklearn.pipeline import Pipeline
                            pipe = Pipeline(steps)
                        
                        # Hyperparameter tuning with nested CV
                        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                        
                        # Set scoring to include both f1 and geometric mean
                        from imblearn.metrics import geometric_mean_score
                        scoring = {
                            'f1': 'f1_weighted',
                            'gmean': make_scorer(geometric_mean_score, average='weighted')
                        }
                        refit = 'gmean'
                        
                        # Check if param grids/distributions exist for this model
                        param_exists = False
                        if search_type == 'grid' and name in self.param_grids:
                            param_exists = True
                            search = GridSearchCV(
                                pipe,
                                param_grid=self.param_grids[name],
                                cv=inner_cv,
                                scoring=scoring,
                                refit=refit,
                                n_jobs=-1,
                                verbose=1
                            )
                        elif search_type == 'random' and name in self.param_distributions:
                            param_exists = True
                            search = RandomizedSearchCV(
                                pipe,
                                param_distributions=self.param_distributions[name],
                                n_iter=50,
                                cv=inner_cv,
                                scoring=scoring,
                                refit=refit,
                                n_jobs=-1,
                                random_state=self.random_state,
                                verbose=1
                            )
                        
                        if not param_exists:
                            logging.warning(f"No parameters found for {name} with search_type={search_type}, skipping tuning")
                            model = pipe
                            # Ensure DataFrame structure is maintained during fit
                            if isinstance(X_train, pd.DataFrame):
                                model.fit(X_train, y_train)
                            else:
                                model.fit(pd.DataFrame(X_train, columns=feature_names), y_train)
                            nested_scores = None
                        else:
                            # Outer CV loop for nested CV evaluation
                            nested_scores = {
                                'f1': [],
                                'gmean': []
                            }
                            
                            # Ensure X_train is DataFrame for consistent processing
                            X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=feature_names)
                            
                            for train_idx, test_idx in outer_cv.split(X_train_df, y_train):
                                X_tr = X_train_df.iloc[train_idx]
                                X_val = X_train_df.iloc[test_idx]
                                y_tr, y_val = y_train[train_idx], y_train[test_idx]
                                
                                search.fit(X_tr, y_tr)
                                best_model = search.best_estimator_
                                y_pred = best_model.predict(X_val)
                                f1 = f1_score(y_val, y_pred, average='weighted')
                                gmean = geometric_mean_score(y_val, y_pred, average='weighted')
                                nested_scores['f1'].append(f1)
                                nested_scores['gmean'].append(gmean)
                            
                            logging.info(f"Nested CV F1 scores: {nested_scores['f1']}")
                            logging.info(f"Nested CV G-mean scores: {nested_scores['gmean']}")
                            if nested_scores['f1']:
                                logging.info(f"Mean nested CV F1 score: {np.mean(nested_scores['f1']):.4f} (±{np.std(nested_scores['f1']):.4f})")
                                logging.info(f"Mean nested CV G-mean score: {np.mean(nested_scores['gmean']):.4f} (±{np.std(nested_scores['gmean']):.4f})")
                            
                            # Final fit with the entire training data
                            search.fit(X_train_df, y_train)
                            model = search.best_estimator_
                            
                            # Clean and log best parameters
                            if hasattr(self, '_clean_params'):
                                best_params = self._clean_params(search.best_params_)
                            else:
                                best_params = search.best_params_
                            logging.info(f"{name} best params: {best_params}")
                    else:
                        # Train without a pipeline - ensure DataFrame consistency
                        model = clone(base_model)
                        # Fit with DataFrame to maintain feature names
                        if isinstance(X_train, pd.DataFrame):
                            model.fit(X_train, y_train)
                        else:
                            X_train_df = pd.DataFrame(X_train, columns=feature_names)
                            model.fit(X_train_df, y_train)
                        nested_scores = None
                    
                    # Evaluate the model - ensure DataFrame consistency
                    X_test_eval = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
                    metrics, report, curve_data = self.evaluate_model(
                        model, X_test_eval, y_test, feature_names, name
                    )
                    
                    # Add geometric mean to metrics if not already present
                    if 'gmean' not in metrics:
                        try:
                            y_pred = model.predict(X_test_eval)
                            metrics['gmean'] = geometric_mean_score(y_test, y_pred, average='weighted')
                        except Exception as e:
                            logging.warning(f"Could not calculate G-mean: {str(e)}")
                            metrics['gmean'] = 0.0
                    
                    training_time = time.time() - start_time

                    logging.info(f"{name} training completed in {training_time:.2f}s")
                    logging.info(f"{name} F1 score: {metrics.get('f1', 0):.4f}")
                    logging.info(f"{name} G-mean score: {metrics.get('gmean', 0):.4f}")
                    
                    # Track the best model (using G-mean instead of F1)
                    if metrics.get('gmean', 0) > best_score:
                        best_score = metrics.get('gmean', 0)
                        best_model_name = name
                    
                    model_results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'report': report,
                        'curve_data': curve_data,
                        'training_time': training_time,
                        'nested_scores': nested_scores
                    }

                except Exception as e:
                    logging.error(f"Error training {name}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Create ensemble of top models if we have multiple successful models
            if len(model_results) >= 2:
                try:
                    logging.info("Training ensemble models...")
                    
                    # Take top 3 models or fewer if we don't have 3
                    top_models = sorted(
                        model_results.items(), 
                        key=lambda x: x[1]['metrics'].get('gmean', 0),
                        reverse=True
                    )[:min(3, len(model_results))]
                    
                    estimators = [(name, model_dict['model']) for name, model_dict in top_models]
                    
                    # Ensure X_train is DataFrame for ensemble training
                    X_train_ensemble = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=feature_names)
                    X_test_ensemble = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
                    
                    # Create a voting ensemble
                    ensemble = VotingClassifier(
                        estimators=estimators,
                        voting='soft',
                        n_jobs=-1
                    )
                    
                    ensemble.fit(X_train_ensemble, y_train)
                    
                    # Evaluate voting ensemble
                    metrics, report, curve_data = self.evaluate_model(
                        ensemble, X_test_ensemble, y_test, feature_names, "VotingEnsemble"
                    )
                    
                    # Add geometric mean to metrics if not already present
                    if 'gmean' not in metrics:
                        try:
                            y_pred = ensemble.predict(X_test_ensemble)
                            metrics['gmean'] = geometric_mean_score(y_test, y_pred, average='weighted')
                        except Exception as e:
                            logging.warning(f"Could not calculate G-mean for VotingEnsemble: {str(e)}")
                            metrics['gmean'] = 0.0
                    
                    model_results["VotingEnsemble"] = {
                        'model': ensemble,
                        'metrics': metrics,
                        'report': report,
                        'curve_data': curve_data,
                        'training_time': sum(model_dict['training_time'] for _, model_dict in top_models),
                        'nested_scores': None
                    }
                    
                    logging.info(f"Voting Ensemble G-mean score: {metrics.get('gmean', 0):.4f}")
                    
                    # Check if voting ensemble is better than individual models
                    if metrics.get('gmean', 0) > best_score:
                        best_score = metrics.get('gmean', 0)
                        best_model_name = "VotingEnsemble"
                    
                    # Define base models
                    base_models = [
                        (name, model_dict['model']) for name, model_dict in top_models
                    ]
                    
                    # Define meta-model - using a more robust model with balanced weights
                    meta_model = LogisticRegression(
                        class_weight='balanced', 
                        max_iter=1000,
                        C=1.0,
                        solver='liblinear'
                    )
                    
                    # Create stacking ensemble
                    stacking = StackingClassifier(
                        estimators=base_models,
                        final_estimator=meta_model,
                        cv=5,
                        n_jobs=-1,
                        passthrough=False
                    )
                    
                    logging.info("Training stacking ensemble...")
                    stacking_start_time = time.time()
                    stacking.fit(X_train_ensemble, y_train)
                    stacking_training_time = time.time() - stacking_start_time
                    
                    # Evaluate stacking ensemble
                    stacking_metrics, stacking_report, stacking_curve_data = self.evaluate_model(
                        stacking, X_test_ensemble, y_test, feature_names, "StackingEnsemble"
                    )
                    
                    # Add geometric mean to metrics if not already present
                    if 'gmean' not in stacking_metrics:
                        try:
                            y_pred = stacking.predict(X_test_ensemble)
                            stacking_metrics['gmean'] = geometric_mean_score(y_test, y_pred, average='weighted')
                        except Exception as e:
                            logging.warning(f"Could not calculate G-mean for StackingEnsemble: {str(e)}")
                            stacking_metrics['gmean'] = 0.0
                    
                    model_results["StackingEnsemble"] = {
                        'model': stacking,
                        'metrics': stacking_metrics,
                        'report': stacking_report,
                        'curve_data': stacking_curve_data,
                        'training_time': stacking_training_time,
                        'nested_scores': None
                    }
                    
                    logging.info(f"Stacking Ensemble G-mean score: {stacking_metrics.get('gmean', 0):.4f}")
                    
                    # Check if stacking ensemble is better than current best
                    if stacking_metrics.get('gmean', 0) > best_score:
                        best_score = stacking_metrics.get('gmean', 0)
                        best_model_name = "StackingEnsemble"
                    
                    # Prioritize ensembles with a small margin
                    ensemble_margin = 0.02  # 2% margin
                    for ensemble_name in ['VotingEnsemble', 'StackingEnsemble']:
                        if ensemble_name in model_results:
                            ensemble_gmean = model_results[ensemble_name]['metrics'].get('gmean', 0)
                            if ensemble_gmean >= (best_score - ensemble_margin):
                                best_score = ensemble_gmean
                                best_model_name = ensemble_name
                        
                except Exception as e:
                    logging.error(f"Error training ensemble models: {str(e)}")
                    traceback.print_exc()
            
            self.plot_combined_curves(model_results)
            self.visualize_hyperparameter_performance(model_results)

            # Select the best model
            if best_model_name:
                logging.info(f"Best model: {best_model_name} with G-mean score: {best_score:.4f}")
                self.best_model = model_results[best_model_name]['model']
                
                # Ensure models directory exists
                if not hasattr(self, 'models_dir'):
                    logging.warning("models_dir not defined, creating default")
                    from pathlib import Path
                    self.models_dir = Path("models")
                    self.models_dir.mkdir(exist_ok=True)
                
                # Save the best model
                try:
                    self.save_model(self.best_model, best_model_name)
                except Exception as e:
                    logging.error(f"Failed to save model: {str(e)}")
                
                # Save feature names for later use
                try:
                    with open(self.models_dir / "feature_names.pkl", 'wb') as f:
                        pickle.dump(feature_names, f)
                except Exception as e:
                    logging.error(f"Failed to save feature names: {str(e)}")
                
                # Generate SHAP explanations for the best model (if not ensemble)
                if not best_model_name.endswith("Ensemble"):
                    try:
                        # Ensure DataFrame consistency for SHAP
                        X_test_shap = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
                        self.generate_shap_explanations(
                            self.best_model, X_test_shap, feature_names, best_model_name
                        )
                    except Exception as e:
                        logging.error(f"SHAP explanation generation failed: {str(e)}")
                
                return model_results, best_model_name
            else:
                logging.error("No model was successfully trained")
                return {}, None
                
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            traceback.print_exc()
        return {}, None
    
    def plot_combined_curves(self, model_results):
        """
        Enhanced comprehensive visualization combining ROC, PR, Threshold Optimization, and Calibration curves
        with improved styling and statistical insights
        """
        
        try:
            # Enhanced color palette and styling options
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            line_styles = ['-', '--', '-.', ':']
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
            
            # Create comprehensive figure layout (2x3 grid for better organization)
            fig = plt.figure(figsize=(24, 16))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            fig.suptitle('Comprehensive Model Performance Analysis', fontsize=20, fontweight='bold', y=0.96)
            
            # Track performance for summary
            roc_aucs = []
            pr_aucs = []
            
            # 1. Enhanced ROC Curves
            ax1 = fig.add_subplot(gs[0, 0])
            for i, (name, result) in enumerate(model_results.items()):
                if 'curve_data' in result and all(key in result['curve_data'] for key in ['fpr', 'tpr', 'roc_auc']):
                    fpr = result['curve_data']['fpr']
                    tpr = result['curve_data']['tpr']
                    roc_auc = result['curve_data']['roc_auc']
                    roc_aucs.append((name, roc_auc))
                    
                    # Enhanced plotting with multiple styles
                    ax1.plot(fpr, tpr, 
                            color=colors[i % len(colors)], 
                            linestyle=line_styles[i % len(line_styles)],
                            label=f"{name} (AUC = {roc_auc:.3f})", 
                            linewidth=2.5, alpha=0.8)
                    
                    # Mark optimal threshold point if available
                    if 'optimal_roc_point' in result['curve_data']:
                        opt_point = result['curve_data']['optimal_roc_point']
                        ax1.scatter(opt_point[0], opt_point[1], 
                                color=colors[i % len(colors)], 
                                s=80, marker=markers[i % len(markers)], 
                                edgecolors='black', linewidth=1, zorder=5)
            
            # Enhanced ROC styling
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Random Classifier')
            ax1.set_xlim([-0.02, 1.02])
            ax1.set_ylim([-0.02, 1.02])
            ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax1.set_title('ROC Curves with Optimal Points', fontsize=14, fontweight='bold')
            ax1.legend(loc="lower right", fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
            ax1.grid(True, linestyle='--', alpha=0.4)
            ax1.set_facecolor('#f8f9fa')
            
            # 2. Enhanced Precision-Recall Curves
            ax2 = fig.add_subplot(gs[0, 1])
            for i, (name, result) in enumerate(model_results.items()):
                if 'curve_data' in result and all(key in result['curve_data'] for key in ['precision', 'recall']):
                    precision = result['curve_data']['precision']
                    recall = result['curve_data']['recall']
                    pr_auc = result['curve_data'].get('pr_auc')
                    
                    if pr_auc is not None:
                        pr_aucs.append((name, pr_auc))
                        label = f"{name} (AUC = {pr_auc:.3f})"
                    else:
                        label = name
                        
                    ax2.plot(recall, precision, 
                            color=colors[i % len(colors)], 
                            linestyle=line_styles[i % len(line_styles)],
                            label=label, linewidth=2.5, alpha=0.8)
            
            # Enhanced PR styling
            ax2.set_xlim([-0.02, 1.02])
            ax2.set_ylim([-0.02, 1.02])
            ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
            ax2.legend(loc="lower left", fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
            ax2.grid(True, linestyle='--', alpha=0.4)
            ax2.set_facecolor('#f8f9fa')
            
            # 3. Combined Threshold Optimization
            ax3 = fig.add_subplot(gs[0, 2])
            has_threshold_data = False
            
            for i, (name, result) in enumerate(model_results.items()):
                # Check for threshold data in multiple possible locations
                threshold_data = None
                if 'curve_data' in result and 'roc_thresholds' in result['curve_data']:
                    # Calculate F1 and Youden's J from available data
                    if 'y_pred_proba' in result['curve_data'] and result['curve_data']['y_pred_proba'] is not None:
                        thresholds = np.linspace(0.1, 0.9, 30)
                        f1_scores = []
                        youden_j = []
                        
                        # This would need y_test from the original evaluation
                        # For now, we'll use the ROC thresholds as approximation
                        roc_thresholds = result['curve_data']['roc_thresholds']
                        fpr = result['curve_data']['fpr']
                        tpr = result['curve_data']['tpr']
                        
                        # Calculate Youden's J for each threshold
                        youden_scores = tpr - fpr
                        
                        # Sample subset of thresholds for visualization
                        sample_indices = np.linspace(0, len(roc_thresholds)-1, min(30, len(roc_thresholds)), dtype=int)
                        sampled_thresholds = roc_thresholds[sample_indices]
                        sampled_youden = youden_scores[sample_indices]
                        
                        # Plot Youden's J statistic
                        ax3.plot(sampled_thresholds, sampled_youden, 
                                color=colors[i % len(colors)],
                                linestyle=line_styles[i % len(line_styles)],
                                linewidth=2,
                                marker=markers[i % len(markers)],
                                markersize=4,
                                label=f"{name} Youden's J")
                        has_threshold_data = True
                
                # Check for existing threshold data structure
                elif 'metrics' in result and 'threshold_data' in result['metrics']:
                    threshold_data = result['metrics']['threshold_data']
                    thresholds = threshold_data['thresholds']
                    f1_scores = threshold_data.get('f1_scores', [])
                    youden_j = threshold_data.get('youden_j', [])
                    
                    if f1_scores.size > 0 and len(f1_scores) > 0:
                        ax3.plot(thresholds, f1_scores, 
                                color=colors[i % len(colors)],
                                linestyle=line_styles[i % len(line_styles)],
                                linewidth=2,
                                label=f'{name} F1')
                    
                    if youden_j:
                        ax3b = ax3.twinx()
                        ax3b.plot(thresholds, youden_j,
                                color=colors[i % len(colors)],
                                linestyle=line_styles[(i+1) % len(line_styles)],
                                linewidth=1.5,
                                alpha=0.7,
                                label=f'{name} Youden J')
                        ax3b.set_ylabel("Youden's J Statistic", fontsize=12, fontweight='bold')
                    
                    has_threshold_data = True
            
            if has_threshold_data:
                ax3.set_xlabel('Threshold', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
                ax3.set_title('Threshold Optimization', fontsize=14, fontweight='bold')
                ax3.legend(loc='upper right', fontsize=9)
                ax3.grid(True, alpha=0.4)
                ax3.set_facecolor('#f8f9fa')
            else:
                # Fallback: Show optimal thresholds as bar chart
                model_names = []
                roc_thresholds = []
                pr_thresholds = []
                
                for name, result in model_results.items():
                    if 'curve_data' in result:
                        model_names.append(name)
                        roc_thresholds.append(result['curve_data'].get('optimal_roc_threshold', 0.5))
                        pr_thresholds.append(result['curve_data'].get('optimal_pr_threshold', 0.5))
                
                if model_names:
                    x = np.arange(len(model_names))
                    width = 0.35
                    
                    ax3.bar(x - width/2, roc_thresholds, width, label='ROC Optimal', 
                        color='skyblue', alpha=0.8)
                    ax3.bar(x + width/2, pr_thresholds, width, label='PR Optimal', 
                        color='lightcoral', alpha=0.8)
                    
                    ax3.set_xlabel('Models', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Optimal Threshold', fontsize=12, fontweight='bold')
                    ax3.set_title('Optimal Thresholds Comparison', fontsize=14, fontweight='bold')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(model_names, rotation=45, ha='right')
                    ax3.legend()
                    ax3.grid(True, alpha=0.4)
                    ax3.set_facecolor('#f8f9fa')
            
            # 4. Combined Calibration Plot
            ax4 = fig.add_subplot(gs[1, 0])
            has_calibration = False
            
            for i, (name, result) in enumerate(model_results.items()):
                # Check for calibration data
                if 'metrics' in result and 'calibration' in result['metrics']:
                    prob_true, prob_pred = result['metrics']['calibration']
                    ax4.plot(prob_pred, prob_true, 
                            marker=markers[i % len(markers)],
                            color=colors[i % len(colors)],
                            linestyle=line_styles[i % len(line_styles)],
                            markersize=6,
                            linewidth=2,
                            label=name)
                    has_calibration = True
                
                # Alternative: Create calibration from probability predictions
                elif 'curve_data' in result and 'y_pred_proba' in result['curve_data']:
                    y_proba = result['curve_data']['y_pred_proba']
                    if y_proba is not None:
                        # This would need y_test for proper calibration
                        # For visualization purposes, we'll show the distribution
                        ax4.hist(y_proba, bins=10, alpha=0.3, 
                                color=colors[i % len(colors)], 
                                label=f'{name} Prob Distribution')
                        has_calibration = True
            
            if has_calibration and 'prob_true' in locals():
                ax4.plot([0, 1], [0, 1], 'k:', linewidth=2, label="Perfect Calibration")
                ax4.set_xlim([-0.05, 1.05])
                ax4.set_ylim([-0.05, 1.05])
                ax4.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
                ax4.set_ylabel('True Probability', fontsize=12, fontweight='bold')
                ax4.set_title('Calibration Curves', fontsize=14, fontweight='bold')
            else:
                ax4.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax4.set_title('Probability Distributions', fontsize=14, fontweight='bold')
            
            ax4.legend(loc="upper left", fontsize=9)
            ax4.grid(True, alpha=0.4)
            ax4.set_facecolor('#f8f9fa')
            
            # 5. Performance Comparison Radar Chart
            ax5 = fig.add_subplot(gs[1, 1], projection='polar')
            
            # Metrics for radar chart
            radar_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall', 'accuracy']
            angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, (name, result) in enumerate(model_results.items()):
                if 'metrics' in result:
                    metrics = result['metrics']
                    values = []
                    
                    for metric in radar_metrics:
                        if metric in ['roc_auc', 'pr_auc']:
                            # Get from curve_data
                            if 'curve_data' in result:
                                values.append(result['curve_data'].get(metric, 0))
                            else:
                                values.append(metrics.get(metric, 0))
                        else:
                            values.append(metrics.get(metric, 0))
                    
                    values += values[:1]  # Complete the circle
                    
                    ax5.plot(angles, values, 'o-', linewidth=2, 
                            color=colors[i % len(colors)], label=name)
                    ax5.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
            ax5.set_ylim(0, 1)
            ax5.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
            ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
            
            # 6. Summary Statistics Table
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis('tight')
            ax6.axis('off')
            
            # Create performance summary table
            table_data = []
            headers = ['Model', 'ROC AUC', 'PR AUC', 'F1 Score', 'Precision', 'Recall']
            
            for name, result in model_results.items():
                metrics = result.get('metrics', {})
                curve_data = result.get('curve_data', {})
                
                row = [
                    name,
                    f"{curve_data.get('roc_auc', 0):.3f}",
                    f"{curve_data.get('pr_auc', 0):.3f}",
                    f"{metrics.get('f1_score', 0):.3f}",
                    f"{metrics.get('precision', 0):.3f}",
                    f"{metrics.get('recall', 0):.3f}"
                ]
                table_data.append(row)
            
            # Sort by ROC AUC
            table_data.sort(key=lambda x: float(x[1]), reverse=True)
            
            table = ax6.table(cellText=table_data, colLabels=headers, 
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight best performers
            for i in range(1, len(table_data) + 1):
                if i == 1:  # Best performer
                    for j in range(len(headers)):
                        table[(i, j)].set_facecolor('#FFD700')  # Gold
            
            ax6.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
            
            # Final layout adjustments
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save in multiple formats with enhanced quality
            for fmt in ['png', 'pdf', 'svg']:
                plt.savefig(self.plots_dir / f"comprehensive_model_analysis.{fmt}", 
                        dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            plt.close()
            
            # Enhanced logging with comprehensive summary
            logging.info("=" * 60)
            logging.info("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
            logging.info("=" * 60)
            
            if roc_aucs:
                roc_aucs.sort(key=lambda x: x[1], reverse=True)
                logging.info("ROC AUC Rankings:")
                for i, (name, auc) in enumerate(roc_aucs, 1):
                    logging.info(f"  {i}. {name}: {auc:.3f}")
            
            if pr_aucs:
                pr_aucs.sort(key=lambda x: x[1], reverse=True)
                logging.info("\nPR AUC Rankings:")
                for i, (name, auc) in enumerate(pr_aucs, 1):
                    logging.info(f"  {i}. {name}: {auc:.3f}")
            
            # Overall best model
            if roc_aucs and pr_aucs:
                # Calculate combined score (average of normalized ROC and PR AUC)
                combined_scores = {}
                for name, _ in model_results.items():
                    roc_score = next((auc for n, auc in roc_aucs if n == name), 0)
                    pr_score = next((auc for n, auc in pr_aucs if n == name), 0)
                    combined_scores[name] = (roc_score + pr_score) / 2
                
                best_overall = max(combined_scores.items(), key=lambda x: x[1])
                logging.info(f"\n🏆 BEST OVERALL MODEL: {best_overall[0]} (Combined Score: {best_overall[1]:.3f})")
            
            logging.info("=" * 60)
            logging.info("✅ Comprehensive model analysis visualization completed!")
            logging.info(f"📁 Saved as: comprehensive_model_analysis.[png|pdf|svg]")
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"❌ Comprehensive visualization failed: {str(e)}")
            logging.error(f"📍 Traceback: {traceback.format_exc()}")
            
            # Fallback to basic combined curves
            try:
                logging.info("🔄 Attempting fallback to basic combined curves...")
                self._plot_basic_combined_curves(model_results)
            except Exception as fallback_error:
                logging.error(f"❌ Fallback also failed: {str(fallback_error)}")

    def _plot_basic_combined_curves(self, model_results):
        """Fallback function for basic combined ROC and PR curves"""
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Basic Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # ROC Curves
        for i, (name, result) in enumerate(model_results.items()):
            if 'curve_data' in result:
                curve_data = result['curve_data']
                if 'fpr' in curve_data and 'tpr' in curve_data:
                    ax1.plot(curve_data['fpr'], curve_data['tpr'], 
                            color=colors[i % len(colors)], 
                            label=f"{name} (AUC = {curve_data.get('roc_auc', 0):.3f})", 
                            linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR Curves
        for i, (name, result) in enumerate(model_results.items()):
            if 'curve_data' in result:
                curve_data = result['curve_data']
                if 'precision' in curve_data and 'recall' in curve_data:
                    ax2.plot(curve_data['recall'], curve_data['precision'], 
                            color=colors[i % len(colors)], 
                            label=f"{name} (AUC = {curve_data.get('pr_auc', 0):.3f})", 
                            linewidth=2)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "basic_combined_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("✅ Basic combined curves saved successfully")

    def visualize_hyperparameter_performance(self, model_results):
        """Enhanced visualization of hyperparameter tuning results with statistical insights"""
        
        try:
            # Prepare data for visualization
            model_names = []
            all_scores = []
            mean_scores = []
            std_scores = []
            
            for model_name, results in model_results.items():
                if results.get('nested_scores') and 'gmean' in results['nested_scores']:
                    scores = results['nested_scores']['gmean']
                    model_names.append(model_name)
                    all_scores.append(scores)
                    mean_scores.append(np.mean(scores))
                    std_scores.append(np.std(scores))
            
            if not model_names:
                logging.warning("No hyperparameter tuning results found to visualize")
                return
            
            # Enhanced plotting with better styling
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle('Hyperparameter Tuning Performance Analysis', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Color scheme
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            
            # Left plot: Scatter plot with error bars
            for i, (model_name, scores) in enumerate(zip(model_names, all_scores)):
                # Add jitter for better visibility
                x_jitter = np.random.normal(i, 0.05, len(scores))
                ax1.scatter(x_jitter, scores, color=colors[i], alpha=0.6, 
                        s=50, edgecolors='black', linewidth=0.5, label=model_name)
                
                # Add mean with error bars
                ax1.errorbar(i, mean_scores[i], yerr=std_scores[i], 
                            fmt='D', color='red', markersize=8, capsize=8, 
                            capthick=2, elinewidth=2, alpha=0.8)
            
            ax1.set_xticks(range(len(model_names)))
            ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
            ax1.set_ylabel('G-mean Score', fontsize=14, fontweight='bold')
            ax1.set_title('Cross-Validation Scores Distribution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_facecolor('#f8f9fa')
            
            # Add statistical annotations
            max_score_idx = np.argmax(mean_scores)
            ax1.annotate(f'Best: {model_names[max_score_idx]}\n{mean_scores[max_score_idx]:.3f}±{std_scores[max_score_idx]:.3f}',
                        xy=(max_score_idx, mean_scores[max_score_idx]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Right plot: Box plot for better statistical comparison
            box_data = [scores for scores in all_scores]
            bp = ax2.boxplot(box_data, patch_artist=True, notch=True, showmeans=True)
            ax2.set_xticks(np.arange(1, len(model_names)+1))
            ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)              
            
            # Customize box plot colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1.5)
            
            plt.setp(bp['means'], marker='D', markerfacecolor='red', 
                    markeredgecolor='black', markersize=6)
            
            ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
            ax2.set_ylabel('G-mean Score', fontsize=14, fontweight='bold')
            ax2.set_title('Statistical Distribution Comparison', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_facecolor('#f8f9fa')
            
            # Add performance summary text
            summary_text = "Performance Summary:\n"
            sorted_results = sorted(zip(model_names, mean_scores, std_scores), 
                                key=lambda x: x[1], reverse=True)
            
            for i, (name, mean_score, std_score) in enumerate(sorted_results[:3]):
                summary_text += f"{i+1}. {name}: {mean_score:.3f}±{std_score:.3f}\n"
            
            fig.text(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92, bottom=0.15)
            
            # Save in multiple formats
            for fmt in ['png', 'pdf', 'svg']:
                plt.savefig(self.plots_dir / f"hyperparameter_performance.{fmt}", 
                        dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            plt.close()
            
            # Log detailed results
            logging.info("Hyperparameter Performance Summary:")
            for name, mean_score, std_score in sorted_results:
                logging.info(f"  {name}: {mean_score:.3f} ± {std_score:.3f}")
                
            logging.info("Enhanced hyperparameter performance visualization completed")
            
        except Exception as e:
            logging.error(f"Enhanced hyperparameter visualization failed: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

    def save_model(self, model, model_name):
        """Save the model to disk"""
        try:
            model_path = self.models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logging.error(f"Model saving failed: {str(e)}")
            return False
    
    def load_model(self, model_name):
        """Load a model from disk"""
        try:
            model_path = self.models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            return None
    
    def generate_shap_explanations(self, model, X_test, feature_names, model_name):
        """Generate and save SHAP explanations for model interpretability"""
        try:
            logging.info(f"Generating SHAP explanations for {model_name}")
            
            # Sample X_test if it's too large
            if X_test.shape[0] > 100:
                indices = np.random.choice(X_test.shape[0], 100, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test
            
            # Use the modern shap.Explainer which automatically selects the appropriate explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Plot and save SHAP summary
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.tight_layout()
            
            # Ensure valid filename by replacing spaces with underscores
            safe_model_name = model_name.replace(' ', '_').lower()
            plt.savefig(self.plots_dir / f"{safe_model_name}_shap_summary.png")
            plt.close()
            
            # Save SHAP dependency plots for top features
            if len(shap_values.shape) > 1:
                # For multi-output models, use the mean absolute SHAP value
                if len(shap_values.shape) == 3:  # For multi-class models
                    shap_sum = np.abs(shap_values.values[:,:,1]).mean(axis=0)  # Focus on positive class
                else:
                    shap_sum = np.abs(shap_values.values).mean(axis=0)
                
                top_indices = np.argsort(shap_sum)[-5:]
                
                for idx in top_indices:
                    plt.figure(figsize=(10, 7))
                    feature_idx = idx
                    shap.dependence_plot(feature_idx, shap_values.values, X_sample, 
                                        feature_names=feature_names, show=False)
                    
                    plt.tight_layout()
                    feat_name = feature_names[idx].replace(" ", "_").lower()
                    plt.savefig(self.plots_dir / f"{safe_model_name}_shap_dependence_{feat_name}.png")
                    plt.close()
            
            logging.info(f"SHAP explanations generated and saved for {model_name}")
            
        except Exception as e:
            logging.error(f"SHAP explanation generation failed: {str(e)}")
            # Add traceback for better debugging
            import traceback
            logging.error(traceback.format_exc())
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            logging.error("No model has been trained yet")
            return None
        
        try:
            # Apply preprocessing steps consistent with training
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)[:, 1]
            
            return predictions, probabilities
        
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return None, None
    
    def run_pipeline(self):
        """Run the complete analysis pipeline"""
        try:
            logging.info("Starting CKD analysis pipeline")
            
            # Load and preprocess data
            X, y = self.load_data()
            
            # Train models
            model_results, best_model_name = self.train_models(X, y)
            
            if best_model_name:
                logging.info(f"Pipeline completed successfully. Best model: {best_model_name}")
                return True
            else:
                logging.error("Pipeline completed but no satisfactory model was found")
                return False
        
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Configure processor with desired settings
    processor = CKDPreprocessor(
        random_state=42,
        cv_folds=5,
        test_size=0.2,
        use_smote=True,
        use_feature_selection=True,
        imputation_strategy='knn',
        scaler_type='standard'
    )
    
    # Run the complete pipeline
    success = processor.run_pipeline()
    
    if success:
        logging.info("CKD analysis completed successfully")
        
        # Load the best model for deployment or further analysis
        best_model = processor.best_model