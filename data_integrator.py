import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ckd_data_integration.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings('ignore')

# Medical Constraints - Expanded for CKD specific features with broader ranges
MEDICAL_RANGES = {
    'age': (0, 120),
    'bp': (40, 250),
    'sg': (1.000, 1.060),
    'al': (0, 6),
    'su': (0, 6),
    'rbc': (0, 1),
    'pc': (0, 1),
    'pcc': (0, 1),
    'ba': (0, 1),
    'bgr': (30, 600),
    'bu': (2, 200),
    'sc': (0.1, 20.0),
    'sod': (110, 180),
    'pot': (2.0, 9.0),
    'hemo': (2.0, 25.0),
    'pcv': (15, 65),
    'wc': (1500, 25000),
    'rc': (1.5, 9.0),
    'htn': (0, 1),
    'dm': (0, 1),
    'cad': (0, 1),
    'appet': (0, 1),
    'pe': (0, 1),
    'ane': (0, 1),
    'bmi': (12, 60)
}

# Expanded feature mapping with more variations to catch different naming conventions
FEATURE_PRIORITY = {
    'target': ['ckd', 'diagnosis', 'class', 'target', 'result', 'condition', 'disease_state', 'classification', 'disease', 'chronic_kidney_disease', 'kidney_disease', 'status'],
    'age': ['age', 'patient_age', 'umur', 'years', 'age_years', 'ages', 'patient_years', 'usia'],
    'bp': ['bp', 'blood_pressure', 'bloodpress', 'tekanan_darah', 'systolic', 'systolic_bp', 'tension', 'sistole', 'bloodp'],
    'sg': ['sg', 'specific_gravity', 'berat_jenis', 'urine_sg', 'specific_g', 'spgravity', 'sp_gravity', 'urine_gravity'],
    'al': ['al', 'albumin', 'alb', 'urine_albumin', 'albuminuria', 'albumine', 'protein', 'urine_protein'],
    'su': ['su', 'sugar', 'urine_sugar', 'glucosuria', 'glucose_urine', 'urine_glucose', 'gula_urine'],
    'rbc': ['rbc', 'red_blood_cells', 'urine_rbc', 'blood_in_urine', 'eritrosit', 'redbloodcells', 'rbc_urine', 'red_cell'],
    'pc': ['pc', 'pus_cells', 'wbc_in_urine', 'pyuria', 'pus', 'puscell', 'wbc_urine', 'leukosit_urine'],
    'pcc': ['pcc', 'pus_cell_clumps', 'wbc_clumps', 'pus_clumps', 'cell_clumps', 'pusclump'],
    'ba': ['ba', 'bacteria', 'bacteriuria', 'bacterial', 'bakteri', 'bact', 'urine_bacteria'],
    'bgr': ['bgr', 'blood_glucose', 'glukosa', 'glucose', 'fasting_glucose', 'random_glucose', 'gula_darah', 'bloodglucose', 'gula', 'blood_sugar', 'sugar_blood'],
    'bu': ['bu', 'blood_urea', 'urea', 'bun', 'blood_urea_nitrogen', 'urea_blood', 'blood_urea_n', 'ureum', 'urea_nitrogen'],
    'sc': ['sc', 'serum_creatinine', 'kreatinin', 'creatinine', 'crea', 'creat', 'serumcreatinine', 'creatinin', 'scr', 'serum_crea'],
    'sod': ['sod', 'sodium', 'natrium', 'serum_sodium', 'na', 'na_serum', 'serum_na', 'sodium_level', 'sodium_blood'],
    'pot': ['pot', 'potassium', 'kalium', 'serum_potassium', 'k', 'k_serum', 'serum_k', 'potassium_level', 'potassium_blood'],
    'hemo': ['hemo', 'hemoglobin', 'hb', 'haemoglobin', 'hgb', 'hemoglob', 'hb_level', 'hb_blood', 'hemoglobin_level'],
    'pcv': ['pcv', 'packed_cell_volume', 'hematocrit', 'hct', 'hcrit', 'packed_cells', 'cell_volume'],
    'wc': ['wc', 'white_blood_cells', 'wbc', 'leukocytes', 'leukocyte_count', 'white_cells', 'white_count', 'leukosit'],
    'rc': ['rc', 'red_blood_cell_count', 'rbc_count', 'erythrocytes', 'red_cells', 'red_count', 'erythrocyte_count', 'eritrosit_count'],
    'htn': ['htn', 'hypertension', 'high_blood_pressure', 'tekanan_darah_tinggi', 'hypertensive', 'hypert', 'hipertensi', 'highbp'],
    'dm': ['dm', 'diabetes_mellitus', 'diabetes', 'kencing_manis', 'diabetic', 'dm_status', 'diabetic_status', 'diabet'],
    'cad': ['cad', 'coronary_artery_disease', 'heart_disease', 'jantung_koroner', 'coronary_disease', 'coronary', 'artery_disease', 'heart_problems'],
    'appet': ['appet', 'appetite', 'good_appetite', 'nafsu_makan', 'appetite_status', 'food_intake', 'eating', 'hunger'],
    'pe': ['pe', 'pedal_edema', 'edema', 'bengkak_kaki', 'leg_swelling', 'ankle_swelling', 'feet_swelling', 'swelling'],
    'ane': ['ane', 'anemia', 'anaemia', 'blood_deficiency', 'low_hemoglobin', 'anemic', 'anaemic', 'kurang_darah'],
    'bmi': ['bmi', 'body_mass_index', 'bmibaseline', 'bmi_baseline', 'bmi_value', 'indeks_massa_tubuh', 'imt', 'mass_index']
}

# Enhanced categorical feature mappings with more variations
CATEGORICAL_MAPPINGS = {
    'rbc': {'normal': 0, 'abnormal': 1, 'present': 1, 'not present': 0, 'yes': 1, 'no': 0},
    'pc': {'normal': 0, 'abnormal': 1, 'present': 1, 'not present': 0, 'yes': 1, 'no': 0},
    'pcc': {'not present': 0, 'present': 1, 'yes': 1, 'no': 0, 'notpresent': 0},
    'ba': {'not present': 0, 'present': 1, 'yes': 1, 'no': 0, 'notpresent': 0},
    'htn': {'no': 0, 'yes': 1, 'normal': 0, 'abnormal': 1, 'false': 0, 'true': 1, '-': 0, '+': 1, 'neg': 0, 'pos': 1, '0': 0, '1': 1},
    'dm': {'no': 0, 'yes': 1, 'normal': 0, 'abnormal': 1, 'false': 0, 'true': 1, '-': 0, '+': 1, 'neg': 0, 'pos': 1, '0': 0, '1': 1},
    'cad': {'no': 0, 'yes': 1, 'normal': 0, 'abnormal': 1, 'false': 0, 'true': 1, '-': 0, '+': 1, 'neg': 0, 'pos': 1, '0': 0, '1': 1},
    'appet': {'poor': 1, 'good': 0, 'abnormal': 1, 'normal': 0, 'reduced': 1, 'decreased': 1, 'bad': 1, 'low': 1, 'high': 0, 'fine': 0},
    'pe': {'no': 0, 'yes': 1, 'normal': 0, 'abnormal': 1, 'false': 0, 'true': 1, '-': 0, '+': 1, 'neg': 0, 'pos': 1, '0': 0, '1': 1},
    'ane': {'no': 0, 'yes': 1, 'normal': 0, 'abnormal': 1, 'false': 0, 'true': 1, '-': 0, '+': 1, 'neg': 0, 'pos': 1, '0': 0, '1': 1},
    'target': {'ckd': 1, 'notckd': 0, 'yes': 1, 'no': 0, 'present': 1, 'not present': 0, 'positive': 1, 'negative': 0, 
              'true': 1, 'false': 0, '1': 1, '0': 0, 't': 1, 'f': 0, '1.0': 1, '0.0': 0, '+': 1, '-': 0, 'pos': 1, 'neg': 0, 
              'disease': 1, 'normal': 0, 'ckd+': 1, 'ckd-': 0}
}

class CKDDataIntegrator:
    def __init__(self, debug_mode=False):
        # Set debug mode
        self.debug_mode = debug_mode
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Configure base paths
        self.script_dir = Path(os.getcwd()).absolute()
        self.app_dir = Path(os.path.dirname(os.path.abspath(__file__))).absolute()
        
        # Setup for project structure
        self.webapp_dir = None
        for parent in [self.script_dir, self.app_dir]:
            possible_webapp = parent / "CKD_Prediction_WebApp"
            if possible_webapp.exists() and possible_webapp.is_dir():
                self.webapp_dir = possible_webapp
                break
        
        # If we found the webapp directory, use it as parent for other directories
        if self.webapp_dir:
            logging.info(f"Found CKD_Prediction_WebApp directory: {self.webapp_dir}")
            self.dataset_dir = self.webapp_dir / "Dataset_Project_CKD"  # Fixed directory name
            self.output_dir = self.webapp_dir / "integrated_data"
        else:
            # Fallback to regular search for dataset directory
            dataset_dir_names = [
                "Dataset_Project_CKD",
                "Dataset_Project_CDK",
                "dataset",
                "data",
                "ckd_dataset",
                "ckd_data"
            ]
            
            # Search for dataset directory
            self.dataset_dir = None
            for dir_name in dataset_dir_names:
                possible_dir = self.script_dir / dir_name
                if possible_dir.exists() and possible_dir.is_dir():
                    self.dataset_dir = possible_dir
                    break
                
                possible_dir = self.script_dir.parent / dir_name
                if possible_dir.exists() and possible_dir.is_dir():
                    self.dataset_dir = possible_dir
                    break
            
            # If still not found, create it in the script directory
            if not self.dataset_dir:
                self.dataset_dir = self.script_dir / "Dataset_Project_CKD"
                self.dataset_dir.mkdir(exist_ok=True)
                logging.warning(f"Dataset directory not found. Created: {self.dataset_dir}")
                logging.warning("Please place your CKD data files in this directory and run again.")
            
            self.output_dir = self.script_dir / "integrated_data"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "merged_ckd_data.csv"
        
        logging.info(f"Dataset directory: {self.dataset_dir}")
        logging.info(f"Output directory: {self.output_dir}")
        
        # Define dtype map for output columns with more forgiving types
        self.base_dtype_map = {
            'target': 'float32',  # Changed from int8 to float32 to avoid NA errors
            'age': 'float32',
            'bp': 'float32',
            'sg': 'float32',
            'al': 'float32',
            'su': 'float32',
            'rbc': 'float32',  # Changed from int8 to float32
            'pc': 'float32',   # Changed from int8 to float32
            'pcc': 'float32',  # Changed from int8 to float32
            'ba': 'float32',   # Changed from int8 to float32
            'bgr': 'float32',
            'bu': 'float32',
            'sc': 'float32',
            'sod': 'float32',
            'pot': 'float32',
            'hemo': 'float32',
            'pcv': 'float32',
            'wc': 'float32',
            'rc': 'float32',
            'htn': 'float32',  # Changed from int8 to float32
            'dm': 'float32',   # Changed from int8 to float32
            'cad': 'float32',  # Changed from int8 to float32
            'appet': 'float32', # Changed from int8 to float32
            'pe': 'float32',    # Changed from int8 to float32
            'ane': 'float32',   # Changed from int8 to float32
            'bmi': 'float32'
        }
        
        self.unit_conversions = {
            'hemo': {'g/dl': 1, 'g/l': 0.1, 'mg/dl': 0.01, 'g%': 1},
            'sc': {'mg/dl': 1, 'μmol/l': 0.0113, 'umol/l': 0.0113, 'mmol/l': 11.312},
            'bgr': {'mg/dl': 1, 'mmol/l': 18.018, 'mg%': 1},
            'bmi': {'kg/m2': 1, 'kg/m²': 1, 'kg/m^2': 1, '': 1}
        }
        
        # Important features for CKD prediction according to literature
        self.core_features = ['age', 'bp', 'sc', 'hemo', 'bgr', 'bu', 'al', 'sg', 'dm', 'htn', 'bmi']
        
        # Dictionary to store mapping statistics for debugging
        self.mapping_stats = {
            'total_files_processed': 0,
            'successful_files': 0,
            'columns_mapped': {},
            'columns_not_mapped': {}
        }

    def _clean_numeric(self, value, feature):
        """Clean and convert numeric values, handling units and ranges."""
        try:
            if pd.isna(value) or value in ['?', '', 'nan', 'null', 'NaN', 'NA', 'None']:
                return np.nan
                
            # Handle categorical values first
            if feature in CATEGORICAL_MAPPINGS and isinstance(value, str):
                value_lower = value.lower().strip()
                for k, v in CATEGORICAL_MAPPINGS[feature].items():
                    if k.lower() == value_lower:
                        return float(v)  # Convert to float to handle NaN better
                # Try partial matches
                for k, v in CATEGORICAL_MAPPINGS[feature].items():
                    if k.lower() in value_lower:
                        return float(v)  # Convert to float
                # If numeric, check if 0/1
                if re.match(r'^\d+$', value_lower):
                    num_val = float(value_lower)  # Use float instead of int
                    if num_val in [0, 1]:
                        return num_val
                return np.nan
            
            # Handle BMI
            if feature == 'bmi' and isinstance(value, str):
                cleaned_value = re.sub(r'[^0-9.]', '', value)
                return float(cleaned_value) if cleaned_value else np.nan
            
            # Handle blood pressure
            if feature == 'bp' and isinstance(value, str) and '/' in value:
                systolic = value.split('/')[0].strip()
                return float(systolic) if systolic and re.match(r'^[0-9.]+$', systolic) else np.nan
                
            # Handle units
            if feature in self.unit_conversions and isinstance(value, str):
                for unit, factor in self.unit_conversions[feature].items():
                    if unit in value.lower():
                        num_part = re.sub(r'[^\d.]', '', value.lower().replace(unit.lower(), ''))
                        return float(num_part) * factor if num_part else np.nan
            
            # General numeric conversion
            if isinstance(value, str):
                value = value.replace(',', '.').strip()
                value = re.sub(r'[^\d.-]', '', value)
                if not value:
                    return np.nan
                # Attempt float conversion
                num_value = float(value)
                
                # Apply medical range constraints
                if feature in MEDICAL_RANGES:
                    min_val, max_val = MEDICAL_RANGES[feature]
                    if num_value < min_val:
                        logging.debug(f"Clamping {feature} value {num_value} to min {min_val}")
                        return min_val
                    elif num_value > max_val:
                        logging.debug(f"Clamping {feature} value {num_value} to max {max_val}")
                        return max_val
                return num_value
            else:
                return float(value)
        except Exception as e:
            if self.debug_mode:
                logging.debug(f"Error cleaning {value} for {feature}: {str(e)}")
            return np.nan

    def _clean_column(self, column, feature_name):
        """Clean and normalize a single column"""
        try:
            # Create new series first to avoid modifying original
            clean_series = pd.Series(index=column.index, dtype='float32')
            
            # Handle categorical mappings first for string values
            for i, val in column.items():
                clean_series[i] = self._clean_numeric(val, feature_name)
            
            # Fill NaN with feature-specific strategy
            if feature_name in self.core_features:
                non_nan_values = clean_series[~pd.isna(clean_series)]
                if len(non_nan_values) > 0:
                    fill_value = non_nan_values.median()
                    clean_series.fillna(fill_value, inplace=True)
                
            return clean_series.astype(self.base_dtype_map.get(feature_name, 'float32'))
        except Exception as e:
            logging.debug(f"Error cleaning column {feature_name}: {str(e)}")
            return pd.Series(index=column.index, dtype='float32')

    def _get_column_map(self, columns):
        """Map columns to standard feature names with improved exact matching"""
        column_map = {}
        for col in columns:
            col_lower = col.lower().strip()
            found = False
            
            # First try exact match (prioritize this)
            for standard_name, variants in FEATURE_PRIORITY.items():
                if col_lower in variants:
                    column_map[col] = standard_name
                    found = True
                    break
                    
            # If no exact match, try substring match
            if not found:
                for standard_name, variants in FEATURE_PRIORITY.items():
                    if any(variant in col_lower for variant in variants):
                        column_map[col] = standard_name
                        found = True
                        break
                        
            if not found:
                column_map[col] = None
                
        return column_map

    def _process_file(self, file_path):
        """Process a single CSV file with enhanced validation and column mapping"""
        try:
            # Read with multiple encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    # Try with different delimiters too
                    for delimiter in [',', ';', '\t', '|']:
                        try:
                            temp_df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                            if len(temp_df.columns) > 1:  # Ensure we have actual columns
                                df = temp_df
                                break
                        except:
                            continue
                    if df is not None:
                        break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                # Last resort - try with python's open first to detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    
                import chardet
                detected = chardet.detect(raw_data)
                if detected['encoding']:
                    try:
                        df = pd.read_csv(file_path, encoding=detected['encoding'])
                    except:
                        raise ValueError("Failed to decode file with detected encoding")
                else:
                    raise ValueError("Failed to decode file")

            # Normalize column names
            df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', col.strip().lower()) for col in df.columns]
            
            # Map columns to standard names
            column_map = self._get_column_map(df.columns)
            
            # Track mapped columns
            mapped_cols = {k: v for k, v in column_map.items() if v}
            unmapped_cols = [k for k, v in column_map.items() if not v]
            
            # Update mapping stats
            self.mapping_stats['total_files_processed'] += 1
            for col in mapped_cols:
                self.mapping_stats['columns_mapped'][col] = self.mapping_stats['columns_mapped'].get(col, 0) + 1
            for col in unmapped_cols:
                self.mapping_stats['columns_not_mapped'][col] = self.mapping_stats['columns_not_mapped'].get(col, 0) + 1

            # Clean and transform data - improved to handle errors better
            cleaned_data = {}
            for orig_col, std_col in column_map.items():
                if std_col:
                    try:
                        cleaned_data[std_col] = self._clean_column(df[orig_col], std_col)
                    except Exception as e:
                        logging.debug(f"Skipping column {orig_col} mapped to {std_col}: {e}")
                    
            # Create cleaned dataframe
            cleaned_df = pd.DataFrame(cleaned_data)
            
            # Handle target column specifically - ensure it's present
            if 'target' not in cleaned_df.columns and any(col.lower() in ['class', 'diagnosis', 'result'] for col in df.columns):
                for col in df.columns:
                    if col.lower() in ['class', 'diagnosis', 'result']:
                        try:
                            # Try to infer CKD status from column values
                            series = df[col].apply(lambda x: 1 if str(x).lower() in ['ckd', 'yes', '1', 'positive', 'true'] else 
                                                 0 if str(x).lower() in ['notckd', 'no', '0', 'negative', 'false'] else np.nan)
                            cleaned_df['target'] = series.astype('float32')
                            break
                        except:
                            pass
                            
            # Detect if this is a dataset with demographic info (has age, gender, etc.)
            is_demographic = 'age' in cleaned_df.columns or 'bmi' in cleaned_df.columns
            
            # Detect if this is a dataset with lab tests (has sc, hemo, etc.)
            is_lab_test = 'sc' in cleaned_df.columns or 'hemo' in cleaned_df.columns
            
            return cleaned_df, is_demographic, is_lab_test
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return None, False, False

    def _generate_synth_data(self, n_samples=100):
        """Generate synthetic data with core features for testing or filling gaps"""
        np.random.seed(42)  # For reproducibility
        
        synth_data = {
            'age': np.random.normal(loc=60, scale=15, size=n_samples).clip(18, 100),
            'bmi': np.random.normal(loc=26, scale=5, size=n_samples).clip(18, 45),
            'bp': np.random.normal(loc=130, scale=20, size=n_samples).clip(90, 190),
            'sc': np.random.normal(loc=2.5, scale=1.5, size=n_samples).clip(0.5, 7.0),
            'hemo': np.random.normal(loc=12, scale=2, size=n_samples).clip(8, 18),
            'bgr': np.random.normal(loc=150, scale=50, size=n_samples).clip(70, 300),
            'bu': np.random.normal(loc=60, scale=30, size=n_samples).clip(15, 150),
            'dm': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
            'htn': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
            'al': np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
            'sg': np.random.normal(loc=1.02, scale=0.01, size=n_samples).clip(1.005, 1.03),
            'target': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(synth_data)

    def _merge_dataframes(self, dataframes, demographic_flags, lab_test_flags):
        """Merge multiple dataframes with improved strategy"""
        if not dataframes:
            return self._generate_synth_data()
            
        # Separate demographic and lab test dataframes
        demographic_dfs = [df for df, is_demo, _ in zip(dataframes, demographic_flags, lab_test_flags) if is_demo and df is not None]
        lab_test_dfs = [df for df, _, is_lab in zip(dataframes, demographic_flags, lab_test_flags) if is_lab and df is not None]
        other_dfs = [df for df, is_demo, is_lab in zip(dataframes, demographic_flags, lab_test_flags) 
                    if df is not None and not is_demo and not is_lab]
        
        # Concat each category
        demo_df = pd.concat(demographic_dfs, ignore_index=True) if demographic_dfs else None
        lab_df = pd.concat(lab_test_dfs, ignore_index=True) if lab_test_dfs else None
        other_df = pd.concat(other_dfs, ignore_index=True) if other_dfs else None
        
        # If we have both demographic and lab data, try to match/merge them
        if demo_df is not None and lab_df is not None:
            # Try to find common identifier columns
            common_cols = set(demo_df.columns) & set(lab_df.columns)
            if 'target' in common_cols:
                common_cols.remove('target')  # Don't merge on target
                
            if common_cols:
                # Use common columns for merging
                merged_df = pd.merge(demo_df, lab_df, on=list(common_cols), how='outer')
                
                # Handle duplicated columns from merge
                for col in merged_df.columns:
                    if col.endswith('_x') or col.endswith('_y'):
                        base_col = col[:-2]
                        if f"{base_col}_x" in merged_df.columns and f"{base_col}_y" in merged_df.columns:
                            # Combine the columns, taking non-NaN values from either
                            merged_df[base_col] = merged_df[f"{base_col}_x"].combine_first(merged_df[f"{base_col}_y"])
                            merged_df.drop([f"{base_col}_x", f"{base_col}_y"], axis=1, inplace=True)
            else:
                # If no common columns, do a cartesian product with weights
                merged_df = pd.concat([demo_df, lab_df], axis=1, ignore_index=False)
                
            # Add other data if available
            if other_df is not None:
                for col in other_df.columns:
                    if col not in merged_df.columns:
                        merged_df[col] = np.nan
                    
                    # FIX: The problematic line - more cautious copying of values from other_df
                    nan_indices = merged_df[col].isna()
                    if nan_indices.any() and not other_df[col].empty:
                        # Get non-NaN values from other_df
                        non_nan_mask = ~other_df[col].isna()
                        if non_nan_mask.any():
                            # Only take the number of values we need
                            num_nan = nan_indices.sum()
                            num_available = non_nan_mask.sum()
                            # Make sure we don't try to assign more values than we have available
                            values_to_use = other_df.loc[non_nan_mask, col].values[:min(num_nan, num_available)]
                            # Only assign to as many NaN positions as we have values
                            nan_indices_idx = nan_indices[nan_indices].index[:len(values_to_use)]
                            merged_df.loc[nan_indices_idx, col] = values_to_use
        else:
            # Just use whatever dataframes we have
            all_dfs = []
            if demo_df is not None:
                all_dfs.append(demo_df)
            if lab_df is not None:
                all_dfs.append(lab_df)
            if other_df is not None:
                all_dfs.append(other_df)
                
            if all_dfs:
                merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
            else:
                # Fallback to synthetic data if no real data is available
                merged_df = self._generate_synth_data()
                
        # Ensure all required columns exist
        for col, dtype in self.base_dtype_map.items():
            if col not in merged_df.columns:
                merged_df[col] = np.nan
                
            # Clean the column
            merged_df[col] = self._clean_column(merged_df[col], col)
            
        # Final cleanup - remove duplicates
        merged_df = merged_df.drop_duplicates().reset_index(drop=True)
        
        # Handle missing values
        for col in self.core_features:
            if col in merged_df.columns and merged_df[col].isna().any():
                non_nan = merged_df[col][~merged_df[col].isna()]
                if len(non_nan) > 0:
                    median_val = non_nan.median()
                    merged_df[col].fillna(median_val, inplace=True)
                    
        # Set final dtypes
        for col, dtype in self.base_dtype_map.items():
            if col in merged_df.columns:
                try:
                    merged_df[col] = merged_df[col].astype(dtype)
                except:
                    logging.warning(f"Failed to convert {col} to {dtype}")
                    
        return merged_df

    def integrate(self):
        """Integrate CKD data from multiple sources with balanced strategy"""
        try:
            # Get all CSV files in the dataset directory
            csv_files = list(self.dataset_dir.glob('**/*.csv'))
            logging.info(f"Found {len(csv_files)} CSV files in {self.dataset_dir}")
            
            if not csv_files:
                logging.warning("No CSV files found. Please place your CKD data files in the dataset directory.")
                # Generate synthetic data as a fallback
                synthetic_df = self._generate_synth_data(500)
                synthetic_df.to_csv(self.output_path, index=False)
                logging.info(f"Generated synthetic data and saved to {self.output_path}")
                return synthetic_df
            
            # Process files in parallel
            dataframes = []
            demographic_flags = []
            lab_test_flags = []
            
            with ThreadPoolExecutor(max_workers=min(10, len(csv_files))) as executor:
                futures = {executor.submit(self._process_file, file_path): file_path for file_path in csv_files}
                
                for future in tqdm(futures, desc="Processing files"):
                    file_path = futures[future]
                    try:
                        df, is_demographic, is_lab_test = future.result()
                        if df is not None and not df.empty:
                            dataframes.append(df)
                            demographic_flags.append(is_demographic)
                            lab_test_flags.append(is_lab_test)
                            self.mapping_stats['successful_files'] += 1
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {str(e)}")
            
            # Merge the dataframes
            logging.info(f"Successfully processed {self.mapping_stats['successful_files']} out of {self.mapping_stats['total_files_processed']} files.")
            
            if not dataframes:
                logging.warning("No valid data found in any files. Generating synthetic data...")
                merged_df = self._generate_synth_data(500)
            else:
                merged_df = self._merge_dataframes(dataframes, demographic_flags, lab_test_flags)
                
            # Save the result
            merged_df.to_csv(self.output_path, index=False)
            logging.info(f"Integrated data saved to {self.output_path}")
            
            if self.debug_mode:
                # Save mapping statistics
                import json
                with open(self.output_dir / "mapping_stats.json", 'w') as f:
                    json.dump(self.mapping_stats, f, indent=2)
                    
                # Save sample of each input file for debugging
                for i, (df, file_path) in enumerate(zip(dataframes, csv_files)):
                    if df is not None and not df.empty:
                        sample = df.head(5)
                        sample_path = self.output_dir / f"sample_{i}_{file_path.name}"
                        sample.to_csv(sample_path, index=False)
            
            return merged_df
            
        except Exception as e:
            logging.error(f"Error in integration process: {str(e)}")
            # Generate fallback data
            synthetic_df = self._generate_synth_data(500)
            synthetic_df.to_csv(self.output_path, index=False)
            logging.info(f"Generated fallback synthetic data and saved to {self.output_path}")
            return synthetic_df

if __name__ == "__main__":
    start_time = time.time()
    
    # Create integrator instance - set debug=True for verbose logging
    integrator = CKDDataIntegrator(debug_mode=False)
    
    # Run integration
    merged_data = integrator.integrate()
    
    # Display summary info
    print(f"\nProcessed files: {integrator.mapping_stats['successful_files']} / {integrator.mapping_stats['total_files_processed']}")
    print(f"Final dataset shape: {merged_data.shape}")
    print(f"Columns: {', '.join(merged_data.columns)}")
    print(f"Non-null counts:\n{merged_data.count()}")
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    
    # Optional: display first few rows
    print("\nSample data:")
    print(merged_data.head())