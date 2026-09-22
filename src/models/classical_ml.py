import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def build_radiomics_pipeline(classifier_type: str = 'xgboost', num_classes: int = 4) -> Pipeline:
    """
    Constructs an end-to-end preprocessing and classification pipeline for radiomics features.
    
    Args:
        classifier_type: 'xgboost' or 'random_forest'. Both are supported for SHAP 
                         and uncertainty extraction[cite: 2, 3].
        num_classes: Number of dementia stages (default is 4).
    """
    # --- 1. Preprocessing Steps ---
    # Impute missing values with the median of the column[cite: 2]
    imputer = SimpleImputer(strategy='median')
    
    # Remove features that have the same value across all samples (constant features)[cite: 2]
    var_filter = VarianceThreshold(threshold=0.0) 
    
    # Apply z-score normalization to standardize feature scales[cite: 2]
    scaler = StandardScaler() 
    
    # --- 2. Classifier Configuration ---
    if classifier_type.lower() == 'xgboost':
        clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            random_state=42
        )
    elif classifier_type.lower() == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced' # Assists with the dataset's class imbalance
        )
    else:
        raise ValueError("classifier_type must be either 'xgboost' or 'random_forest'")
        
    # --- 3. Assemble Pipeline ---
    pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('variance_filter', var_filter),
        ('scaler', scaler),
        ('classifier', clf)
    ])
    
    return pipeline

# Example execution during training:
# radiomics_pipeline = build_radiomics_pipeline(classifier_type='xgboost')
# radiomics_pipeline.fit(X_train_radiomics_df, y_train)