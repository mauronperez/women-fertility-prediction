"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "app/model"
ARTIFACTS_DIR = MODEL_DIR  # Directory containing feature_columns.txt and preprocessing.pkl

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from src/serving/model first (preferred local path)
        import glob
        local_model_paths = glob.glob("src/serving/model/*/artifacts/model")
        if not local_model_paths:
            # Fallback to mlruns if src/serving/model doesn't exist
            local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            
            # CRITICAL: Set ARTIFACTS_DIR to parent directory containing the artifacts
            # latest_model = "src/serving/model/.../artifacts/model"
            # artifacts_dir = "src/serving/model/.../artifacts"
            ARTIFACTS_DIR = os.path.dirname(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
            print(f"✅ Using artifacts directory: {ARTIFACTS_DIR}")
        else:
            raise Exception("No model found in src/serving/model or mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
try:
    feature_file = os.path.join(ARTIFACTS_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "age": {"Less than 30": 0, "Older than 30": 1},           # Demographics
    "married": {"No": 0, "Yes": 1},               # Has partner
    "job": {"No": 0, "Yes": 1},            # Has dependents  
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["housing_payment","level_studies","income","number_living","house_income"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw woman data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
        Main prediction function for fertility outcome inference.
        
        This function provides the complete inference pipeline from raw patient data
        to business-friendly prediction output. It's called by both the FastAPI endpoint
        and the Gradio interface to ensure consistent predictions.
        
        Pipeline:
        1. Convert input dictionary to DataFrame
        2. Apply feature transformations (identical to training)
        3. Generate model prediction using loaded XGBoost model
        4. Convert prediction to user-friendly string
        
        Args:
            input_dict: Dictionary containing raw patient data with keys matching
                    the FertilityData schema (features may include age, BMI, 
                    hormonal levels, medical history, lifestyle factors, etc.)
                    
        Returns:
            Human-readable prediction string:
            - "Likely to have kids" for patients with favorable fertility indicators (model prediction = 1)
            - "Not likely to have kids" for patients with concerning indicators (model prediction = 0)
            
        Example:
            >>> patient_data = {
            ...     "age": "Less than 30", "Ever married": "Yes" ... # other features
            ... }
            >>> predict(patient_data)
            "Not likely to have kids"
        """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Output ===
    if result == 1:
        return "She's likely have kids"      # High risk - needs intervention
    else:
        return "Not likely to have kids"  # Low risk - maintain normal service
