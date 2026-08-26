import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# HELPER: CLEAN AND ENCODE A BINARY TARGET COLUMN
# ============================================================

def robust_binary_encoder(df, target_col):
    """
    Cleans a binary target column and converts it to 0 and 1.
    """
    df = df.copy()

    # Clean text values
    if df[target_col].dtype == "object" or str(df[target_col].dtype) == "string":
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    unique_vals = list(df[target_col].dropna().unique())

    # If already numeric 0/1, keep it
    if set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}:
        df[target_col] = df[target_col].astype(int)

    else:
        mapping = {}

        for val in unique_vals:
            val_str = str(val).strip().lower()

            if val_str in [
                "0", "0.0", "n", "no", "negative",
                "normal", "notckd", "absence", "healthy"
            ]:
                mapping[val] = 0

            elif val_str in [
                "1", "1.0", "y", "yes", "positive",
                "abnormal", "ckd", "presence", "disease"
            ]:
                mapping[val] = 1

        # If there are exactly two unknown classes, map them automatically
        if not mapping and len(unique_vals) == 2:
            mapping = {
                unique_vals[0]: 0,
                unique_vals[1]: 1
            }

        if mapping:
            df[target_col] = df[target_col].map(mapping)

    # Remove rows with invalid/missing target
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    return df


# ============================================================
# MAIN PREPROCESSING + EXPORT FUNCTION
# ============================================================

def process_features_and_scale(df, target_col, scaler_filename):
    """
    1. Clean target
    2. Separate features and target
    3. Encode categorical features
    4. Split 80% train / 20% test
    5. Scale using StandardScaler
    6. Save all stages as CSV files
    7. Save scaler and feature names as PKL files
    """

    # Create output folder automatically
    os.makedirs("processed_data", exist_ok=True)

    # Get disease name from scaler filename
    # Example: scaler_kidney.pkl -> kidney
    disease_name = (
        scaler_filename
        .replace("scaler_", "")
        .replace(".pkl", "")
    )

    print(f"\n📂 Processing {disease_name.upper()} data...")

    # --------------------------------------------------------
    # 1. CLEAN TARGET
    # --------------------------------------------------------
    df = robust_binary_encoder(df, target_col)

    # Save cleaned dataset
    cleaned_path = f"processed_data/{disease_name}_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"   ✅ Saved: {cleaned_path}")

    # --------------------------------------------------------
    # 2. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # --------------------------------------------------------
    # 3. ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------
    X = pd.get_dummies(X, drop_first=True)

    # Convert boolean columns to integers
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Save encoded dataset
    encoded_df = X.copy()
    encoded_df[target_col] = y.values

    encoded_path = f"processed_data/{disease_name}_encoded.csv"
    encoded_df.to_csv(encoded_path, index=False)
    print(f"   ✅ Saved: {encoded_path}")

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Save original training set
    train_original = X_train.copy()
    train_original[target_col] = y_train

    train_original_path = (
        f"processed_data/{disease_name}_train_original.csv"
    )
    train_original.to_csv(train_original_path, index=False)
    print(f"   ✅ Saved: {train_original_path}")

    # Save original testing set
    test_original = X_test.copy()
    test_original[target_col] = y_test

    test_original_path = (
        f"processed_data/{disease_name}_test_original.csv"
    )
    test_original.to_csv(test_original_path, index=False)
    print(f"   ✅ Saved: {test_original_path}")

    # --------------------------------------------------------
    # 5. SCALE FEATURES
    # --------------------------------------------------------
    scaler = StandardScaler()

    # IMPORTANT:
    # Fit scaler ONLY on training data to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train.index
    )

    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test.index
    )

    # Add target column back
    X_train_scaled_df[target_col] = y_train
    X_test_scaled_df[target_col] = y_test

    # Save scaled training set
    train_scaled_path = (
        f"processed_data/{disease_name}_train_scaled.csv"
    )
    X_train_scaled_df.to_csv(train_scaled_path, index=False)
    print(f"   ✅ Saved: {train_scaled_path}")

    # Save scaled testing set
    test_scaled_path = (
        f"processed_data/{disease_name}_test_scaled.csv"
    )
    X_test_scaled_df.to_csv(test_scaled_path, index=False)
    print(f"   ✅ Saved: {test_scaled_path}")

    # --------------------------------------------------------
    # 6. SAVE SCALER AND FEATURE LIST
    # --------------------------------------------------------
    joblib.dump(scaler, scaler_filename)

    features_filename = scaler_filename.replace(
        "scaler_",
        "features_"
    )

    joblib.dump(list(X.columns), features_filename)

    print(f"   💾 Saved scaler: {scaler_filename}")
    print(f"   💾 Saved features: {features_filename}")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================
# DIABETES DATASET
# ============================================================

def preprocess_diabetes():
    print("\n⏳ Preprocessing Diabetes Dataset...")

    df = pd.read_csv("datasets/diabetes data.csv")

    # Replace impossible physiological zeros with median values
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in zero_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    target_col = (
        "Outcome"
        if "Outcome" in df.columns
        else df.columns[-1]
    )

    X_train, X_test, y_train, y_test = process_features_and_scale(
        df,
        target_col,
        "scaler_diabetes.pkl"
    )

    print("✅ Diabetes preprocessing complete.")

    return X_train, X_test, y_train, y_test


# ============================================================
# HEART DISEASE DATASET
# ============================================================

def preprocess_heart():
    print("\n⏳ Preprocessing Heart Disease Dataset...")

    df = pd.read_csv("datasets/heart data.csv")

    target_col = (
        "target"
        if "target" in df.columns
        else df.columns[-1]
    )

    # Convert multi-class heart target into binary:
    # 0 = no disease, >0 = disease
    if (
        pd.api.types.is_numeric_dtype(df[target_col])
        and df[target_col].max() > 1
    ):
        df[target_col] = df[target_col].apply(
            lambda x: 1 if x > 0 else 0
        )

    X_train, X_test, y_train, y_test = process_features_and_scale(
        df,
        target_col,
        "scaler_heart.pkl"
    )

    print("✅ Heart Disease preprocessing complete.")

    return X_train, X_test, y_train, y_test


# ============================================================
# KIDNEY DISEASE DATASET - NEW 7 FEATURE VERSION
# ============================================================

def preprocess_kidney():
    print("\n⏳ Preprocessing Chronic Kidney Disease Dataset...")

    df = pd.read_csv("datasets/kidney data.csv")

    # Use ONLY these 7 features from the new dataset
    features = [
        "Age",
        "Creatinine_Level",
        "BUN",
        "Diabetes",
        "Hypertension",
        "GFR",
        "Urine_Output"
    ]

    target_col = "CKD_Status"

    # Check that required columns exist
    required_columns = features + [target_col]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required kidney dataset columns are missing: "
            + ", ".join(missing_columns)
        )

    # Keep exactly 7 input features + 1 target
    df = df[features + [target_col]].copy()

    X_train, X_test, y_train, y_test = process_features_and_scale(
        df,
        target_col,
        "scaler_kidney.pkl"
    )

    print("✅ Kidney Disease preprocessing complete.")

    return X_train, X_test, y_train, y_test


# ============================================================
# RUN ALL PREPROCESSING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🚀 STARTING ALL DATA PREPROCESSING")
    print("=" * 60)

    X_train_dia, X_test_dia, y_train_dia, y_test_dia = (
        preprocess_diabetes()
    )

    X_train_hrt, X_test_hrt, y_train_hrt, y_test_hrt = (
        preprocess_heart()
    )

    X_train_kdn, X_test_kdn, y_train_kdn, y_test_kdn = (
        preprocess_kidney()
    )

    print("\n" + "=" * 60)
    print("🎉 ALL DATASETS PREPROCESSED SUCCESSFULLY!")
    print("📁 Check the 'processed_data' folder to view:")
    print("   • cleaned data")
    print("   • encoded data")
    print("   • training data before scaling")
    print("   • testing data before scaling")
    print("   • scaled training data")
    print("   • scaled testing data")
    print("=" * 60)
