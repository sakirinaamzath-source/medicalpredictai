import os
import glob
import joblib
import pandas as pd
from sklearn.tree import export_text

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("pkl_view", exist_ok=True)


# ============================================================
# 1. VIEW ALL FEATURE FILES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE PKL FILES")
print("=" * 70)

for file in glob.glob("features_*.pkl"):

    features = joblib.load(file)

    disease = (
        os.path.basename(file)
        .replace("features_", "")
        .replace(".pkl", "")
    )

    print(f"\n📁 {file}")
    print(features)

    # Save as CSV
    df = pd.DataFrame({
        "No": range(1, len(features) + 1),
        "Feature": features
    })

    output = f"pkl_view/{disease}_features.csv"
    df.to_csv(output, index=False)

    print(f"✅ Saved readable file: {output}")


# ============================================================
# 2. VIEW ALL SCALER FILES
# ============================================================

print("\n" + "=" * 70)
print("SCALER PKL FILES")
print("=" * 70)

for file in glob.glob("scaler_*.pkl"):

    scaler = joblib.load(file)

    disease = (
        os.path.basename(file)
        .replace("scaler_", "")
        .replace(".pkl", "")
    )

    feature_file = f"features_{disease}.pkl"
    features = joblib.load(feature_file)

    print(f"\n📁 {file}")
    print("Type:", type(scaler))

    # Create readable table
    df = pd.DataFrame({
        "Feature": features,
        "Mean": scaler.mean_,
        "Variance": scaler.var_,
        "Scale_StdDev": scaler.scale_
    })

    print(df.to_string(index=False))

    # Save as CSV
    output = f"pkl_view/{disease}_scaler.csv"
    df.to_csv(output, index=False)

    print(f"✅ Saved readable file: {output}")


# ============================================================
# 3. VIEW ALL MODEL FILES
# ============================================================

print("\n" + "=" * 70)
print("MODEL PKL FILES")
print("=" * 70)

for file in glob.glob("model_*.pkl"):

    model = joblib.load(file)

    disease = (
        os.path.basename(file)
        .replace("model_", "")
        .replace(".pkl", "")
    )

    feature_file = f"features_{disease}.pkl"
    features = joblib.load(feature_file)

    print(f"\n📁 {file}")
    print("Model Type:", type(model))
    print("Number of Features:", model.n_features_in_)

    # Show model classes
    if hasattr(model, "classes_"):
        print("Classes:", model.classes_)

    # Show feature names stored inside the model
    if hasattr(model, "feature_names_in_"):
        print("Model Feature Names:")
        print(list(model.feature_names_in_))

    # ----------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # ----------------------------------------

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))

        output = f"pkl_view/{disease}_model_importance.csv"
        importance_df.to_csv(output, index=False)

        if hasattr(model, "estimators_"):

            tree_output = f"pkl_view/{disease}_all_trees.txt"

            with open(
                tree_output,
                "w",
                encoding="utf-8"
            ) as f:

                for i, tree in enumerate(
                    model.estimators_,
                    start=1
                ):

                    f.write("\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"DECISION TREE #{i}\n")
                    f.write("=" * 70 + "\n")

                    rules = export_text(
                        tree,
                        feature_names=list(features)
                    )

                    f.write(rules)
                    f.write("\n")

            print(f"✅ Saved all trees: {tree_output}")