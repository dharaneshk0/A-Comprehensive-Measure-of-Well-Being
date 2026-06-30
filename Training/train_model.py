"""
Executes the HDI training pipeline (Epics 3-7) end-to-end and saves HDI.pkl.

This mirrors exactly what HumDevIndex.ipynb does, so the saved model is the
same one the notebook would produce.
"""
import pandas as pd
import numpy as np
import matploylib.pylot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    dataset_path = os.path.join(project, "Dataset", "HDI.csv")
    model_path = os.path.join(project, "Flask", "HDI.pkl")

    # --- Epic 3: load dataset ---
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows from {dataset_path}")

    # --- Epic 4: select features / target ---
    feature_cols = [
        "Life expectancy",
        "Mean years of schooling",
        "Expected years of schooling",
        "Gross National Income (GNI) per capita",
    ]
    X = df[feature_cols]
    y = df["HDI"]

    # Check for nulls (handled if any, here there are none)
    if df[feature_cols + ["HDI"]].isnull().any().any():
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        y = y.fillna(y.median())
        print("Missing values handled.")

    # --- Epic 5: train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # --- Epic 6: fit + evaluate ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # --- Epic 7: save model ---
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model -> {model_path}")

    # Sanity-check the three project scenarios
    scenarios = {
        "Very High (Norway-like)": [82.6, 12.9, 18.2, 66000],
        "Medium (emerging)":       [72.5, 8.0,  13.0, 11000],
        "Low (intervention)":      [60.0, 3.5,  8.0,  2000],
    }
    print("\nScenario predictions:")
    for name, vals in scenarios.items():
        row = pd.DataFrame([vals], columns=feature_cols)
        pred = float(model.predict(row)[0])
        print(f"  {name:25s} -> HDI = {pred:.3f}")


if __name__ == "__main__":
    main()
