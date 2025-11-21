#!/usr/bin/env python3
import os, time
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

folder_path = "../data/processed_project"
numerical_features = ["elv", "fill", "slope", "asp", "flow", "precip"]
categorical_features = ["soil"]
features = numerical_features + categorical_features
target = "wet"
perform_grid_search = True
grid_cs = [0.01, 0.1, 1.0, 10.0]
random_state = 42
test_size = 0.2
n_jobs = -1

def load_geojsons(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".geojson")])
    if not files:
        raise FileNotFoundError(f"No .geojson files found in {folder}")
    gdfs = [gpd.read_file(os.path.join(folder, f)) for f in files]
    return pd.concat(gdfs, ignore_index=True)

def prepare_data(df, target):
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    X = df[features].copy()
    y = df[target].astype(int).copy()
    for c in categorical_features:
        X[c] = X[c].astype(str)
    return X, y

def build_pipeline(random_state):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numerical_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features)
    ])

    clf = LogisticRegression(
        penalty="l2", solver="saga", class_weight="balanced", max_iter=2000, random_state=random_state, n_jobs=n_jobs
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def hyperparameter_search(pipe, X_train, y_train):
    param_grid = {"clf__C": grid_cs}
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=n_jobs, verbose=1)
    start = time.time()
    grid.fit(X_train, y_train)
    print(f"Grid search completed in {time.time() - start:.1f}s")
    print("Best C:", grid.best_params_["clf__C"])
    return grid.best_estimator_

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("F1 score:", f1_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

def show_feature_importance(model):
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel()
    try:
        names = model.named_steps["pre"].get_feature_names_out()
    except Exception:
        names = np.array(features)
    df = pd.DataFrame({"feature": names, "coef": coefs, "abs_coef": np.abs(coefs)})
    df = df.sort_values("abs_coef", ascending=False)
    print("\nAll feature coefficients (sorted by magnitude):")
    print(df.to_string(index=False))

def main():
    df = load_geojsons(folder_path)
    X, y = prepare_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipe = build_pipeline(random_state)
    model = hyperparameter_search(pipe, X_train, y_train) if perform_grid_search else pipe.fit(X_train, y_train)
    evaluate(model, X_test, y_test)
    show_feature_importance(model)
    joblib.dump(model, "linear_model.pkl")
    print("Model saved as linear_model.pkl")

if __name__ == "__main__":
    main()