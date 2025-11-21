#!/usr/bin/env python3
import os
import time
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
grid_n_estimators = [100, 200]
random_state = 42
test_size = 0.2
n_jobs = -1

def load_geojsons(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".geojson")])
    if not files:
        raise FileNotFoundError(f"No .geojson files found in {folder}")
    gdfs = [gpd.read_file(os.path.join(folder, f)) for f in files]
    return pd.concat(gdfs, ignore_index=True)

def prepare_data(df, features, target):
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    X = df[features].copy()
    for c in categorical_features:
        if c in X.columns:
            X[c] = X[c].astype(str)
    y = df[target].astype(int).copy()
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
    clf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
        verbose=1
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def hyperparameter_search(pipe, X_train, y_train):
    param_grid = {"clf__n_estimators": grid_n_estimators}
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=n_jobs, verbose=1)
    start = time.time()
    grid.fit(X_train, y_train)
    print(f"Grid search completed in {time.time() - start:.1f}s")
    print("Best n_estimators:", grid.best_params_["clf__n_estimators"])
    return grid.best_estimator_

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("F1 score:", f1_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

def show_feature_importance(model):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    num_features = pre.named_transformers_["num"].named_steps["scale"].get_feature_names_out(numerical_features)
    cat_features = pre.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_features, cat_features])
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    print(fi_df.to_string(index=False))

def main():
    df = load_geojsons(folder_path)
    X, y = prepare_data(df, features, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    pipe = build_pipeline(random_state)
    model = hyperparameter_search(pipe, X_train, y_train) if perform_grid_search else pipe.fit(X_train, y_train)
    evaluate(model, X_test, y_test)
    show_feature_importance(model)
    joblib.dump(model, "rf_model.pkl")
    print("Model saved as rf_model.pkl")

if __name__ == "__main__":
    main()