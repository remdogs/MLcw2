"""Method development for PCR (classification) and RFS (regression) using the preprocessed dataset and selected features, including baselines, hyperparameter tuning, evaluation on a hold-out split, and training final models saved to models/."""
import json
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


RANDOM_STATE = 42
DATA_PATH = Path("Dataset/preprocessed_dataset_cleaned.csv")
PCR_FEATURE_LIST_PATH = Path("Dataset/SelectedFeaturePCR.csv")
RFS_FEATURE_LIST_PATH = Path("Dataset/SelectedFeaturesRFS.csv")
ID_COL = "ID"
PCR_COL = "pCR (outcome)"
RFS_COL = "RelapseFreeSurvival (outcome)"


def load_datasets():
    df = pd.read_csv(DATA_PATH)

    pcr_features_df = pd.read_csv(PCR_FEATURE_LIST_PATH)
    rfs_features_df = pd.read_csv(RFS_FEATURE_LIST_PATH)

    pcr_features = pcr_features_df["feature"].tolist()
    rfs_features = rfs_features_df["feature"].tolist()

    missing_pcr = set(pcr_features) - set(df.columns)
    missing_rfs = set(rfs_features) - set(df.columns)
    assert not missing_pcr, f"Missing PCR features in dataset: {missing_pcr}"
    assert not missing_rfs, f"Missing RFS features in dataset: {missing_rfs}"

    df_pcr = df.dropna(subset=[PCR_COL])
    df_rfs = df.dropna(subset=[RFS_COL])

    X_pcr = df_pcr[pcr_features].copy()
    y_pcr = df_pcr[PCR_COL].astype(int)
    X_rfs = df_rfs[rfs_features].copy()
    y_rfs = df_rfs[RFS_COL].astype(float)

    return df, X_pcr, y_pcr, X_rfs, y_rfs


def tune_pcr_models(X_train_pcr, y_pcr_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    log_reg_pcr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    rf_clf_pcr = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    svc_pcr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    log_reg_grid = GridSearchCV(
        log_reg_pcr,
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )
    rf_grid = GridSearchCV(
        rf_clf_pcr,
        {
            "n_estimators": [200, 500],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )
    svc_grid = GridSearchCV(
        svc_pcr,
        {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )

    log_reg_grid.fit(X_train_pcr, y_pcr_train)
    pd.DataFrame(log_reg_grid.cv_results_).to_csv(
        results_dir / "pcr_logreg_grid_results.csv", index=False
    )
    print(
        "Logistic Regression PCR - best params:",
        log_reg_grid.best_params_,
        "best balanced accuracy:",
        f"{log_reg_grid.best_score_:.4f}",
    )

    rf_grid.fit(X_train_pcr, y_pcr_train)
    pd.DataFrame(rf_grid.cv_results_).to_csv(
        results_dir / "pcr_rf_grid_results.csv", index=False
    )
    print(
        "Random Forest PCR - best params:",
        rf_grid.best_params_,
        "best balanced accuracy:",
        f"{rf_grid.best_score_:.4f}",
    )

    svc_grid.fit(X_train_pcr, y_pcr_train)
    print(
        "SVC PCR - best params:",
        svc_grid.best_params_,
        "best balanced accuracy:",
        f"{svc_grid.best_score_:.4f}",
    )

    best_grid = max(
        [log_reg_grid, rf_grid, svc_grid], key=lambda g: g.best_score_
    )
    return best_grid.best_estimator_


def tune_rfs_models(X_train_rfs, y_rfs_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    rf_reg_rfs = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    gbr_rfs = GradientBoostingRegressor(random_state=RANDOM_STATE)

    rf_grid = GridSearchCV(
        rf_reg_rfs,
        {
            "n_estimators": [200, 500],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )
    gbr_grid = GridSearchCV(
        gbr_rfs,
        {
            "n_estimators": [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3, 4],
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )

    rf_grid.fit(X_train_rfs, y_rfs_train)
    pd.DataFrame(rf_grid.cv_results_).to_csv(
        results_dir / "rfs_rf_grid_results.csv", index=False
    )
    gbr_grid.fit(X_train_rfs, y_rfs_train)
    pd.DataFrame(gbr_grid.cv_results_).to_csv(
        results_dir / "rfs_gbr_grid_results.csv", index=False
    )

    rf_mae_cv = -rf_grid.best_score_
    gbr_mae_cv = -gbr_grid.best_score_

    print(
        "Random Forest RFS - best params:",
        rf_grid.best_params_,
        "best CV MAE:",
        f"{rf_mae_cv:.4f}",
    )
    print(
        "Gradient Boosting RFS - best params:",
        gbr_grid.best_params_,
        "best CV MAE:",
        f"{gbr_mae_cv:.4f}",
    )

    if rf_mae_cv <= gbr_mae_cv:
        return rf_grid.best_estimator_
    return gbr_grid.best_estimator_


def train_and_save_final_models(best_pcr_model, best_rfs_model, features_pcr, features_rfs):
    df_full = pd.read_csv(DATA_PATH)

    df_full_pcr = df_full.dropna(subset=[PCR_COL])
    df_full_rfs = df_full.dropna(subset=[RFS_COL])

    X_full_pcr = df_full_pcr[features_pcr].copy()
    y_full_pcr = df_full_pcr[PCR_COL].astype(int)

    X_full_rfs = df_full_rfs[features_rfs].copy()
    y_full_rfs = df_full_rfs[RFS_COL].astype(float)

    best_pcr_model.fit(X_full_pcr, y_full_pcr)
    best_rfs_model.fit(X_full_rfs, y_full_rfs)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pcr_model, models_dir / "final_pcr_model.joblib")
    joblib.dump(best_rfs_model, models_dir / "final_rfs_model.joblib")

    with open(models_dir / "pcr_features.json", "w") as f_pcr:
        json.dump(features_pcr, f_pcr)
    with open(models_dir / "rfs_features.json", "w") as f_rfs:
        json.dump(features_rfs, f_rfs)

    print(f"Saved final PCR model and features to {models_dir}")
    print(f"Saved final RFS model and features to {models_dir}")


def _ensure_1d_target(target, col_name):
    if isinstance(target, pd.DataFrame):
        if col_name in target.columns:
            return target[col_name]
        return target.squeeze()
    return target.squeeze()


def run_method_development():
    _, X_pcr, y_pcr, X_rfs, y_rfs = load_datasets()

    features_pcr = list(X_pcr.columns)
    features_rfs = list(X_rfs.columns)

    X_train_pcr_raw, X_test_pcr_raw, y_train_pcr_raw, y_test_pcr_raw = train_test_split(
        X_pcr,
        y_pcr,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_pcr,
    )
    X_train_rfs_raw, X_test_rfs_raw, y_train_rfs_raw, y_test_rfs_raw = train_test_split(
        X_rfs, y_rfs, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train_pcr = X_train_pcr_raw[features_pcr].copy()
    X_test_pcr = X_test_pcr_raw[features_pcr].copy()
    X_train_rfs = X_train_rfs_raw[features_rfs].copy()
    X_test_rfs = X_test_rfs_raw[features_rfs].copy()

    y_train_pcr = _ensure_1d_target(y_train_pcr_raw, PCR_COL)
    y_test_pcr = _ensure_1d_target(y_test_pcr_raw, PCR_COL)
    y_train_rfs = _ensure_1d_target(y_train_rfs_raw, RFS_COL)
    y_test_rfs = _ensure_1d_target(y_test_rfs_raw, RFS_COL)

    print(f"X_train_pcr shape: {X_train_pcr.shape}")
    print(f"X_test_pcr shape: {X_test_pcr.shape}")
    print(f"X_train_rfs shape: {X_train_rfs.shape}")
    print(f"X_test_rfs shape: {X_test_rfs.shape}")
    print(f"y_train_pcr shape: {y_train_pcr.shape}")
    print(f"y_test_pcr shape: {y_test_pcr.shape}")
    print(f"y_train_rfs shape: {y_train_rfs.shape}")
    print(f"y_test_rfs shape: {y_test_rfs.shape}")
    print("Class balance y_train_pcr (counts):")
    print(y_train_pcr.value_counts(dropna=False))
    print("Class balance y_train_pcr (proportions):")
    print(y_train_pcr.value_counts(normalize=True, dropna=False))
    print("Class balance y_test_pcr (counts):")
    print(y_test_pcr.value_counts(dropna=False))
    print("Class balance y_test_pcr (proportions):")
    print(y_test_pcr.value_counts(normalize=True, dropna=False))

    baseline_pcr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    baseline_pcr.fit(X_train_pcr, y_train_pcr)
    y_pred_pcr = baseline_pcr.predict(X_test_pcr)
    baseline_pcr_bal_acc = balanced_accuracy_score(y_test_pcr, y_pred_pcr)
    print(f"Baseline PCR (LogReg) balanced accuracy: {baseline_pcr_bal_acc:.4f}")
    print("Baseline PCR classification report:")
    print(classification_report(y_test_pcr, y_pred_pcr, digits=4))
    cm_baseline = confusion_matrix(y_test_pcr, y_pred_pcr, labels=[0, 1])
    cm_baseline_df = pd.DataFrame(cm_baseline, index=[0, 1], columns=[0, 1])
    print("Baseline PCR confusion matrix (rows=true, cols=pred):")
    print(cm_baseline_df)

    baseline_rfs = RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    baseline_rfs.fit(X_train_rfs, y_train_rfs)
    y_pred_rfs = baseline_rfs.predict(X_test_rfs)
    baseline_rfs_mae = mean_absolute_error(y_test_rfs, y_pred_rfs)
    baseline_rfs_r2 = r2_score(y_test_rfs, y_pred_rfs)
    print(f"Baseline RFS (RF) MAE: {baseline_rfs_mae:.4f}")
    print(f"Baseline RFS (RF) R^2: {baseline_rfs_r2:.4f}")

    y_pred_const = np.full_like(y_test_rfs, fill_value=y_train_rfs.mean(), dtype=float)
    const_mae = mean_absolute_error(y_test_rfs, y_pred_const)
    print(f"Constant baseline RFS MAE (predict mean): {const_mae:.4f}")

    best_pcr_model = tune_pcr_models(X_train_pcr, y_train_pcr)
    tuned_pcr_preds = best_pcr_model.predict(X_test_pcr)
    tuned_pcr_bal_acc = balanced_accuracy_score(y_test_pcr, tuned_pcr_preds)
    print(f"Tuned PCR model balanced accuracy: {tuned_pcr_bal_acc:.4f}")
    cm_tuned = confusion_matrix(y_test_pcr, tuned_pcr_preds, labels=[0, 1])
    cm_tuned_df = pd.DataFrame(cm_tuned, index=[0, 1], columns=[0, 1])
    print("Tuned PCR confusion matrix (rows=true, cols=pred):")
    print(cm_tuned_df)

    best_rfs_model = tune_rfs_models(X_train_rfs, y_train_rfs)
    tuned_rfs_preds = best_rfs_model.predict(X_test_rfs)
    tuned_rfs_mae = mean_absolute_error(y_test_rfs, tuned_rfs_preds)
    tuned_rfs_r2 = r2_score(y_test_rfs, tuned_rfs_preds)
    print(f"Tuned RFS model MAE: {tuned_rfs_mae:.4f}")
    print(f"Tuned RFS model R^2: {tuned_rfs_r2:.4f}")

    print("\n=== Summary of model performance on hold-out test split ===")
    print(
        f"PCR balanced accuracy - baseline: {baseline_pcr_bal_acc:.4f} | tuned: {tuned_pcr_bal_acc:.4f}"
    )
    print(
        f"RFS MAE - baseline: {baseline_rfs_mae:.4f} | tuned: {tuned_rfs_mae:.4f}"
    )

    train_and_save_final_models(best_pcr_model, best_rfs_model, features_pcr, features_rfs)


if __name__ == "__main__":
    run_method_development()
