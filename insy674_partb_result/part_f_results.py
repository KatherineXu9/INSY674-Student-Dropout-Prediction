from pathlib import Path
import hashlib
import json
import platform
import zipfile

import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier


# 1. Load the original data and frozen model configuration.

SEED = 42
ROOT = Path.cwd()
OUTPUT = ROOT / "part_f_results"
OUTPUT.mkdir(exist_ok=True)

CONFIG_PATH = (
    ROOT
    / "insy674_partb_result"
    / "part_e_results"
    / "model_selection_for_F.json"
)

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

zip_paths = sorted(ROOT.glob("predict*students*dropout*.zip"))

if zip_paths:
    with zipfile.ZipFile(zip_paths[0]) as archive:
        csv_name = next(
            name
            for name in archive.namelist()
            if Path(name).name.lower() == "data.csv"
        )
        with archive.open(csv_name) as file:
            df = pd.read_csv(file, sep=";", encoding="utf-8-sig")
else:
    df = pd.read_csv(ROOT / "data.csv", sep=";", encoding="utf-8-sig")

df.columns = df.columns.str.strip().str.replace(
    "\ufeff", "", regex=False
)

assert df.shape == (4424, 37)
assert config["chosen_candidate"] == "Soft Voting Ensemble"
assert config["selection_feature_set"] == "First Semester"
assert config["train_test_split"]["random_state"] == SEED
assert config["train_test_split"]["test_size"] == 0.2

features = config["feature_columns"]

assert len(features) == 24
assert "Target" not in features
assert not any("2nd sem" in column for column in features)

financial_columns = {
    "Scholarship holder",
    "Debtor",
    "Tuition fees up to date",
}
assert not financial_columns.intersection(features)


# 2. Reproduce the original train/test split.

X = df.drop(columns="Target")
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

assert len(X_train) == 3539
assert len(X_test) == 885
assert X_train.index.intersection(X_test.index).empty

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

class_names = encoder.classes_
class_codes = np.arange(len(class_names))
dropout_code = int(encoder.transform(["Dropout"])[0])

assert list(class_names) == ["Dropout", "Enrolled", "Graduate"]

X_train_selected = X_train[features]
X_test_selected = X_test[features]


# 3. Reproduce the original feature engineering and preprocessing.

def add_first_semester_features(data):
    data = data.copy()

    enrolled = data["Curricular units 1st sem (enrolled)"]
    denominator = enrolled.replace(0, np.nan)

    data["First semester pass rate"] = (
        data["Curricular units 1st sem (approved)"] / denominator
    )
    data["Evaluations per enrolled course"] = (
        data["Curricular units 1st sem (evaluations)"] / denominator
    )
    data["No first-semester enrollment"] = (
        enrolled == 0
    ).astype(int)

    return data


categorical_columns = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

selected_categorical = [
    column for column in features
    if column in categorical_columns
]

selected_numeric = [
    column for column in features
    if column not in categorical_columns
] + [
    "First semester pass rate",
    "Evaluations per enrolled course",
    "No first-semester enrollment",
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = Pipeline([
    (
        "feature_engineering",
        FunctionTransformer(
            add_first_semester_features,
            validate=False,
        ),
    ),
    (
        "preprocessing",
        ColumnTransformer([
            ("numeric", numeric_pipeline, selected_numeric),
            (
                "categorical",
                categorical_pipeline,
                selected_categorical,
            ),
        ]),
    ),
])


# 4. Rebuild the selected ensemble using saved hyperparameters.
# Do not perform additional tuning using the test set.

estimators = {
    "XGBoost": XGBClassifier(
        n_estimators=150,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=SEED,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=SEED,
    ),
}

tuned_pipelines = {}

for name, estimator in estimators.items():
    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", estimator),
    ])

    pipeline.set_params(**config["tuned_parameters"][name])
    tuned_pipelines[name] = pipeline

ensemble = VotingClassifier(
    estimators=[
        ("xgb", tuned_pipelines["XGBoost"]),
        ("gb", tuned_pipelines["Gradient Boosting"]),
        ("lr", tuned_pipelines["Logistic Regression"]),
    ],
    voting="soft",
    n_jobs=1,
)

logistic_baseline = Pipeline([
    ("preprocessing", clone(preprocessor)),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
    ),
])

models = {
    "Majority-class baseline": DummyClassifier(
        strategy="most_frequent",
    ),
    "Logistic Regression baseline": logistic_baseline,
    "Selected soft-voting ensemble": ensemble,
}


# 5. Fit on training records and evaluate held-out test records.

comparison_rows = []
ensemble_predictions = None

for name, model in models.items():
    print(f"\nFitting: {name}")

    # All learned preprocessing is fitted on training data only.
    model.fit(X_train_selected, y_train_encoded)
    predictions = model.predict(X_test_selected)

    comparison_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test_encoded, predictions
        ),
        "Macro F1": f1_score(
            y_test_encoded,
            predictions,
            labels=class_codes,
            average="macro",
            zero_division=0,
        ),
        "Dropout precision": precision_score(
            y_test_encoded,
            predictions,
            labels=[dropout_code],
            average="macro",
            zero_division=0,
        ),
        "Dropout recall": recall_score(
            y_test_encoded,
            predictions,
            labels=[dropout_code],
            average="macro",
            zero_division=0,
        ),
        "Dropout F1": f1_score(
            y_test_encoded,
            predictions,
            labels=[dropout_code],
            average="macro",
            zero_division=0,
        ),
    })

    prediction_table = pd.DataFrame({
        "row_index": X_test.index,
        "Actual": encoder.inverse_transform(y_test_encoded),
        "Predicted": encoder.inverse_transform(predictions),
    })

    filename = {
        "Majority-class baseline": "majority_test_predictions.csv",
        "Logistic Regression baseline": "logistic_test_predictions.csv",
        "Selected soft-voting ensemble": "test_predictions.csv",
    }[name]

    prediction_table.to_csv(OUTPUT / filename, index=False)

    if name == "Selected soft-voting ensemble":
        ensemble_predictions = predictions

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(
    OUTPUT / "test_model_comparison.csv",
    index=False,
)

print("\nHELD-OUT TEST RESULTS")
print(comparison.round(4).to_string(index=False))


# 6. Save the final ensemble's confusion matrix and class metrics.

cm = confusion_matrix(
    y_test_encoded,
    ensemble_predictions,
    labels=class_codes,
)

cm_table = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)
cm_table.index.name = "Actual"
cm_table.columns.name = "Predicted"

cm_table.to_csv(OUTPUT / "test_confusion_matrix.csv")

report = classification_report(
    y_test_encoded,
    ensemble_predictions,
    labels=class_codes,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_table = pd.DataFrame(report).T
report_table.to_csv(OUTPUT / "test_classification_report.csv")

print("\nCONFUSION MATRIX")
print(cm_table)

print("\nCLASS-LEVEL PERFORMANCE")
print(report_table.round(4))


# 7. Calculate error counts and save individual misclassifications.

prediction_table = pd.DataFrame({
    "row_index": X_test.index,
    "Actual": encoder.inverse_transform(y_test_encoded),
    "Predicted": encoder.inverse_transform(ensemble_predictions),
})

errors = prediction_table[
    prediction_table["Actual"] != prediction_table["Predicted"]
].copy()

errors.to_csv(OUTPUT / "test_misclassified_students.csv", index=False)

error_types = (
    errors.groupby(["Actual", "Predicted"])
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)

error_types.to_csv(OUTPUT / "test_error_types.csv", index=False)

true_positive = int(cm[dropout_code, dropout_code])
false_negative = int(cm[dropout_code, :].sum() - true_positive)
false_positive = int(cm[:, dropout_code].sum() - true_positive)

error_summary = {
    "correctly_identified_dropout": true_positive,
    "missed_dropout": false_negative,
    "false_dropout_alerts": false_positive,
    "actual_dropout_total": int(cm[dropout_code, :].sum()),
    "predicted_dropout_total": int(cm[:, dropout_code].sum()),
}

(OUTPUT / "test_error_summary.json").write_text(
    json.dumps(error_summary, indent=2),
    encoding="utf-8",
)

print("\nERROR TYPES")
print(error_types.to_string(index=False))
print("\nDROPOUT ERROR SUMMARY")
print(json.dumps(error_summary, indent=2))


# 8. Verify the calculations and record reproducibility information.

assert cm.sum() == 885
assert len(prediction_table) == 885
assert prediction_table["row_index"].is_unique

ensemble_accuracy = accuracy_score(
    y_test_encoded, ensemble_predictions
)
ensemble_macro_f1 = f1_score(
    y_test_encoded, ensemble_predictions, average="macro"
)

assert np.isclose(
    np.trace(cm) / cm.sum(),
    ensemble_accuracy,
)

assert np.isclose(
    report_table.loc[class_names, "f1-score"].mean(),
    ensemble_macro_f1,
)

pd.DataFrame({
    "row_index": X_train.index,
    "split": "train",
}).to_csv(OUTPUT / "train_row_indices.csv", index=False)

(OUTPUT / "model_selection_used.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

metadata = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "random_state": SEED,
    "training_records": len(X_train),
    "test_records": len(X_test),
    "test_class_counts": y_test.value_counts().to_dict(),
    "model_config_sha256": hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest(),
    "dataset_sha256": hashlib.sha256(
        df.to_csv(index=False).encode("utf-8")
    ).hexdigest(),
    "note": (
        "Saved model configuration used without test-set tuning. "
        "Results may differ across software versions."
    ),
}

(OUTPUT / "evaluation_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print(f"\nAll evaluation outputs saved to: {OUTPUT.resolve()}")