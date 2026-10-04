"""Employee attrition prediction dashboard."""

from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "employee_attrition.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
RANDOM_STATE = 42
IRRELEVANT_COLUMNS = {
    "employeenumber",
    "employeecount",
    "standardhours",
    "over18",
    "id",
    "employeeid",
}
POSITIVE_LABELS = {"yes", "true", "1", "leave", "left", "attrition"}

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def canonical_name(value: str) -> str:
    """Normalize column names to make common CSV variations easy to detect."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns: list[str], *names: str) -> str | None:
    normalized = {canonical_name(column): column for column in columns}
    for name in names:
        if canonical_name(name) in normalized:
            return normalized[canonical_name(name)]
    return None


def attrition_labels(series: pd.Series) -> tuple[pd.Series, str]:
    """Convert common attrition labels to 0=Stay and 1=Leave."""
    values = series.astype("string").str.strip().str.lower()
    observed = set(values.dropna().unique())
    if len(observed) != 2:
        raise ValueError(
            "The target column must contain exactly two classes, such as Yes/No or Leave/Stay."
        )
    positives = observed & POSITIVE_LABELS

    if positives:
        if len(positives) > 1:
            raise ValueError("The target contains multiple labels that could mean Leave.")
        positive = positives.pop()
        encoded = (values == positive).astype("int8")
        return encoded, positive
    # For non-standard binary labels, use the lexically last class as Leave.
    positive = sorted(observed)[-1]
    return (values == positive).astype("int8"), positive


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str, str]:
    """Find the target, remove identifiers, and create a binary attrition target."""
    if frame.empty:
        raise ValueError("The selected CSV has no rows.")
    target = find_column(list(frame.columns), "Attrition", "Left", "Leave", "Target")
    if target is None:
        raise ValueError(
            "Couldn't find a target column. Name the attrition column Attrition, "
            "Left, Leave, or Target."
        )

    cleaned = frame.dropna(subset=[target]).copy()
    y, positive_label = attrition_labels(cleaned[target])
    excluded = {
        column
        for column in cleaned.columns
        if canonical_name(column) in IRRELEVANT_COLUMNS
    }
    excluded.add(target)
    X = cleaned.drop(columns=list(excluded)).dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable feature columns remain after removing the target and IDs.")
    if y.nunique() != 2:
        raise ValueError("The target must include both Leave and Stay examples.")
    return X, y, target, positive_label


def build_pipeline(X: pd.DataFrame, estimator: object) -> Pipeline:
    """Build preprocessing and classifier as one leakage-safe estimator."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessing = ColumnTransformer(transformers=transformers)
    return Pipeline(
        steps=[("preprocessor", preprocessing), ("classifier", estimator)]
    )


@st.cache_resource(show_spinner="Training and evaluating both models...")
def train_models(frame: pd.DataFrame, target_column: str, test_fraction: float):
    X, y, _, _ = prepare_data(frame)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "At least two Leave and two Stay records are needed for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    results = {}
    for name, estimator in estimators.items():
        pipeline = build_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        results[name] = {
            "pipeline": pipeline,
            "prediction": prediction,
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "confusion": confusion_matrix(y_test, prediction, labels=[0, 1]),
            "report": classification_report(
                y_test,
                prediction,
                labels=[0, 1],
                target_names=["Stay", "Leave"],
                output_dict=True,
                zero_division=0,
            ),
        }
    # F1 balances finding likely leavers with avoiding excessive false alarms.
    best_name = max(results, key=lambda name: results[name]["f1"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(results[best_name]["pipeline"], MODEL_PATH)
    return results, best_name, y_test


def display_metric_cards(result: dict) -> None:
    columns = st.columns(4)
    for column, label, key in zip(
        columns,
        ("Accuracy", "Precision", "Recall", "F1-score"),
        ("accuracy", "precision", "recall", "f1"),
    ):
        column.metric(label, f"{result[key]:.1%}")


def plot_count(frame: pd.DataFrame, x: str, hue: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if hue:
        sns.countplot(data=frame, x=x, hue=hue, palette="deep", ax=ax)
    else:
        sns.countplot(data=frame, x=x, hue=x, palette="deep", legend=False, ax=ax)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel("Employees")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_numeric(frame: pd.DataFrame, x: str, target: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(
        data=frame,
        x=x,
        hue=target,
        bins=20,
        kde=True,
        palette={"Stay": "#1f9d8b", "Leave": "#f07b63"},
        ax=ax,
    )
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel("Employees")
    fig.tight_layout()
    return fig


def plot_box(frame: pd.DataFrame, x: str, target: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(
        data=frame,
        x=target,
        y=x,
        palette={"Stay": "#1f9d8b", "Leave": "#f07b63"},
        hue=target,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Attrition")
    ax.set_ylabel(x.replace("_", " "))
    fig.tight_layout()
    return fig


def render_eda(frame: pd.DataFrame, target: str) -> None:
    columns = list(frame.columns)
    role = find_column(columns, "JobRole")
    overtime = find_column(columns, "OverTime", "Overtime")
    satisfaction = find_column(columns, "JobSatisfaction")
    age = find_column(columns, "Age")
    income = find_column(columns, "MonthlyIncome")
    tenure = find_column(columns, "YearsAtCompany")
    target_for_plot = "Attrition Status"
    plot_frame = frame.copy()
    plot_frame[target_for_plot] = plot_frame[target].astype(str).str.strip().str.title()

    st.subheader("Interactive exploration")
    categorical_filters = [
        column
        for column in (
            find_column(columns, "Department"),
            find_column(columns, "Gender"),
            overtime,
        )
        if column is not None
    ]
    filter_columns = st.columns(max(1, len(categorical_filters)))
    for container, column in zip(filter_columns, categorical_filters):
        choices = sorted(plot_frame[column].dropna().astype(str).unique().tolist())
        selected = container.multiselect(
            f"Filter {column.replace('_', ' ')}",
            choices,
            placeholder="All values",
            key=f"eda-filter-{column}",
        )
        if selected:
            plot_frame = plot_frame[plot_frame[column].astype(str).isin(selected)]

    groupable = [column for column in (role, overtime, satisfaction) if column]
    if groupable and not plot_frame.empty:
        selected_feature = st.selectbox(
            "Compare Leave rate by",
            groupable,
            format_func=lambda value: value.replace("_", " "),
            key="eda-rate-feature",
        )
        grouped = (
            plot_frame.groupby(selected_feature, dropna=False)[target_for_plot]
            .apply(lambda values: values.eq("Leave").mean() * 100)
            .sort_values(ascending=False)
            .rename("Leave rate (%)")
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.barplot(
            data=grouped,
            x=selected_feature,
            y="Leave rate (%)",
            color="#f07b63",
            ax=ax,
        )
        ax.set_ylabel("Employees leaving (%)")
        ax.set_xlabel(selected_feature.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, max(10, float(grouped["Leave rate (%)"].max()) * 1.15))
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    elif plot_frame.empty:
        st.warning("No employees match these filters. Clear one or more filters to see charts.")
        return

    st.subheader("Attrition patterns")
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_count(plot_frame, target_for_plot), clear_figure=True)
        if role:
            st.pyplot(plot_count(plot_frame, role, target_for_plot), clear_figure=True)
        if income:
            st.pyplot(plot_box(plot_frame, income, target_for_plot), clear_figure=True)
    with right:
        if age:
            st.pyplot(plot_numeric(plot_frame, age, target_for_plot), clear_figure=True)
        if overtime:
            st.pyplot(plot_count(plot_frame, overtime, target_for_plot), clear_figure=True)
        if satisfaction:
            st.pyplot(plot_count(plot_frame, satisfaction, target_for_plot), clear_figure=True)
        if tenure:
            st.pyplot(plot_box(plot_frame, tenure, target_for_plot), clear_figure=True)


def render_prediction_form(X: pd.DataFrame, pipeline: Pipeline) -> None:
    st.write(
        "Build an employee profile below. Inputs adapt automatically to the selected dataset."
    )
    with st.form("employee-prediction-form"):
        values = {}
        columns = st.columns(2)
        for index, column in enumerate(X.columns):
            container = columns[index % 2]
            series = X[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if numeric.empty:
                    values[column] = container.number_input(column, value=0.0)
                    continue
                minimum, maximum = float(numeric.min()), float(numeric.max())
                median = float(numeric.median())
                if minimum < maximum and (maximum - minimum) <= 500:
                    values[column] = container.slider(
                        column,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(median, minimum), maximum),
                    )
                else:
                    values[column] = container.number_input(
                        column, min_value=minimum, max_value=maximum, value=median
                    )
            else:
                choices = series.dropna().astype(str).value_counts().index.tolist()
                if not choices:
                    choices = ["Unknown"]
                values[column] = container.selectbox(column, choices)
        submitted = st.form_submit_button(
            "✨ Predict attrition", type="primary", width="stretch"
        )

    if submitted:
        employee = pd.DataFrame([values], columns=X.columns)
        prediction = int(pipeline.predict(employee)[0])
        probability = float(pipeline.predict_proba(employee)[0][1])
        outcome, explanation, color = (
            ("LIKELY TO LEAVE", "🔴", "leave")
            if prediction
            else ("LIKELY TO STAY", "🟢", "stay")
        )
        st.markdown(
            f"""
            <div class="prediction-card {color}">
                <div class="prediction-kicker">MODEL ASSESSMENT</div>
                <div class="prediction-title">{explanation} Employee is {outcome}</div>
                <div class="prediction-copy">Estimated probability of leaving</div>
                <div class="prediction-percent">{probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        confidence = "Higher" if probability >= 0.65 else "Moderate" if probability >= 0.35 else "Lower"
        st.progress(probability, text=f"{confidence} estimated leave risk")
        st.caption(
            "Probability is a model estimate for demonstration, not a certainty or "
            "a substitute for fair, human-led employment decisions."
        )


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px;}
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff 0%, #f2f7fa 100%);
            border: 1px solid #e6edf1; padding: 1rem 1.1rem; border-radius: 16px;
            box-shadow: 0 5px 18px rgba(21, 51, 68, .045);
        }
        [data-testid="stMetricLabel"] {color: #627382;}
        [data-testid="stMetricValue"] {font-size: clamp(1rem, 2vw, 1.8rem);}
        .hero {
            padding: 2.2rem 2.4rem; border-radius: 24px; margin: .5rem 0 1.4rem;
            color: white; background: linear-gradient(118deg, #123a4a 0%, #137d78 58%, #36a889 100%);
            box-shadow: 0 16px 40px rgba(19, 93, 92, .18);
        }
        .hero-eyebrow {text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; opacity: .78;}
        .hero h1 {color: white; font-size: clamp(2rem, 4vw, 3.1rem); margin: .45rem 0;}
        .hero p {font-size: 1.05rem; opacity: .9; max-width: 760px; margin-bottom: 0;}
        .prediction-card {
            padding: 1.5rem 1.7rem; margin: 1.3rem 0 1rem; border-radius: 20px;
            border: 1px solid #dce7e9; background: linear-gradient(120deg, #f5fbf8, #fff);
        }
        .prediction-card.leave {background: linear-gradient(120deg, #fff5f1, #fff); border-color: #f5d8cd;}
        .prediction-kicker {font-size: .74rem; letter-spacing: .14em; color: #637783; font-weight: 700;}
        .prediction-title {font-size: 1.55rem; font-weight: 750; margin: .35rem 0;}
        .prediction-copy {color: #647783; margin-top: .8rem;}
        .prediction-percent {font-size: 2.3rem; font-weight: 750; color: #167e75;}
        .prediction-card.leave .prediction-percent {color: #cb5947;}
        div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
            border-radius: 11px; font-weight: 650;
        }
        h2, h3 {letter-spacing: -.02em;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## 🧭 People Insights")
        st.caption("EMPLOYEE ATTRITION ANALYTICS")
        page = st.radio(
            "Dashboard section",
            [
                "Home / Overview",
                "Dataset",
                "EDA & Visualizations",
                "Feature Selection",
                "Model Training",
                "Model Comparison",
                "Attrition Prediction",
            ],
            label_visibility="collapsed",
            captions=[
                "Key workforce signals",
                "Explore and inspect records",
                "Discover attrition patterns",
                "Understand model drivers",
                "Metrics and validation",
                "Compare model performance",
                "Estimate an employee's risk",
            ],
        )
        st.divider()
        st.subheader("Choose dataset")
        uploaded = st.file_uploader("Upload an employee CSV", type=["csv"])
        test_fraction = st.slider(
            "Test-set size", min_value=0.15, max_value=0.40, value=0.20, step=0.05
        )
        st.caption("No upload? The included sample CSV is used.")

    try:
        if uploaded is not None:
            frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            source_name = uploaded.name
        else:
            frame = pd.read_csv(DEFAULT_DATA)
            source_name = "Bundled sample dataset"
        if frame.columns.duplicated().any():
            raise ValueError("The CSV contains duplicate column names. Please make them unique.")
        X, y, target, positive_label = prepare_data(frame)
        if y.value_counts().min() < 2:
            raise ValueError("The dataset needs at least two records in each attrition class.")
        frame_for_eda = frame.dropna(subset=[target]).copy()
        frame_for_eda[target] = y.map({0: "Stay", 1: "Leave"}).values
        results, best_name, y_test = train_models(frame, target, test_fraction)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Unable to prepare this dataset: {error}")
        st.stop()

    best_result = results[best_name]
    st.sidebar.success(f"{len(frame):,} employees · {len(X.columns)} features")
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">People analytics · machine learning</div>
            <h1>Employee Attrition Intelligence</h1>
            <p>Explore the signals behind workforce changes, compare predictive models,
            and turn employee data into a clearer view of retention risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Home / Overview":
        st.subheader("Workforce at a glance")
        st.caption(
            f"{source_name} · Leave means {positive_label.title()} in **{target}** · "
            "Models evaluated on a stratified holdout set"
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Employees", f"{len(frame):,}")
        metric_columns[1].metric("Leave rate", f"{y.mean():.1%}")
        metric_columns[2].metric("Stay rate", f"{1-y.mean():.1%}")
        metric_columns[3].metric("Best model", best_name, f"F1 {best_result['f1']:.1%}")
        st.progress(y.mean(), text="Share of employees marked Leave")

        overview_left, overview_right = st.columns([1.35, 1])
        with overview_left:
            st.subheader("Retention snapshot")
            st.pyplot(plot_count(frame_for_eda, "Attrition"), clear_figure=True)
        with overview_right:
            st.subheader("Model performance")
            st.metric("Selected model", best_name)
            st.metric("Accuracy", f"{best_result['accuracy']:.1%}")
            st.metric("Leave-class F1", f"{best_result['f1']:.1%}")
            st.info(
                "The best model is selected by Leave-class F1, balancing the ability "
                "to find leavers against false alarms."
            )
        st.subheader("Your analytics workflow")
        journey = st.columns(4)
        steps = [
            ("01 · Explore", "Inspect workforce data and attrition patterns."),
            ("02 · Compare", "Evaluate two models on stratified test data."),
            ("03 · Explain", "See the strongest signals identified by the model."),
            ("04 · Predict", "Try an employee profile and review its risk estimate."),
        ]
        for column, (title, description) in zip(journey, steps):
            column.markdown(f"**{title}**")
            column.caption(description)
        st.caption(
            f"Dataset: {len(X.columns)} candidate features · Target column: {target} · "
            f"Train/test split: {1-test_fraction:.0%}/{test_fraction:.0%}"
        )

    elif page == "Dataset":
        st.header("Dataset")
        st.caption(f"Source: {source_name}")
        first, second, third = st.columns(3)
        first.metric("Rows", f"{frame.shape[0]:,}")
        second.metric("Columns", frame.shape[1])
        third.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
        st.subheader("Preview")
        preview_rows = st.slider("Rows to preview", 5, 50, 15, 5)
        st.dataframe(frame.head(preview_rows), width="stretch")
        st.subheader("Column information")
        summary = pd.DataFrame(
            {
                "Column": frame.columns,
                "Data type": frame.dtypes.astype(str).values,
                "Missing": frame.isna().sum().values,
                "Unique values": frame.nunique(dropna=True).values,
            }
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(
            f"Target: {target}. Identifier columns are excluded from training; "
            "remaining missing values are imputed within each training pipeline."
        )

    elif page == "EDA & Visualizations":
        st.header("Exploratory data analysis")
        st.write("Counts and distributions are grouped by the dataset's Leave/Stay target.")
        render_eda(frame_for_eda, target)

    elif page == "Feature Selection":
        st.header("Important features")
        st.write(
            "Random Forest impurity importance is calculated after one-hot encoding. "
            "Categories are shown as individual encoded features."
        )
        forest = results["Random Forest"]["pipeline"]
        preprocessor = forest.named_steps["preprocessor"]
        importances = forest.named_steps["classifier"].feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        importance_frame = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
        )
        top_n = st.slider(
            "Number of top features",
            min_value=1,
            max_value=min(30, len(importance_frame)),
            value=min(15, len(importance_frame)),
        )
        importance_frame = importance_frame.head(top_n).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importance_frame["Feature"], importance_frame["Importance"], color="#4878a8")
        ax.set_xlabel("Random Forest importance")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.dataframe(
            importance_frame.sort_values("Importance", ascending=False),
            hide_index=True,
            width="stretch",
        )

    elif page == "Model Training":
        st.header("Model training and evaluation")
        st.write(
            f"Stratified split: {1 - test_fraction:.0%} training / "
            f"{test_fraction:.0%} testing. Missing-value imputers and one-hot encoders "
            "are fitted only on training rows inside each Scikit-learn pipeline."
        )
        st.info(f"Best model by Leave-class F1: **{best_name}**")
        display_metric_cards(best_result)
        st.subheader(f"{best_name} confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            best_result["confusion"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Stay", "Leave"],
            yticklabels=["Stay", "Leave"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, clear_figure=True)
        st.subheader("Classification report")
        report = pd.DataFrame(best_result["report"]).T
        st.dataframe(report.style.format(precision=3), width="stretch")
        st.caption(f"Best model pipeline saved to `{MODEL_PATH.relative_to(ROOT)}`.")

    elif page == "Model Comparison":
        st.header("Model comparison")
        comparison = pd.DataFrame(
            {
                name: {
                    "Accuracy": result["accuracy"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1-score": result["f1"],
                }
                for name, result in results.items()
            }
        ).T
        primary_metric = st.selectbox(
            "Highlight comparison metric",
            ["F1-score", "Recall", "Precision", "Accuracy"],
        )
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        comparison[[primary_metric]].plot(
            kind="bar",
            ax=ax,
            ylim=(0, 1),
            rot=0,
            color=["#1f9d8b"],
            legend=False,
        )
        ax.set_ylabel("Score")
        ax.set_title(f"{primary_metric} on held-out test data")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.success(f"Selected **{best_name}** based on the highest Leave-class F1-score.")

    else:
        st.header("Employee attrition prediction")
        st.write(
            f"The selected model is **{best_name}**. Enter the employee's details "
            "and submit the form to estimate the probability of leaving."
        )
        st.info(
            f"Model: {best_name} · Test-set F1: {best_result['f1']:.1%} · "
            f"Built from {len(X.columns)} employee features"
        )
        render_prediction_form(X, best_result["pipeline"])


if __name__ == "__main__":
    main()
