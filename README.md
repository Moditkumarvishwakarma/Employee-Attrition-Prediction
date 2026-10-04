# Employee Attrition Prediction

A beginner-friendly Streamlit application that demonstrates an end-to-end
machine-learning workflow for predicting whether an employee is likely to
**Leave** or **Stay**. It includes an IBM HR Analytics-style sample dataset and
also accepts a user-provided CSV.

## Project objective

Explore employee and job-related factors associated with attrition, compare two
classification models, and interactively estimate an employee's probability of
leaving. This is an educational demonstration; it should not be used as the
sole basis for employment decisions.

## Features

- Upload an employee CSV or use `data/employee_attrition.csv`.
- Review dataset dimensions, a preview, column types, unique values, and missing
  data.
- Explore attrition by age, job role, overtime, job satisfaction, monthly
  income, and years at company (charts are shown when the relevant columns exist),
  with interactive department, gender, and overtime filters.
- Compare leave rates by job role, overtime, or job satisfaction, and choose the
  number of top features to display.
- Automatically identify categorical and numeric predictors.
- Impute missing numeric values with the median and categorical values with
  the most frequent value; one-hot encode categorical values.
- Compare Logistic Regression and Random Forest using a stratified test split.
- Report accuracy, precision, recall, F1-score, confusion matrix, and
  classification report.
- Rank one-hot encoded feature importances from the Random Forest model.
- Predict Leave/Stay and show the estimated probability for an employee entered
  through a data-driven form.
- Use a responsive, light-themed dashboard with an at-a-glance workforce
  overview and model performance summary.

## Project structure

```text
employee-attrition-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── employee_attrition.csv
├── models/
│   └── model.pkl             # created when the app trains the selected model
└── images/                   # optional presentation screenshots
```

## Requirements and installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run locally

From the project directory:

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in a
browser. Use the sidebar to switch dashboard sections, upload a CSV, or adjust
the test-set size.

## Dataset format

The target column should be named `Attrition`, `Left`, `Leave`, or `Target` and
contain two classes. `Yes`/`No` is recommended (`Yes` means Leave; `No` means
Stay). Binary labels such as `Leave`/`Stay` are also supported. The application
recognizes common employee identifier columns such as `EmployeeNumber`,
`EmployeeCount`, `StandardHours`, and `Over18`, and excludes them from model
features. All other columns are treated as candidate employee features.

Rows without a target are excluded. A stratified split requires at least two
records for each target class. Missing predictor values are imputed as part of
the model pipeline.

## Workflow and implementation

1. **Data collection:** load the bundled CSV or upload another CSV from the
   sidebar.
2. **Preprocessing:** locate the binary target, remove known identifier fields,
   and separate predictors and target. The positive target class maps to
   `Leave`; the other maps to `Stay`.
3. **Exploration:** inspect the data and plot attrition patterns where
   recognized columns are available.
4. **Train/test split:** split with stratification and a fixed random seed.
5. **Modeling:** fit Logistic Regression and Random Forest pipelines. Each
   pipeline learns imputers and one-hot encoding from training data only; unseen
   categories are safely ignored at prediction time.
6. **Evaluation and selection:** compare held-out accuracy, precision, recall,
   and F1-score. Select the model with the highest Leave-class F1-score, which
   balances finding leavers and avoiding excessive false alarms.
7. **Feature selection:** display Random Forest feature importances, including
   one-hot encoded category levels.
8. **Prediction:** enter details through a form generated from the selected
   dataset's feature columns. The result includes the predicted class and
   probability of Leave.

The best pipeline is saved to `models/model.pkl` after training. It is
automatically regenerated for the active dataset and test split.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Notes

- The included CSV is a synthetic, reproducible-style demonstration dataset;
  results from it are not intended to represent any real employer or workforce.
- A dataset with a different target name or unusual label values should be
  checked before use; the app reports a clear error when it cannot identify a
  binary target.
- Model quality depends on the quality, representativeness, and size of the
  input data. Attrition predictions can reflect bias in historical data.
