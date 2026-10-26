# Salary Prediction Using Machine Learning

## Project Overview

This project explores salary prediction using supervised machine learning regression algorithms. It uses job-related attributes such as job type, education level, field of study, industry, years of experience, and distance from a metropolis to estimate salary.

The workflow includes data loading, quality checks, merging salary labels with job attributes, exploratory data analysis (EDA), categorical encoding, train-test splitting, model training, and regression evaluation.

## Problem Statement

Salary varies with factors such as role seniority, education, industry, professional experience, and location. The objective is to build and compare regression models that estimate salary from these job characteristics.

## Dataset

The notebook loads three CSV files:

- `test_dataset1.csv` — job attributes intended for testing/prediction.
- `train_dataset2.csv` — job attributes used for model development.
- `train_salaries3.csv` — salary labels associated with `jobId`.

The notebook reports one million rows in each file. The training features and salary labels are merged using `jobId`, producing a dataset with **1,000,000 rows and 9 columns**.

### Features

| Column | Description |
|---|---|
| `jobId` | Job record identifier |
| `companyId` | Company identifier |
| `jobType` | Job role or seniority, such as `JUNIOR`, `MANAGER`, `CFO`, or `CEO` |
| `degree` | Education level, such as `HIGH_SCHOOL`, `BACHELORS`, `MASTERS`, or `DOCTORAL` |
| `major` | Field of study |
| `industry` | Industry category |
| `yearsExperience` | Years of professional experience |
| `milesFromMetropolis` | Distance from the metropolis, in miles |
| `salary` | Target variable to predict |

The notebook does not specify the salary unit.

## Tools and Libraries

- Python
- Pandas and NumPy — data handling and numerical operations
- Matplotlib and Seaborn — visualization and EDA
- Scikit-learn — preprocessing, model training, model selection, and evaluation
- XGBoost — attempted as an additional model, but the notebook reports that the package was not installed

## Project Workflow

1. **Data loading** — load job attributes and salary labels from CSV files.
2. **Data inspection** — inspect sample rows, dimensions, data types, and summary statistics.
3. **Data quality checks** — check missing values and duplicate records.
4. **Data integration** — merge job attributes with salary labels using `jobId`.
5. **Feature preparation** — remove identifier columns (`jobId` and `companyId`) from the modeling dataframe and one-hot encode categorical features using `pd.get_dummies()`.
6. **Exploratory Data Analysis** — visualize experience, salary, education, major, job type, industry, and distance; inspect correlations among numeric columns.
7. **Train-test split** — use `train_test_split` with `train_size=0.20` and `random_state=42` in the reported baseline experiments.
8. **Model training** — fit and compare regression algorithms.
9. **Evaluation** — assess predictions using R² and RMSE.
10. **Hyperparameter tuning** — experiment with `RandomizedSearchCV` for selected estimators.

## Data Quality and EDA Notes

The notebook reports the following for the merged modeling dataset:

- **Rows:** 1,000,000
- **Columns:** 9 before dropping identifiers
- **Missing values:** 0 reported in the inspected training-feature and salary-label files
- **Duplicate rows:** 0 reported in the inspected feature and salary-label data
- **Mean years of experience:** approximately 11.99
- **Mean distance from metropolis:** approximately 49.53 miles
- **Mean salary:** approximately 116.06 (unit not specified)

Reported correlations among numeric variables:

- `yearsExperience` and `salary`: approximately **0.375**
- `milesFromMetropolis` and `salary`: approximately **-0.298**
- `yearsExperience` and `milesFromMetropolis`: approximately **0.001**

These are associations in this dataset and do not establish causation.

## Models Evaluated

- Linear Regression
- Decision Tree Regressor
- K-Neighbors Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Support Vector Regressor (hyperparameter-tuning attempt)
- XGBoost Regressor (attempted, but unavailable in the recorded environment)

## Baseline Results Reported in the Notebook

| Model | R² Score | RMSE |
|---|---:|---:|
| Gradient Boosting Regressor | 0.7495 | 19.3725 |
| Linear Regression | 0.7435 | 19.6041 |
| Random Forest Regressor | 0.7137 | 20.7120 |
| AdaBoost Regressor | 0.5687 | 25.4212 |
| Decision Tree Regressor | 0.4861 | 27.7482 |
| K-Neighbors Regressor | 0.4509 | 28.6837 |

Based on these recorded baseline results, **Gradient Boosting Regressor** has the highest reported R² and lowest reported RMSE among the listed baseline comparisons.

**Evaluation caveat:** The notebook uses different training subsets for some models (including Decision Tree and KNN), and the hyperparameter-tuning section contains inconsistent variable usage and scoring calls. Treat these figures as the results recorded in the notebook, not as a fully controlled benchmark. For a rigorous comparison, use the same train-test split and preprocessing for every model, and calculate metrics consistently with `r2_score(y_test, y_pred)` and `sqrt(mean_squared_error(y_test, y_pred))`.

## How to Run

1. Clone or download this repository.
2. Place the CSV files in the project directory, or update the file paths in the notebook.
3. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the notebook and run the cells in order.

The source code currently uses a Windows-specific file path. Replace it with a path that works on your machine, for example:

```python
import pandas as pd

df1 = pd.read_csv("test_dataset1.csv")
df2 = pd.read_csv("train_dataset2.csv")
df3 = pd.read_csv("train_salaries3.csv")
```

To run the XGBoost experiment, install the optional dependency:

```bash
pip install xgboost
```

## Suggested Repository Structure

```text
Salary-Prediction-ML/
├── README.md
├── notebook.ipynb
├── test_dataset1.csv
├── train_dataset2.csv
└── train_salaries3.csv
```

Only include dataset files in a public repository if their license and sharing terms allow it.

## Potential Improvements

- Use a reproducible preprocessing-and-modeling pipeline.
- Use the same train-test split for all model comparisons and consider cross-validation.
- Fit encoders and scalers on training data only, then transform test data.
- Verify that `jobId` values match as expected before merging the feature and salary files.
- Investigate zero or unusually high salary values before deciding whether they are valid observations or data-quality issues.
- Tune models with correctly matched estimators, parameter distributions, and evaluation metrics.
- Evaluate on the separate test dataset only after selecting the final model and preprocessing steps.
- Save the trained model and preprocessing pipeline for reuse.

## Disclaimer

This is an educational machine learning project. Model outputs are estimates based on the available data and should not be used as the sole basis for real-world compensation decisions.

## Author

**Rushikesh Palade**

- GitHub: `https://github.com/rushikeshpalde02-zip`
- LinkedIn: `https://www.linkedin.com/in/RushikeshPalade`
