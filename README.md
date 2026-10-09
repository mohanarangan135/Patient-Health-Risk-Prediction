# Patient Health Risk Prediction Using Machine Learning in R

## Project Overview

Patient Health Risk Prediction is a machine learning and statistical analysis project developed using R. It analyzes patient health indicators, lifestyle habits, medical history, and healthcare expenses to classify patients into **High Risk** and **Low Risk** categories.

The project implements K-Nearest Neighbors (KNN), Decision Tree, and Logistic Regression classification models. It also uses Simple Linear Regression, Multiple Linear Regression, and Stepwise Regression to predict blood pressure and compare model performance.

The dataset contains simulated academic patient data and is intended for educational and research purposes, not real-world medical diagnosis.

## Objectives

- Analyze patient health and lifestyle data.
- Handle missing values and duplicate records.
- Explore data distributions, outliers, and correlations.
- Predict patient health risk using machine learning algorithms.
- Predict blood pressure using regression models.
- Compare model performance using evaluation metrics and visualizations.
- Apply trained models to example patient records.

## Technologies Used

- **Programming Language:** R
- **Data Analysis:** Base R
- **Machine Learning:** KNN, Decision Tree, Logistic Regression
- **Regression:** Simple Linear, Multiple Linear, Stepwise Regression
- **Data Visualization:** ggplot2
- **Model Evaluation:** caret, pROC
- **Decision Tree Visualization:** rpart, rpart.plot

## Dataset Details

- Dataset: `patient_health_risk_dataset.csv`
- Original dataset size: 3,000 rows and 23 columns
- Cleaned dataset: 2,985 records after duplicate removal
- Target variable: `health_risk`
- Target classes: `High_Risk` and `Low_Risk`

### Features

The dataset includes age, gender, BMI, blood pressure, heart rate, glucose level, cholesterol, oxygen saturation, waist circumference, smoking status, alcohol consumption, physical activity, sleep hours, family history, medication count, hospital visits, stress level, exercise frequency, diet quality, chronic conditions, and healthcare expenses.

## Project Workflow

1. Load the dataset and required R libraries.
2. Inspect the dataset structure and summary statistics.
3. Identify and handle missing values.
4. Remove duplicate records.
5. Visualize health risk distribution and healthcare expenses.
6. Prepare categorical and numerical variables.
7. Split the dataset into training and testing sets.
8. Train and tune KNN and Decision Tree models.
9. Train Logistic Regression and blood pressure regression models.
10. Evaluate and compare the models.
11. Predict outcomes for example new patient records.

## Machine Learning Models

### 1. K-Nearest Neighbors (KNN)

KNN classifies a patient based on the classes of the nearest observations. The input features are encoded and standardized before training.

- Best K reported in the presentation: 15
- Test accuracy: 80.03%
- ROC-AUC: 0.8608

### 2. Decision Tree

The Decision Tree classifies patients using interpretable decision rules. The tree is pruned to reduce overfitting.

- Test accuracy: 80.37%
- Recall: 53.51%
- F1-score: 62.86%
- ROC-AUC: 0.7789

### 3. Logistic Regression

Logistic Regression estimates the probability of a patient belonging to the High Risk category.

- Test accuracy: 87.08%
- Recall: 73.51%
- ROC-AUC: 0.9156

### 4. Regression Models

The project uses Simple Linear Regression, Multiple Linear Regression, and Stepwise Regression to predict blood pressure.

The Multiple Linear Regression model achieved the highest reported R-squared among these three regression models.

## Reported Results

| Model | Accuracy / R-squared | ROC-AUC |
|---|---:|---:|
| KNN | 80.03% | 0.8608 |
| Decision Tree | 80.37% | 0.7789 |
| Logistic Regression | 87.08% | 0.9156 |
| Simple Linear Regression | R² = 0.2363 | — |
| Multiple Linear Regression | R² = 0.3530 | — |
| Stepwise Regression | R² = 0.3481 | — |

*Note: Results are those reported in the project presentation and may vary when the scripts are run on a different dataset or with different settings.*

## Installation

Install R from https://cran.r-project.org/.

Install the required packages in the R console:

```r
install.packages(c(
  "caret",
  "class",
  "rpart",
  "rpart.plot",
  "ggplot2",
  "pROC"
))
```

## How to Run

1. Download or clone this repository.
2. Open the project folder in RStudio.
3. Ensure `patient_health_risk_dataset.csv` is available.
4. Run the required R script.
5. When prompted, select the CSV dataset using the file chooser.
6. Review the console output, model metrics, and generated plots.

### Scripts

- `analysis.R` — Data preprocessing, exploratory analysis, KNN, Decision Tree, and model comparison.
- `regression_analysis.R` — Statistical analysis, hypothesis tests, classification, and regression models.
- `final_prediction.R` — Model training and predictions for example new patients.

## Expected Outputs

- Dataset summaries and missing-value checks.
- Health risk distribution charts.
- KNN parameter selection chart.
- Decision Tree visualization.
- Confusion matrices and model evaluation metrics.
- Regression performance comparisons.
- Example patient risk predictions and predicted blood pressure.

## Limitations

- The dataset is simulated and is not a substitute for validated clinical data.
- The models are educational prototypes and should not be used to make medical decisions.
- Class imbalance can reduce the detection of High Risk patients.
- Further validation, cross-validation, threshold tuning, and testing on real-world data would be needed before practical use.

## Future Enhancements

- Improve detection of High Risk patients.
- Apply cross-validation and hyperparameter tuning.
- Evaluate Random Forest and other classification algorithms.
- Build an interactive Shiny dashboard.
- Validate models using suitable real-world, ethically sourced healthcare datasets.

## Author

**Mohanarangan M**

Department: Master of Computer Applications (MCA)

Project: Patient Health Risk Prediction Using Machine Learning in R

## License

This project is intended for academic and educational purposes. Add an appropriate open-source license if you intend to permit reuse and redistribution.

