# INSY674-Student-Dropout-Prediction
Predicting student dropout and academic success using machine learning.

# Student Dropout and Academic Success Prediction

## 1. Project Objective

This project investigates whether machine learning can identify students at risk of dropping out early enough to support university intervention. It predicts three academic statuses—Dropout, Enrolled, and Graduate—and examines the predictive value of academic performance and financial indicators.

The intended users are academic advisors and student support teams. Predictions could help them prioritize outreach and assess whether students need tutoring, financial guidance, or other support. This proof-of-value study evaluates predictive potential; it does not establish that using the model will reduce dropout.

## 2. Research Questions

1. How effectively can machine learning distinguish between students who drop out, remain enrolled, or graduate?
2. How does prediction performance change when first-semester academic results are added to enrollment information?
3. What predictive value do academic and financial features provide individually and in combination?

The third question concerns predictive associations. It does not establish whether academic or financial difficulties cause students to drop out.

## 3. Dataset Description

The project uses the Predict Students’ Dropout and Academic Success dataset from the UCI Machine Learning Repository. It contains 4,424 student records, 36 input features, and one target variable.

The features cover enrollment and demographic information, previous academic qualifications, financial indicators, first- and second-semester academic performance, and economic conditions. Examples include admission grades, age at enrollment, scholarship status, tuition payment status, courses passed, and semester grades.

The target records students’ status at the end of the normal duration of their course:

| Target class | Number of students | Percentage |
|---|---:|---:|
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |
| Graduate | 2,209 | 49.9% |

Enrolled students were still studying at that observation point; their eventual graduation or dropout outcomes are not provided.

An initial file check found no blank or common missing-value markers and no fully duplicated rows. Further exploration will examine coded categories, unusual values, and feature distributions.

## 4. Problem Framing and Prediction Timing

This is a supervised multiclass classification task because each historical student record has a known target label. The study uses offline batch training and evaluation.

Two prediction scenarios are proposed:

- **Enrollment scenario:** Use only features confirmed to be available at enrollment.
- **First-semester scenario:** Add academic results available at the end of the first semester.

Second-semester academic features will be excluded from both scenarios. The availability of financial and other administrative variables will be checked before assigning them to either feature set. Variables with uncertain recording times will be excluded from timing-specific claims or explicitly treated as assumptions.

Comparing these scenarios will show the trade-off between earlier prediction and the additional information gained after one semester.

## 5. Evaluation Metrics and Baselines

Macro F1 is proposed as the primary model comparison metric. It gives equal weight to the three classes, helping assess performance when class sizes differ.

Dropout recall will be a key application metric because it measures the proportion of actual dropout cases correctly identified. Dropout precision will also be reported to assess how many flagged students actually belong to the dropout class. These metrics capture the trade-off between missing students who may need support and generating unnecessary alerts.

Accuracy, class-level scores, and a confusion matrix will provide additional context. No operational alert threshold is assumed at this stage because advising capacity and the costs of different errors have not been specified.

A majority-class dummy classifier will provide a minimum performance benchmark, while logistic regression will serve as an interpretable model baseline. Models will be compared using the same stratified cross-validation folds on the training data, reporting mean scores and standard deviations. A separate held-out test set will be reserved for final evaluation.

A simple rule-based flag, such as low first-semester performance, could provide a practical comparison if time permits. The project does not assume that this is an existing policy at the institution.

## 6. Proposed Early Warning Application

In a potential application, the model would estimate students’ academic status probabilities at a defined point in their studies. Student support staff could review students with elevated predicted dropout risk, discuss their circumstances, and offer appropriate assistance.

Predictions would support human review and voluntary assistance. They would not determine admission, financial aid eligibility, or academic penalties. Feature importance would describe patterns in model predictions and would not, by itself, determine which intervention a student needs.

The present project covers historical model development and evaluation. Live deployment and measurement of intervention outcomes are outside its scope.

## 7. Scope and Assumptions

The study assumes that historical patterns may provide useful information for similar future students. However, performance on this dataset does not establish that the model will work equally well at other universities or in different educational settings.

The project will assess whether the model improves on baseline predictions, whether first-semester results add useful information, and how academic and financial feature groups contribute to prediction. Claims about reducing dropout or improving student outcomes would require a separate intervention study.

## Data Source

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students’ Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89



## Data Preprocessing and Feature Engineering （Part C）

### Data Cleaning

The dataset contains 4,424 observations, 36 input features, and one target variable. Initial data quality checks identified no missing values or fully duplicated records. Therefore, no observations were removed. Column names were standardized to remove hidden characters and unnecessary whitespace.

Zero values in academic variables were retained because they may represent meaningful academic outcomes rather than missing data.

### Feature Selection and Data Leakage Prevention

Four feature sets were created to support different prediction scenarios:

- **Admission Only:** 18 features, excluding first- and second-semester results, selected financial indicators, and macroeconomic variables.
- **First Semester:** 24 features, combining the admission feature set with six first-semester academic variables.
- **Admission + Financial:** 21 features, adding three financial indicators to the admission feature set.
- **First Semester + Financial:** 27 features, combining admission information, first-semester academic variables, and three financial indicators.

All six second-semester features were excluded because they would not be available at the intended early prediction points.

The financial scenarios include scholarship status, debtor status, and tuition payment status. Their recording times have not been independently verified, so these scenarios are treated as conditional analyses rather than confirmed leakage-free early prediction settings.

### Feature Engineering

Three additional features were created for the first-semester scenarios:

1. **First Semester Pass Rate:** Approved curricular units divided by enrolled curricular units.
2. **Evaluations per Enrolled Course:** Number of evaluations divided by enrolled curricular units.
3. **No First-Semester Enrollment:** A binary indicator identifying students with zero enrolled curricular units.

When enrolled units equal zero, the two ratios are treated as undefined (`NaN`) and handled through median imputation within the preprocessing pipeline. The zero-enrollment indicator preserves information about these students.

### Encoding and Scaling

Categorical features, including numerically coded categories, were processed using OneHotEncoder with `handle_unknown="ignore"`.

Numerical features were processed using median imputation and StandardScaler. Categorical features were assigned most-frequent imputation when needed.

These transformations were combined using Scikit-learn Pipeline and ColumnTransformer to provide a consistent and reusable preprocessing workflow.

### Train/Test Split and Pipeline Validation

The dataset was divided into 80% training data (3,539 observations) and 20% held-out test data (885 observations) using stratified sampling with `random_state=42`.

The preprocessing pipelines were tested using training data only. A Logistic Regression smoke test confirmed that the first-semester pipeline could be successfully fitted with a classifier. Integration checks also verified that all four feature sets could be transformed successfully.

The pipelines are designed to be integrated with downstream classification models so that preprocessing parameters can be fitted separately within each cross-validation training fold, reducing the risk of data leakage.
