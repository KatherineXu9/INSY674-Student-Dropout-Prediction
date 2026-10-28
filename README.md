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

# Evaluation and Interpretation

## Evaluation Objective

This proof-of-value study examines whether machine learning can help academic advisors identify students who may need additional support. The evaluation considers classification performance across **Dropout**, **Enrolled**, and **Graduate** outcomes and the practical consequences of incorrect predictions.

**Macro F1** is the primary comparison metric because it gives equal weight to all three classes. **Dropout recall** measures the proportion of actual dropout cases identified, while **dropout precision** measures the reliability of dropout alerts. Accuracy and the confusion matrix provide additional context.

The results presented here come from five-fold stratified cross-validation and out-of-fold predictions on the training data. They provide evidence of predictive potential and support model selection.

## Model Performance and Selection

A soft-voting ensemble combining tuned Logistic Regression, Gradient Boosting, and XGBoost classifiers was selected based on its mean cross-validation macro F1.

The selected scenario uses admission characteristics and first-semester academic information. Second-semester variables are excluded, and financial indicators are excluded from the main scenario because their recording dates have not been verified.

| Candidate | CV Macro F1 (Mean ± SD) | CV Dropout Recall (Mean ± SD) | CV Accuracy (Mean) |
|---|---:|---:|---:|
| **Soft-voting ensemble** | **0.6516 ± 0.0113** | **0.6939 ± 0.0248** | **0.7304** |
| Logistic Regression | 0.6476 ± 0.0158 | 0.6429 ± 0.0221 | 0.6948 |
| Gradient Boosting | 0.6344 ± 0.0135 | 0.7053 ± 0.0182 | 0.7259 |
| XGBoost | 0.6342 ± 0.0184 | 0.7080 ± 0.0187 | 0.7268 |

The ensemble achieved the highest mean macro F1 and accuracy among the tuned candidates. Its mean dropout recall of **69.39%** indicates that it identified approximately 69 out of every 100 actual dropout cases across validation folds. Approximately 31% were missed on average.

However, its macro F1 exceeded tuned Logistic Regression by only **0.0040**. This small difference does not establish a statistically reliable improvement. Logistic Regression therefore remains a credible alternative when simplicity and ease of interpretation are important.

XGBoost achieved the highest dropout recall, at **70.80%**, but a lower macro F1. The ensemble's selection reflects the project's emphasis on balanced performance across all three classes. A different choice could be appropriate if detecting dropout cases were given greater priority, depending on dropout precision and advising capacity.

## Confusion Matrix and Error Analysis

Out-of-fold predictions from the **untuned first-semester XGBoost model** provide preliminary evidence about classification errors. Each prediction was generated while the corresponding student was excluded from that fold's training data.

The following matrix describes this comparison model across **3,539 training records**. It does not represent the selected ensemble's performance.

| Actual / Predicted | Dropout | Enrolled | Graduate | Total |
|---|---:|---:|---:|---:|
| Dropout | 789 | 132 | 216 | 1,137 |
| Enrolled | 176 | 199 | 260 | 635 |
| Graduate | 108 | 117 | 1,542 | 1,767 |
| **Total** | **1,073** | **448** | **2,018** | **3,539** |


| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dropout | 0.7353 | 0.6939 | 0.7140 |
| Enrolled | 0.4442 | 0.3134 | 0.3675 |
| Graduate | 0.7641 | 0.8727 | 0.8148 |

### What Worked

Graduate was the strongest class, with recall of **87.27%** and F1 of **0.8148**. The model also correctly identified **789 of 1,137 dropout cases**, showing that first-semester information contains useful signals for identifying academic risk.

### What Remained Difficult

Enrolled was the weakest class, with recall of **31.34%** and F1 of **0.3675**. The most frequent individual error was predicting Graduate for an Enrolled student, occurring in **260 cases**.

Enrolled represents students who were still studying at the observation point. Their eventual graduation or dropout outcomes are unknown. This complicates interpretation because the model predicts the recorded status rather than an eventual outcome. The available results do not establish the specific causes of these errors.

### Missed Dropout Cases

The model missed **348 actual dropout cases**: 216 were predicted as Graduate and 132 as Enrolled. If outreach depended heavily on model predictions, these students could receive lower priority despite subsequently dropping out.

A prediction of Graduate should therefore not prevent a student from receiving assistance or override other evidence available to advisors.

### False Dropout Alerts

The model incorrectly classified **284 students as Dropout**, including 176 Enrolled students and 108 Graduates. Approximately **26.47%** of its dropout alerts did not match the observed Dropout label.

False alerts could increase advising workload and create unnecessary concern. However, an incorrect dropout prediction does not establish that support would be unhelpful. The practical value of outreach must be assessed through student and advisor outcomes as well as classification metrics.

### Additional Information for Understanding Errors

Attendance, recent engagement changes, employment demands, caregiving responsibilities, and students' own explanations could provide context that academic and administrative records do not capture.

These are proposed avenues for investigation rather than verified causes of the observed errors. Their usefulness would require further validation.

## Interpretation of the Main Findings

### First-Semester Information Improves Prediction

Under a fixed Logistic Regression comparison, adding first-semester academic information increased macro F1 from **0.4924 to 0.6206** and dropout recall from **52.94% to 69.66%**.

The improvements of **0.1282 in macro F1** and **16.72 percentage points in dropout recall** indicate that observed academic progress provides substantial predictive information beyond admission characteristics.

This supports reassessing students after the first semester. However, the additional information becomes available later, and some students may need assistance before first-semester results are recorded.

### More Complex Models Did Not Always Perform Better

Tuned Logistic Regression achieved a higher macro F1 than either tuned boosting model and performed close to the ensemble. Its macro F1 increased from **0.6206 before tuning to 0.6476 afterward**.

This finding shows that careful tuning of a simpler model can be competitive with more complex methods. The ensemble's modest advantage should be considered alongside its additional complexity.

### Financial Information Shows Predictive Potential

Adding financial indicators increased macro F1 from **0.6206 to 0.6669** in the fixed first-semester Logistic Regression comparison.

However, the recording dates of scholarship status, debtor status, and tuition payment status remain unverified. These results demonstrate predictive associations but do not establish that the variables would be available at the intended prediction date. They are therefore excluded from the main selected scenario.

### Important Predictors Describe Associations

Permutation importance for tuned Logistic Regression identified first-semester academic measures, including approved curricular units and grades, among the leading predictors. This is consistent with the improvement observed when first-semester information was added.

These importance results describe Logistic Regression rather than directly explaining the ensemble. They also do not establish causation. The analysis does not demonstrate that changing a predictor would prevent dropout or identify the appropriate intervention for an individual student.

## Assumptions and Limitations

| Limitation or Assumption | Likely Impact | How It Could Be Addressed |
|---|---|---|
| Reported results come from training-data cross-validation, and tuning and selection reused folds. | Performance estimates may be optimistic and do not establish independent test performance. | Validate the fixed model on independent data. |
| Financial recording dates are unverified. | These predictors may not be available at the intended prediction date. | Verify timestamps before including them in the main model. |
| First-semester academic results are required. | Equivalent predictions cannot be made at admission. | Investigate admission-stage screening followed by a first-semester update. |
| Enrolled is an intermediate status. | Predictions do not establish these students' eventual outcomes. | Obtain longer follow-up or investigate time-to-event modeling. |
| Historical patterns may change across cohorts or institutions. | Performance could decline in other settings. | Validate on later cohorts and external institutional data. |
| Aggregate scores may conceal subgroup differences. | Some groups could experience more missed cases or false alerts. | Assess subgroup precision and recall with sample sizes and uncertainty. |
| Personal circumstances are not fully represented in the predictors. | The model may miss risks that advisors could recognize through conversation. | Investigate additional information where appropriate and retain human review. |
| Prediction does not establish intervention effectiveness. | Accurate alerts may not lead to reduced dropout. | Conduct a prospective pilot assessing workload, student experience, and outcomes. |

No operational alert threshold has been established because advising capacity and the consequences of different errors have not been specified. Any future threshold should be chosen using validation data and evaluated independently.

## Conclusion

The study provides promising evidence that first-semester academic information can support student dropout prediction. The selected ensemble achieved a cross-validation macro F1 of **0.6516**, accuracy of **73.04%**, and dropout recall of **69.39%**.

Its advantage over tuned Logistic Regression was modest. Preliminary error analysis also revealed difficulty recognizing Enrolled students and a meaningful number of missed dropout cases. These findings support using predictions to prioritize advisor review and voluntary assistance.

The project demonstrates predictive potential within a proof-of-value study. Further independent validation and an intervention study would be needed to establish whether model-guided outreach improves student outcomes.
