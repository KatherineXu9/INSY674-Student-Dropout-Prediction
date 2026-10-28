# Student Dropout and Academic Success Prediction

A team machine learning project for INSY674. **Current status:** model development and training-data cross-validation are documented; independent evaluation on the reserved 885-record test set remains outstanding. The selected soft-voting ensemble has a training-CV macro F1 of 0.6516, which is not a final test score.

## Project Objective

This project investigates whether machine learning can identify students at risk of dropping out early enough to support university intervention. It predicts three academic statuses—Dropout, Enrolled, and Graduate—and examines the predictive value of academic performance and financial indicators.

The intended users are academic advisors and student support teams. Predictions could help them prioritize outreach and assess whether students need tutoring, financial guidance, or other support. This proof-of-value study evaluates predictive potential; it does not establish that using the model will reduce dropout.

## Research Questions

1. How effectively can machine learning distinguish between students who drop out, remain enrolled, or graduate?
2. How does prediction performance change when first-semester academic results are added to enrollment information?
3. What predictive value do academic and financial features provide individually and in combination?

The third question concerns predictive associations. It does not establish whether academic or financial difficulties cause students to drop out.

## Dataset Description

The project uses the Predict Students’ Dropout and Academic Success dataset from the UCI Machine Learning Repository. It contains 4,424 student records, 36 input features, and one target variable.

The features cover enrollment and demographic information, previous academic qualifications, financial indicators, first- and second-semester academic performance, and economic conditions. Examples include admission grades, age at enrollment, scholarship status, tuition payment status, courses passed, and semester grades.

The target records students’ status at the end of the normal duration of their course:

| Target class | Number of students | Percentage |
|---|---:|---:|
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |
| Graduate | 2,209 | 49.9% |

Enrolled students were still studying at that observation point; their eventual graduation or dropout outcomes are not provided.

An initial file check found no blank or common missing-value markers and no fully duplicated rows. The exploratory analysis below examines coded categories, unusual values, and feature distributions.

## Problem Framing and Prediction Timing

This is a supervised multiclass classification task because each historical student record has a known target label. The study uses offline batch training and evaluation.

We compare two prediction stages, with financial variants treated as conditional analyses:

- **Enrollment scenario:** Use only features confirmed to be available at enrollment.
- **First-semester scenario:** Add academic results available at the end of the first semester.

Second-semester academic features are excluded from all modeled scenarios. Financial recording times remain unverified. The admission feature group is an operational assumption about availability, rather than a fully verified timestamp dictionary.

Comparing these scenarios illustrates the trade-off between earlier prediction and the additional information gained after one semester.

## Proposed Early Warning Application

In a potential application, the model would estimate students’ academic status probabilities at a defined point in their studies. Student support staff could review students with elevated predicted dropout risk, discuss their circumstances, and offer appropriate assistance.

Predictions would support human review and voluntary assistance. They would not determine admission, financial aid eligibility, or academic penalties. Feature importance would describe patterns in model predictions and would not, by itself, determine which intervention a student needs.

The present project covers historical model development and evaluation. Live deployment and measurement of intervention outcomes are outside its scope.

## Scope and Assumptions

The study assumes that historical patterns may provide useful information for similar future students. However, performance on this dataset does not establish that the model will work equally well at other universities or in different educational settings.

The analyses compare models with a majority baseline, measure the contribution of first-semester results, and examine incremental academic and financial information. Claims about reducing dropout or improving student outcomes would require a separate intervention study.

## Exploratory Data Analysis

### Purpose and research alignment
This EDA examines outcome imbalance, academic patterns, financial indicators and data quality. It supports the team's questions about early prediction and the incremental predictive value of academic and financial information. It does not train a classifier or rank feature groups by predictive performance.

### Source, scope and reproducibility
The UCI dataset contains 4,424 student records, 36 input variables and Target. The three labels describe status at the end of the normal course duration. Enrolled is an unresolved eventual outcome, not a guarantee of future graduation.

A stratified 80%/20% split with random_state=42 assigns **3,539 records to training and 885 to held-out testing**. All outcome-related figures and summaries below use training records only. No held-out performance is calculated. The input file structure and full-row duplicates are checked globally; these checks do not select predictors. `tables/split_assignments.csv` stores the shared row assignment and `run_metadata.json` records the source hash. The preprocessing, model comparison, and tuning notebooks recreate this split using the same seed and stratification; reproduction also requires the original row order.

Six second-semester fields are excluded from substantive exploration. The first-semester scenario includes earlier information plus six first-semester measurements. Availability dates of financial and administrative information remain unverified: results involving these fields are descriptive, conditional on later confirming availability.

### Outcome distribution
| Status | n | Percent |
|---|---|---|
| Dropout | 1137 | 32.13 |
| Enrolled | 635 | 17.94 |
| Graduate | 1767 | 49.93 |

![Training target distribution](insy674_partb_result/figures/01_target_distribution.png)

**Takeaway:** class frequencies differ. An always-Graduate classifier would have training accuracy of 49.9%, without identifying any Dropout cases. This is a descriptive baseline, not a held-out model result. Model comparison prioritizes macro F1 and class-specific dropout recall/precision over accuracy alone.

### Data quality and complete attribute profiling
No literal missing cells were found in the eligible training fields. The full CSV contains 0 completely duplicated rows. These checks do not prove every value is valid or semantically complete. Parent qualification code 34 denotes Unknown in UCI documentation; preserve that distinction from a blank value. Other codes must be checked against the complete dictionary rather than assumed ordinal.

`tables/attribute_profile.csv` covers all 36 features: semantic type, feature group, time scope, missingness, cardinality, numeric range/skew or frequency-table reference, and preparation notes. Second-semester features are documented as excluded rather than having their values explored. Detailed category frequencies and numeric summaries accompany the profile.

The source header contains trailing whitespace in Daytime/evening attendance; the script strips header whitespace. Numeric codes for course, nationality, occupations and qualifications represent categories. Avoid Pearson correlations or continuous scaling of these codes as if the numeric distances were meaningful. Valid extreme ages or academic counts should not be dropped solely because they are boxplot outliers.

### Academic factors
![Academic distributions](insy674_partb_result/figures/02_academic_by_outcome.png)

Training-group medians:
| Target | Admission grade | Curricular units 1st sem (grade) | Curricular units 1st sem (approved) |
|---|---|---|---|
| Dropout | 123.3 | 10.83 | 2.0 |
| Enrolled | 123.9 | 12.0 | 5.0 |
| Graduate | 127.3 | 13.0 | 6.0 |

**Takeaway:** compare the group medians and overlapping distributions rather than treating a single low grade as a deterministic rule. First-semester grade and passed-unit counts show the academic separation visible in the figure; admission grade provides an earlier comparison. These are associations, with predictive value to be measured in CV through model comparison and tuning.

![Age and approval rate](insy674_partb_result/figures/04_age_and_approval_rate.png)

**Takeaway:** approved/enrolled units is a candidate academic progress feature. 148 training records have zero enrolled units and therefore an undefined ratio; keep them explicit and handle them inside the model pipeline. Zero semester grades should not automatically become missing values, because they may reflect no successful assessments. Age distributions overlap across outcomes, so age alone is not an individual risk decision rule.

### Financial indicators
![Financial outcome proportions](insy674_partb_result/figures/03_financial_outcome_rates.png)

- **Scholarship holder:** dropout share is 38.6% for No (n=2,645) and 12.9% for Yes (n=894).
- **Debtor:** dropout share is 28.2% for No (n=3,150) and 64.0% for Yes (n=389).
- **Tuition fees up to date:** dropout share is 86.9% for No (n=419) and 24.8% for Yes (n=3,120).

**Takeaway:** these are within-group outcome percentages, with sample counts shown to avoid confusing larger groups with higher risk. `financial_dropout_rates.csv` includes descriptive 95% Wilson intervals for dropout shares. Small groups are less precise. Financial status may be related to academic performance, background and recording time. These comparisons do not establish causality, independently adjusted effects or which support intervention would work.

### Distributions, context and correlations
![Numeric distributions](insy674_partb_result/figures/05_numeric_distributions.png)

**Takeaway:** inspect the actual shapes, zero concentrations and tails before choosing transformations. Robust scaling or standardization may help linear/distance-based models, but fit transformations inside training folds. Do not assume normality or remove tail observations automatically.

![Numeric correlations](insy674_partb_result/figures/06_numeric_correlations.png)

The strongest absolute numeric-feature correlations are:
| feature_1 | feature_2 | rho |
|---|---|---|
| Curricular units 1st sem (enrolled) | Curricular units 1st sem (approved) | 0.708 |
| Curricular units 1st sem (approved) | Curricular units 1st sem (grade) | 0.644 |
| Admission grade | Previous qualification (grade) | 0.583 |
| Curricular units 1st sem (credited) | Curricular units 1st sem (enrolled) | 0.42 |
| Curricular units 1st sem (enrolled) | Curricular units 1st sem (evaluations) | 0.419 |

**Takeaway:** related academic measurements can carry overlapping information; individual feature importance may be shared across correlated predictors. Spearman correlation is descriptive and does not establish causality. Nominal category codes and Target are excluded from this numeric matrix. Course-level sample sizes and dropout shares are exported in `course_dropout_rates.csv`; rare courses require caution and numeric course codes have no ordered meaning.

### Implications for modeling

The modeling workflow retains raw observations, encodes nominal categories, preserves unknown codes, and handles zero-denominator ratios inside preprocessing pipelines. Imputation, scaling, encoding, and any feature selection must be fitted without using held-out test data. Models use identical stratified CV conditions, with a majority dummy classifier and Logistic Regression as benchmarks.

Incremental comparisons hold the estimator fixed across admission context, added first-semester observations, added financial indicators, and their combination. Course and family background are context; GDP, inflation, and unemployment are macroeconomic context rather than direct student finances. The implemented models exclude these three macroeconomic fields. Administrative timing remains an assumption requiring verification.

### Assumptions, limitations and useful additional data
This is descriptive analysis of a historical institutional dataset. The random split estimates an internal comparison; it does not demonstrate future-cohort or cross-university generalization. Date-stamped administrative records, cohort identifiers and dropout dates would help validate temporal availability and prevent including students whose dropout preceded an intended semester-end prediction. Household income, aid amounts, attendance, support-service usage and intervention outcomes would improve understanding but are not in this CSV. Do not infer that changing a correlated feature will change outcomes.

## Data Preprocessing and Feature Engineering

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

Feature counts above refer to original input columns, before adding three engineered variables to first-semester scenarios and before one-hot encoding. The final encoded dimension depends on categories observed in each training fold.

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

## Model Training and Performance Comparison

### Candidate models

We trained and compared five machine learning models to predict three possible student outcomes: Dropout, Enrolled, and Graduate.

- **Logistic Regression:** A simple and interpretable baseline model.
- **Decision Tree:** Captures nonlinear relationships and provides interpretable decision rules.
- **Random Forest:** Combines multiple decision trees to improve prediction stability.
- **Gradient Boosting:** Builds trees sequentially to correct prediction errors.
- **XGBoost:** An optimized gradient boosting algorithm designed for efficient and accurate predictions.

Using models from different algorithm families allows us to compare their predictive performance, generalization ability, and effectiveness in identifying students at risk of dropping out.

### Validation protocol and metrics

The dataset was divided into 80% training data (3,539 observations) and 20% testing data (885 observations), using stratified sampling to preserve the target class distribution.

We applied **Stratified 5-Fold Cross-Validation** to the training set. In each iteration, the model was trained on four folds and validated on the remaining fold. All learned preprocessing steps were included within the pipeline so that they were fitted within each training fold. This protects against preprocessing leakage, but does not verify the real-world availability dates of input variables.

We evaluated each model using:

- **Accuracy:** Overall proportion of correctly classified students.
- **Macro F1:** Average F1 score across all three classes, giving equal importance to each class.
- **Balanced Accuracy:** Average recall across the three classes.
- **Dropout Recall:** Proportion of actual dropout students correctly identified.
- **Dropout Precision:** Proportion of predicted dropout students who actually dropped out.

Macro F1 was selected as the primary metric because the dataset contains imbalanced classes. Dropout Recall was also emphasized because the project's early warning objective requires identifying students who may need additional academic support.

The notebooks report means and standard deviations across folds. All CV uses StratifiedKFold(n_splits=5, shuffle=True, random_state=42). Fold standard deviations describe variation across folds, not confidence intervals.

### Model Performance Comparison

Source: saved outputs in [the initial comparison notebook](insy674_partb_result/Part_D.ipynb).

The following values are mean validation scores across five training-data folds for the untuned First Semester models. They are neither in-sample training scores nor scores on the 885-record test set.

| Model | Accuracy | Macro F1 | Dropout Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.7152 | 0.6206 | 0.6966 |
| Decision Tree | 0.6199 | 0.5534 | 0.6280 |
| Random Forest | 0.7245 | 0.6153 | 0.7185 |
| Gradient Boosting | 0.7251 | 0.6291 | 0.7132 |
| XGBoost | 0.7149 | 0.6320 | 0.6939 |
| Majority-Class Baseline | 0.4993 | 0.2220 | 0.0000 |

All five models outperformed the majority-class baseline in Macro F1.

**XGBoost achieved the highest Macro F1 (0.6320 ± 0.0149)**, while Gradient Boosting achieved the highest Accuracy (0.7251). Random Forest achieved the highest Dropout Recall (0.7185).

Although XGBoost ranked first in Macro F1, the difference between XGBoost and Gradient Boosting was relatively small. Therefore, their performance should be compared further during hyperparameter tuning.

### Feature Set Comparison

We also evaluated four feature sets to investigate how prediction performance changes as additional information becomes available.

| Feature Set | Best Model by Macro F1 | Macro F1 |
|---|---|---:|
| Admission Only | Logistic Regression | 0.4924 |
| Admission + Financial | Logistic Regression | 0.5461 |
| First Semester | XGBoost | 0.6320 |
| First Semester + Financial | Logistic Regression | 0.6669 |

Two important findings emerged.

**First, first-semester academic performance substantially improved the best observed macro F1.** The best Macro F1 increased from 0.4924 using admission-only features to 0.6320 after including first-semester information. Because the winning estimator changes between these rows, this comparison is descriptive; the fixed-estimator analysis below isolates the incremental feature contribution more consistently.

This suggests that students' actual university performance provides valuable information beyond their admission characteristics.

**Second, financial indicators provided additional predictive value.** Adding financial features further improved the best Macro F1 to 0.6669.

However, financial indicators were treated as exploratory features because their exact availability at the prediction date has not yet been verified. Their inclusion in a final early warning system depends on confirming that the information would be available when predictions are made.

Second-semester features were excluded to maintain the early prediction objective and avoid using future academic information.

### Overfitting and Error Analysis

We compared training and cross-validation performance to evaluate potential overfitting.

For the First Semester feature set, XGBoost achieved a training Macro F1 of 0.9994 but a cross-validation Macro F1 of 0.6320. Random Forest showed a similar pattern, suggesting substantial overfitting.

In comparison, Logistic Regression showed a smaller generalization gap, indicating more stable performance on unseen validation folds.

Out-of-fold errors for the untuned XGBoost comparison model are examined in Evaluation and Interpretation below.

### Model Shortlisting

Based on Macro F1 performance using the First Semester feature set, we selected three models for further optimization:

1. **XGBoost:** Highest Macro F1, with potential for improvement through regularization and hyperparameter tuning.
2. **Gradient Boosting:** Competitive overall performance, strong Dropout Recall, and a smaller generalization gap than XGBoost.
3. **Logistic Regression:** Strong and relatively stable performance despite its simpler structure, making it a useful interpretable benchmark.

These models represent different approaches and provide a meaningful comparison for the next stage.

Although Random Forest achieved the highest Dropout Recall, it was not included in the initial top three because the shortlist was ranked primarily by Macro F1.

The First Semester + Financial scenario also produced promising results and should be reconsidered once the timing of financial information is verified.

## Hyperparameter Tuning and Feature Importance

### Objective and reproducibility

Sources: [tuning report](insy674_partb_result/part_e_results/PART_E_REPORT.md) and the exported CSV files linked in the repository guide.
Tuning uses the same stratified 80/20 split (seed 42), five-fold stratified CV, and four feature sets as the initial model comparison. Only training rows are used, and the 885 test records remain reserved for independent evaluation.

### Hyperparameter tuning (First Semester, no financial indicators)
We applied RandomizedSearchCV with five stratified folds to the three models shortlisted in the initial comparison: XGBoost, Gradient Boosting, and Logistic Regression. Search spaces addressed model complexity, learning rate, regularization, and (for logistic regression) class weighting. This is a randomized search of a bounded hyperparameter space, not an exhaustive optimization or a causal experiment.

| Model               |   CV Macro F1 Mean |   CV Macro F1 Std |   Train - CV Gap |
|:--------------------|-------------------:|------------------:|-----------------:|
| Logistic Regression |             0.6476 |            0.0158 |           0.0292 |
| Gradient Boosting   |             0.6344 |            0.0135 |           0.1455 |
| XGBoost             |             0.6342 |            0.0184 |           0.1671 |

Figure: ![Tuned model and ensemble comparison](insy674_partb_result/part_e_results/01_tuning_and_ensemble.png)

### Soft-voting ensemble and model selection
A three-model soft-voting ensemble was compared with each tuned model using the same first-semester inputs and five-fold CV.

| Candidate            |   Macro F1 Mean |   Macro F1 Std |   Dropout Recall Mean |   Dropout Recall Std |   Accuracy Mean |
|:---------------------|----------------:|---------------:|----------------------:|---------------------:|----------------:|
| Soft Voting Ensemble |          0.6516 |         0.0113 |                0.6939 |               0.0248 |          0.7304 |
| Logistic Regression  |          0.6476 |         0.0158 |                0.6429 |               0.0221 |          0.6948 |
| Gradient Boosting    |          0.6344 |         0.0135 |                0.7053 |               0.0182 |          0.7259 |
| XGBoost              |          0.6342 |         0.0184 |                0.708  |               0.0187 |          0.7268 |

**Provisional training-CV choice:** Soft Voting Ensemble. These cross-validation results help with selection but may be optimistic because the same training data informs tuning and selection. The locked model still requires evaluation once on the reserved held-out test set. No final-test performance claim is made here.

### Incremental academic and financial information
To hold modeling conditions constant, we compared all four feature sets using the same untuned Logistic Regression estimator and the same five stratified folds:

| Feature Set                |   Raw Feature Count |   Macro F1 Mean |   Macro F1 Std |   Dropout Recall Mean |
|:---------------------------|--------------------:|----------------:|---------------:|----------------------:|
| Admission Only             |                  18 |          0.4924 |         0.0182 |                0.5294 |
| First Semester             |                  24 |          0.6206 |         0.0099 |                0.6966 |
| Admission + Financial      |                  21 |          0.5464 |         0.011  |                0.598  |
| First Semester + Financial |                  27 |          0.6669 |         0.0148 |                0.7511 |

- Adding first-semester academic results to admission context: Δ macro F1 = **+0.1282**.
- Adding financial variables to admission context: Δ macro F1 = **+0.0540** (conditional on verified timing).
- Adding financial variables to first-semester context: Δ macro F1 = **+0.0463** (conditional on verified timing).

![Feature-set CV comparison](insy674_partb_result/part_e_results/02_feature_group_ablation.png)

This is a comparison of predictive contributions under a fixed-model CV setup. The admission group already includes some academic-background attributes, so the academic delta specifically measures **added first-semester observations**. The financial fields (scholarship, debtor, tuition-payment status) have unverified recording times. Do not interpret these associations as causal effects or proven deployable early-warning inputs.

### Feature importance
For interpretability we computed permutation importance on training-held-out CV folds using the best tuned single model (Logistic Regression) and reported the reduction in validation macro F1 when an original column was shuffled. The top eight columns were: **Curricular units 1st sem (approved), Curricular units 1st sem (grade), Curricular units 1st sem (enrolled), Curricular units 1st sem (evaluations), Course, Curricular units 1st sem (credited), Age at enrollment, Application mode**.

![First-semester permutation importance](insy674_partb_result/part_e_results/03_primary_feature_importance.png)

A separate untuned Logistic Regression model using First Semester + Financial was interpreted to examine the three finance-related columns:

| Feature                 |   Mean |    Std |
|:------------------------|-------:|-------:|
| Tuition fees up to date | 0.049  | 0.0091 |
| Scholarship holder      | 0.0069 | 0.005  |
| Debtor                  | 0.0044 | 0.0044 |

![Financial-scenario permutation importance](insy674_partb_result/part_e_results/04_conditional_financial_importance.png)

Permutation importance is model-specific and may distribute importance across correlated inputs; it does not imply causality or justify an individual-level intervention by itself.

# Evaluation and Interpretation

## Evaluation Objective and Method

This proof-of-value study evaluates whether machine learning can help academic advisors identify students who may need additional support. Macro F1 is the primary metric because it gives equal weight to Dropout, Enrolled, and Graduate outcomes. Dropout recall measures the proportion of actual dropout cases identified, while dropout precision measures the reliability of dropout alerts.

The final evaluation used the repository's stratified 80/20 split with random state 42: 3,539 training students and 885 held-out test students. The test set contained 284 Dropout, 159 Enrolled, and 442 Graduate students.

The selected soft-voting ensemble combines Logistic Regression, Gradient Boosting, and XGBoost. Its hyperparameters and membership were fixed using the saved model-selection configuration before test predictions were generated. The original preprocessing and feature engineering were reproduced and fitted on the training records only. No hyperparameters, voting weights, or decision thresholds were adjusted after examining the test results.

The model uses 24 original admission and first-semester inputs, with first-semester indicators created inside the preprocessing pipeline. Second-semester information is excluded. Financial indicators are excluded from the main scenario because their recording dates have not been verified.

## Held-Out Test Results and Baseline Comparison

All models were evaluated on the same 885 test students. The majority-class baseline always predicts Graduate. The Logistic Regression baseline is untuned, with the same first-semester preprocessing and inputs, `max_iter=2000`, and random state 42; it is distinct from the tuned Logistic Regression ensemble member.

| Model | Accuracy | Macro F1 | Dropout Precision | Dropout Recall | Dropout F1 |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.4994 | 0.2221 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression baseline | 0.7153 | 0.6100 | 0.7148 | 0.7148 | 0.7148 |
| **Selected soft-voting ensemble** | **0.7266** | **0.6549** | **0.7724** | **0.6690** | **0.7170** |


The ensemble achieved **72.66% accuracy** and **0.6549 macro F1**, improving substantially on the majority-class baseline. Compared with untuned Logistic Regression, it improved accuracy by **1.13 percentage points** and macro F1 by **0.0449**. The ensemble therefore provided better balanced performance across the three outcomes on this test set.

However, the ensemble did not dominate every metric. Its dropout precision was higher than Logistic Regression's by **5.76 percentage points**, while its dropout recall was lower by **4.58 percentage points**. The ensemble identified 190 dropout students, compared with 203 for Logistic Regression, and generated 56 false dropout alerts, compared with 81 for Logistic Regression. Its dropout F1 advantage was small: approximately **0.0022**.

These results identify a practical trade-off. The ensemble provides more reliable dropout alerts and higher overall macro F1, while the baseline detects more actual dropout cases. The operational choice should also consider advising capacity and the consequences of missed cases and false alerts.

## Comparison with Cross-Validation Performance

| Metric | Training CV Mean | Held-Out Test |
|---|---:|---:|
| Macro F1 | 0.6516 | 0.6549 |
| Accuracy | 0.7304 | 0.7266 |
| Dropout recall | 0.6939 | 0.6690 |

The ensemble's test macro F1 was close to its cross-validation mean, and accuracy decreased by approximately 0.39 percentage points. Dropout recall decreased by approximately 2.49 percentage points. These results are broadly consistent with the training-stage estimates, although one random held-out split does not establish performance across future cohorts or other institutions. Small differences should not be interpreted as statistically reliable improvements without additional uncertainty analysis.

## Confusion Matrix and Class-Level Performance

The following confusion matrix describes the **selected ensemble on the held-out test set**. Rows show actual outcomes and columns show predicted outcomes.

| Actual / Predicted | Dropout | Enrolled | Graduate | Total |
|---|---:|---:|---:|---:|
| Dropout | 190 | 42 | 52 | 284 |
| Enrolled | 37 | 62 | 60 | 159 |
| Graduate | 19 | 32 | 391 | 442 |
| **Total** | **246** | **136** | **503** | **885** |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Dropout | 0.7724 | 0.6690 | 0.7170 | 284 |
| Enrolled | 0.4559 | 0.3899 | 0.4203 | 159 |
| Graduate | 0.7773 | 0.8846 | 0.8275 | 442 |

### What Worked

Graduate was the strongest class, with **88.46% recall** and **0.8275 F1**. The ensemble also correctly identified **190 of 284 dropout students**. Its dropout precision of **77.24%** means that approximately 77 of every 100 dropout alerts matched the observed Dropout outcome in this test sample.

### What Remained Difficult

Enrolled remained the weakest class, with **38.99% recall** and **0.4203 F1**. Only 62 of 159 Enrolled students were correctly classified. The most frequent individual error was predicting Graduate for an Enrolled student, occurring in **60 cases**.

Enrolled records students still studying at the observation point. Their eventual graduation or dropout outcomes are unknown. This limits interpretation of the class. Its smaller sample size and overlap with other student profiles are possible explanations for difficulty, rather than explanations established by this evaluation.

### Missed Dropout Cases

The ensemble missed **94 of 284 dropout cases (33.10%)**: 42 were predicted as Enrolled and 52 as Graduate. If outreach relied heavily on predictions, these students could receive lower priority. A prediction of Graduate should therefore not override other evidence or prevent a student from receiving assistance.

### False Dropout Alerts

The ensemble incorrectly flagged **56 non-dropout students**: 37 Enrolled students and 19 Graduates. These represented **22.76% of its 246 dropout alerts**. Additional outreach could create workload and unnecessary concern, although a student who does not drop out may still benefit from support. The practical value of outreach requires separate assessment.

### Additional Information for Understanding Errors

Attendance, recent changes in engagement, employment demands, caregiving responsibilities, and student-reported circumstances could help advisors understand cases not explained by academic and administrative records. These are proposed directions for investigation, not verified causes of the observed errors. Any model changes motivated by this analysis should be developed using training or validation data and assessed on fresh independent data.

## Interpretation of the Main Findings

### First-Semester Information Adds Predictive Value

In the existing fixed Logistic Regression cross-validation comparison, adding first-semester information increased macro F1 from **0.4924 to 0.6206** and dropout recall from **52.94% to 69.66%**. The gains of 0.1282 in macro F1 and 16.72 percentage points in dropout recall support the predictive value of observed academic progress beyond admission characteristics.

These are training-stage feature-set comparisons, separate from the final test results. They support reassessment after the first semester, while recognizing that some students may need assistance before these results become available.

### Model Complexity and Practical Value

The ensemble improved test macro F1 over untuned Logistic Regression, but its dropout F1 was only slightly higher and its dropout recall was lower. Model complexity should therefore be justified by the objective and the observed trade-off rather than accuracy alone.

### Financial Indicators and Feature Importance

The existing training-stage comparison showed that adding financial indicators increased first-semester Logistic Regression macro F1 from **0.6206 to 0.6669**. Their recording dates remain unverified, so these variables were excluded from the final ensemble's inputs.

Permutation importance for tuned Logistic Regression identified first-semester academic measures, including approved curricular units and grades, among the leading predictors. These model-specific results do not directly explain the ensemble and do not establish causation or determine which intervention a student needs.

## Assumptions and Limitations

| Limitation or Assumption | Likely Impact | How It Could Be Addressed |
|---|---|---|
| Evaluation uses one random held-out split from the same dataset. | Performance may differ for later cohorts or other institutions. | Conduct temporal and external validation and report uncertainty. |
| Original training software versions were not specified in the saved model handoff. | Reconstructed predictions may differ across software versions. | Retain the evaluation script, dependency versions, and saved predictions. |
| Financial recording dates are unverified. | Their predictive contribution may not be available at the intended prediction date. | Verify timestamps before incorporating financial inputs. |
| First-semester results are required. | Equivalent predictions cannot be made at admission. | Investigate admission-stage screening followed by reassessment. |
| Enrolled is an intermediate status. | Predictions do not establish these students' eventual outcomes. | Obtain longer follow-up or investigate time-to-event modeling. |
| Aggregate metrics may conceal subgroup differences. | Some groups may experience more missed cases or false alerts. | Assess subgroup performance with sample sizes and uncertainty. |
| Personal circumstances are incompletely represented. | The model may miss risks recognized through advisor conversations. | Retain human review and investigate appropriate additional information. |
| Prediction does not establish intervention effectiveness. | Accurate alerts may not reduce dropout. | Conduct a prospective pilot measuring workload, student experience, and outcomes. |

No operational alert threshold has been established. The evaluation uses the ensemble's default class prediction, which selects the class with the highest averaged probability. Any future alert policy should account for advising capacity, assess probability calibration, and be developed using validation data before independent evaluation.

## Conclusion

The held-out evaluation provides evidence of predictive value: the selected ensemble achieved **72.66% accuracy**, **0.6549 macro F1**, **77.24% dropout precision**, and **66.90% dropout recall**. It improved macro F1 over both baselines, while untuned Logistic Regression detected more dropout cases at the cost of more false alerts.

The central practical limitation is that **94 of 284 dropout cases were missed**, and Enrolled remained difficult to classify. Predictions should support advisor review and voluntary assistance, with other channels available for students whose needs are not identified by the model.

The study supports further investigation of a first-semester early warning system. It does not demonstrate that model-guided outreach reduces dropout; that claim requires a prospective intervention study.


## Repository Guide and Reproduction

The consolidated notebook is the single entry point for the project's exploratory analysis, preprocessing, model comparison, hyperparameter tuning, ensemble comparison, and feature importance.

| File | Purpose |
|---|---|
| [README.md](README.md) | Project overview, methods, reported results, interpretation, and limitations. |
| [Student_Dropout.ipynb](Student_Dropout.ipynb) | Consolidated analysis and modeling code, arranged to run from top to bottom. |
| [predict+students+dropout+and+academic+success.zip](predict+students+dropout+and+academic+success.zip) | Original dataset archive containing the semicolon-separated `data.csv`. |

### Run the notebook

1. Download `Student_Dropout.ipynb` and the dataset ZIP into the same folder. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab. In Colab, upload the dataset ZIP to the runtime's working folder. GitHub displays the notebook but does not execute it.
2. Use a Python environment with pandas, NumPy, SciPy, scikit-learn, XGBoost, Matplotlib, seaborn 0.13 or later, IPython, and tabulate. If packages are missing, run the following in a notebook cell, then restart the kernel/runtime:

   ```python
   %pip install pandas numpy scipy scikit-learn xgboost matplotlib "seaborn>=0.13" ipython tabulate
   ```

3. Keep the ZIP compressed: the notebook reads `data.csv` directly from it. Alternatively, extract `data.csv` beside the notebook without changing its row order. The ZIP lookup also accepts download suffixes such as ` (1)`.
4. Keep `FAST_MODE=False` for the full configured analysis. A quick run with `FAST_MODE=True` uses fewer search trials and permutation repeats and should not be treated as equivalent to the full run.
5. Restart the kernel/runtime and run all cells in order. Tables and figures appear below their code cells as execution progresses. Model comparison, tuning, and permutation importance take longer than the initial data checks. Empty outputs in the downloaded notebook do not represent completed results.

## Data Source

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students’ Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5MC89](https://doi.org/10.24432/C5MC89).

[Dataset documentation and variable definitions](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). Dataset licensed CC BY 4.0.
