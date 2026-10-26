"""Person B: descriptive EDA. No models are trained and no test outcomes analyzed.
Install: python -m pip install pandas numpy matplotlib seaborn scikit-learn
Run: python Person_B_EDA_Script.py --data /path/to/data.csv
Outputs: ./person_b_eda_results (override with --output).
Python 3.13 was used to verify this script.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['Dropout', 'Enrolled', 'Graduate']
COLORS = ['#C65B51', '#D5A541', '#32818A']
FIN = ['Scholarship holder', 'Debtor', 'Tuition fees up to date']
CAT = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
       'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
       "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
       'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
NUM = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
       'Unemployment rate', 'Inflation rate', 'GDP', 'Application order']
SOURCE = 'https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success'

def md(df):
    def clean(x): return str(x).replace('|', '/').replace('\n', ' ')
    return '| ' + ' | '.join(map(clean, df.columns)) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(clean,row)) + ' |' for row in df.itertuples(index=False,name=None))

def save(name):
    plt.savefig(ROOT/'figures'/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close()

def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Path to original semicolon-separated data.csv')
    parser.add_argument('--output', type=Path, default=Path.cwd()/'person_b_eda_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=.2)
    args=parser.parse_args()
    ROOT=args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    for p in ['figures','tables','reports']: (ROOT/p).mkdir(exist_ok=True)
    d=pd.read_csv(args.data, sep=';')
    d.columns=d.columns.str.strip()
    if not set(d.Target.unique()) == set(ORDER): raise ValueError('Unexpected target labels')
    ids=np.arange(len(d))
    tr,te=train_test_split(ids,test_size=args.test_size,random_state=args.seed,stratify=d.Target)
    # Export assignment only: original CSV row positions are identifiers, not predictors.
    assignment=pd.DataFrame({'row_id':ids,'split':np.where(np.isin(ids,tr),'train','test')})
    assignment.to_csv(ROOT/'tables/split_assignments.csv',index=False)
    x=d.iloc[tr].copy()
    features=[c for c in d if c!='Target']
    second=[c for c in features if '2nd sem' in c]
    first=[c for c in features if '1st sem' in c]
    eligible=[c for c in features if c not in second]
    numeric=[c for c in eligible if c not in CAT]
    sns.set_theme(style='whitegrid',font_scale=1.05)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.figsize':(9,5)})
    counts=x.Target.value_counts().reindex(ORDER)
    ct=pd.DataFrame({'Status':ORDER,'n':counts.values,'Percent':(100*counts/len(x)).round(2).values})
    ct.to_csv(ROOT/'tables/class_distribution.csv',index=False)
    fig,ax=plt.subplots()
    bars=ax.bar(ORDER,counts,color=COLORS)
    for b,n in zip(bars,counts): ax.text(b.get_x()+b.get_width()/2,n+15,f'{n:,} ({n/len(x):.1%})',ha='center')
    ax.set(title='Training outcomes are imbalanced',ylabel='Number of students',ylim=(0,max(counts)*1.17))
    ax.text(0,-.17,f'Training set only, n = {len(x):,}',transform=ax.transAxes,color='#555555')
    save('01_target_distribution')
    # All attributes profiled; second-semester values deliberately not explored.
    rows=[]
    for c in features:
        excluded=c in second
        s=x[c]
        role='Financial' if c in FIN else ('Academic' if c in first or c in ['Admission grade','Previous qualification','Previous qualification (grade)'] else 'Context')
        timing='Excluded: after prediction horizon' if excluded else ('End of first semester' if c in first else 'Enrollment candidate; verify administrative timing')
        note='Do not interpret numeric codes as continuous.' if c in CAT else 'Inspect skew and extremes; retain valid observations.'
        if c in ["Mother's qualification", "Father's qualification"]:
            note += f' Unknown code 34: {int((s==34).sum())} ({(s==34).mean():.1%}).'
        if '(grade)' in c and 'sem' in c: note+=' Zero grade may indicate no successful assessment; do not replace with missing automatically.'
        if c in FIN: note+=' Recording date unconfirmed; early-availability assumption must be documented.'
        rows.append({'feature':c,'group':role,'semantic_type':'Categorical code' if c in CAT else ('Count' if c in first or c in second else 'Numeric/ordinal'),
          'timing':timing,'missing_pct':round(s.isna().mean()*100,3) if not excluded else 'not analyzed',
          'unique':s.nunique() if not excluded else 'not analyzed',
          'distribution':('Excluded' if excluded else ('Category frequencies in categorical_counts.csv' if c in CAT else f'min={s.min():.3g}; median={s.median():.3g}; max={s.max():.3g}; skew={s.skew():.2f}')),
          'usefulness_and_preparation':('Excluded from both scenarios.' if excluded else note)})
    profile=pd.DataFrame(rows)
    profile.to_csv(ROOT/'tables/attribute_profile.csv',index=False)
    pd.concat([x[c].value_counts(dropna=False).rename_axis('value').reset_index(name='n').assign(feature=c) for c in CAT],ignore_index=True).to_csv(ROOT/'tables/categorical_counts.csv',index=False)
    x[numeric].describe().T.to_csv(ROOT/'tables/numeric_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    ac=['Admission grade','Curricular units 1st sem (grade)','Curricular units 1st sem (approved)']
    labels=['Admission grade','First-semester grade','First-semester units approved']
    for ax,c,label in zip(axes,ac,labels):
        sns.boxplot(data=x,x='Target',y=c,order=ORDER,hue='Target',hue_order=ORDER,palette=COLORS,legend=False,ax=ax,fliersize=2)
        ax.set(xlabel='',ylabel=label,title=label)
        ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Academic distributions by observed outcome (training set)',y=1.03)
    fig.tight_layout(); save('02_academic_by_outcome')
    med=x.groupby('Target')[ac].median().reindex(ORDER)
    med.to_csv(ROOT/'tables/academic_medians.csv')
    # Within-group rates; Wilson intervals for descriptive uncertainty.
    financial=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,c in zip(axes,FIN):
        g=pd.crosstab(x[c],x.Target).reindex(columns=ORDER,fill_value=0)
        rates=g.div(g.sum(axis=1),axis=0)
        rates.plot.bar(stacked=True,color=COLORS,ax=ax,legend=False,rot=0)
        for j,(v,row) in enumerate(g.iterrows()):
            n=int(row.sum()); k=int(row['Dropout']); p=k/n; z=1.96
            center=(p+z*z/(2*n))/(1+z*z/n)
            half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            financial.append({'feature':c,'value':v,'n':n,'dropout_n':k,'dropout_pct':round(100*p,2),'wilson_low_pct':round(100*(center-half),2),'wilson_high_pct':round(100*(center+half),2)})
            ax.text(j,1.025,f'n={n:,}',ha='center',fontsize=10)
        ax.set(title=c,xlabel='0 = No; 1 = Yes',ylim=(0,1.15))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].set_ylabel('Within-group share')
    fig.legend(ORDER,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.08))
    fig.tight_layout(); save('03_financial_outcome_rates')
    fin=pd.DataFrame(financial); fin.to_csv(ROOT/'tables/financial_dropout_rates.csv',index=False)
    # Engineered indicator is exploratory; C should implement it inside production pipeline.
    enrolled=x['Curricular units 1st sem (enrolled)']
    x['First-semester approval rate']=x['Curricular units 1st sem (approved)'].div(enrolled.where(enrolled>0))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,c,label in zip(axes,['Age at enrollment','First-semester approval rate'],['Age at enrollment','Approved / enrolled units (first semester)']):
        sns.boxplot(data=x,x='Target',y=c,order=ORDER,hue='Target',hue_order=ORDER,palette=COLORS,legend=False,ax=ax,fliersize=2)
        ax.set(xlabel='',ylabel=label)
    fig.tight_layout();save('04_age_and_approval_rate')
    courses=x.groupby('Course').agg(n=('Target','size'),dropout_rate=('Target',lambda s:s.eq('Dropout').mean())).sort_values('n',ascending=False)
    courses.to_csv(ROOT/'tables/course_dropout_rates.csv')
    hist=['Age at enrollment','Admission grade','Curricular units 1st sem (grade)','Curricular units 1st sem (approved)']
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,c in zip(axes.flat,hist): sns.histplot(data=x,x=c,bins=25,ax=ax,color=COLORS[2])
    fig.tight_layout();save('05_numeric_distributions')
    corrcols=['Admission grade','Previous qualification (grade)','Age at enrollment',*first,'Unemployment rate','Inflation rate','GDP']
    corr=x[corrcols].corr(method='spearman')
    corr.to_csv(ROOT/'tables/spearman_correlations.csv')
    short=['Admission','Previous grade','Age','S1 credited','S1 enrolled','S1 evaluations','S1 approved','S1 grade','S1 no evaluations','Unemployment','Inflation','GDP']
    fig,ax=plt.subplots(figsize=(11,9))
    sns.heatmap(corr,xticklabels=short,yticklabels=short,cmap='vlag',vmin=-1,vmax=1,center=0,ax=ax,square=True,cbar_kws={'label':'Spearman correlation'})
    ax.set_title('Numeric feature relationships (training set)')
    save('06_numeric_correlations')
    pairs=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack().rename('rho').reset_index()
    pairs.columns=['feature_1','feature_2','rho']; pairs['abs_rho']=pairs.rho.abs()
    pairs=pairs.sort_values('abs_rho',ascending=False);pairs.to_csv(ROOT/'tables/top_correlations.csv',index=False)
    metadata={'source':SOURCE,'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'total_rows':len(d),'train_n':len(x),'test_n':len(te),'seed':args.seed,'test_size':args.test_size,'row_id':'zero-based original data row, excluding header','full_file_duplicate_rows':int(d.duplicated().sum()),'training_missing_cells':int(x[eligible].isna().sum().sum()),'second_semester_excluded':second}
    (ROOT/'tables/run_metadata.json').write_text(json.dumps(metadata,indent=2))
    rate_lines=[]
    for c in FIN:
        f=fin[fin.feature.eq(c)].set_index('value')
        rate_lines.append(f'- **{c}:** dropout share is {f.loc[0,"dropout_pct"]:.1f}% for No (n={int(f.loc[0,"n"]):,}) and {f.loc[1,"dropout_pct"]:.1f}% for Yes (n={int(f.loc[1,"n"]):,}).')
    report=f'''# Person B: Exploratory Data Analysis Report

## 1. Purpose and research alignment
This EDA examines outcome imbalance, academic patterns, financial indicators and data quality. It supports the team's questions about early prediction and the incremental predictive value of academic and financial information. It does not train a classifier or rank feature groups by predictive performance.

## 2. Source, scope and reproducibility
The UCI dataset contains {len(d):,} student records, 36 input variables and Target. The three labels describe status at the end of the normal course duration. Enrolled is an unresolved eventual outcome, not a guarantee of future graduation.

A stratified {1-args.test_size:.0%}/{args.test_size:.0%} split with random_state={args.seed} assigns **{len(x):,} records to training and {len(te):,} to held-out testing**. All outcome-related figures and summaries below use training records only. No held-out performance is calculated. The input file structure and full-row duplicates are checked globally; these checks do not select predictors. `tables/split_assignments.csv` stores the shared row assignment and `run_metadata.json` records the source hash. The team must adopt this split, or regenerate this EDA with the team's agreed split before modeling.

Six second-semester fields are excluded from substantive exploration. The first-semester scenario includes earlier information plus six first-semester measurements. Availability dates of financial and administrative information remain unverified: results involving these fields are descriptive, conditional on later confirming availability.

## 3. Outcome distribution
{md(ct)}

![Training target distribution](../figures/01_target_distribution.png)

**Takeaway:** class frequencies differ. An always-Graduate classifier would have training accuracy of {counts['Graduate']/len(x):.1%}, without identifying any Dropout cases. This is a descriptive baseline, not a held-out model result. D should prioritize macro F1 and class-specific dropout recall/precision over accuracy alone.

## 4. Data quality and complete attribute profiling
No literal missing cells were found in the eligible training fields. The full CSV contains {int(d.duplicated().sum())} completely duplicated rows. These checks do not prove every value is valid or semantically complete. Parent qualification code 34 denotes Unknown in UCI documentation; preserve that distinction from a blank value. Other codes must be checked against the complete dictionary rather than assumed ordinal.

`tables/attribute_profile.csv` covers all 36 features: semantic type, feature group, time scope, missingness, cardinality, numeric range/skew or frequency-table reference, and preparation notes. Second-semester features are documented as excluded rather than having their values explored. Detailed category frequencies and numeric summaries accompany the profile.

The source header contains trailing whitespace in Daytime/evening attendance; the script strips header whitespace. Numeric codes for course, nationality, occupations and qualifications represent categories. Avoid Pearson correlations or continuous scaling of these codes as if the numeric distances were meaningful. Valid extreme ages or academic counts should not be dropped solely because they are boxplot outliers.

## 5. Academic factors
![Academic distributions](../figures/02_academic_by_outcome.png)

Training-group medians:
{md(med.round(2).reset_index())}

**Takeaway:** compare the group medians and overlapping distributions rather than treating a single low grade as a deterministic rule. First-semester grade and passed-unit counts show the academic separation visible in the figure; admission grade provides an earlier comparison. These are associations, with predictive value to be measured in CV by D/E.

![Age and approval rate](../figures/04_age_and_approval_rate.png)

**Takeaway:** approved/enrolled units is a candidate academic progress feature. {int(enrolled.eq(0).sum())} training records have zero enrolled units and therefore an undefined ratio; keep them explicit and let C handle them inside the model pipeline. Zero semester grades should not automatically become missing values, because they may reflect no successful assessments. Age distributions overlap across outcomes, so age alone is not an individual risk decision rule.

## 6. Financial indicators
![Financial outcome proportions](../figures/03_financial_outcome_rates.png)

{chr(10).join(rate_lines)}

**Takeaway:** these are within-group outcome percentages, with sample counts shown to avoid confusing larger groups with higher risk. `financial_dropout_rates.csv` includes descriptive 95% Wilson intervals for dropout shares. Small groups are less precise. Financial status may be related to academic performance, background and recording time. These comparisons do not establish causality, independently adjusted effects or which support intervention would work.

## 7. Distributions, context and correlations
![Numeric distributions](../figures/05_numeric_distributions.png)

**Takeaway:** inspect the actual shapes, zero concentrations and tails before choosing transformations. Robust scaling or standardization may help linear/distance-based models, but fit transformations inside training folds. Do not assume normality or remove tail observations automatically.

![Numeric correlations](../figures/06_numeric_correlations.png)

The strongest absolute numeric-feature correlations are:
{md(pairs.head(5).drop(columns='abs_rho').round(3))}

**Takeaway:** related academic measurements can carry overlapping information; individual feature importance may be shared across correlated predictors. Spearman correlation is descriptive and does not establish causality. Nominal category codes and Target are excluded from this numeric matrix. Course-level sample sizes and dropout shares are exported in `course_dropout_rates.csv`; rare courses require caution and numeric course codes have no ordered meaning.

## 8. Recommendations for C, D and E
- C: retain raw data; encode nominal categories; document unknown codes; construct approval rate with an explicit zero-denominator policy. Fit imputation, scaling and encoding within CV folds. Feature selection must not use the held-out test set.
- A/C: finalize a timing dictionary. Exclude second-semester features and qualify enrollment-only claims for variables with uncertain recording dates.
- D: use the shared split and identical stratified CV folds; include a majority dummy classifier and interpretable logistic regression. Report macro F1 mean/std and dropout precision/recall.
- E: compare Context, Context + Academic, Context + Financial, and Context + Academic + Financial using the same model and CV conditions. Academic-only vs financial-only importance plots cannot establish incremental value by themselves. Define Course and family background consistently as context; GDP/inflation/unemployment are macroeconomic context, not direct student finances.
- F: reserve test evaluation until model choices are fixed; discuss uncertain administrative timing, observational associations, unresolved Enrolled outcomes and transportability to other institutions.

## 9. Assumptions, limitations and useful additional data
This is descriptive analysis of a historical institutional dataset. The random split estimates an internal comparison; it does not demonstrate future-cohort or cross-university generalization. Date-stamped administrative records, cohort identifiers and dropout dates would help validate temporal availability and prevent including students whose dropout preceded an intended semester-end prediction. Household income, aid amounts, attendance, support-service usage and intervention outcomes would improve understanding but are not in this CSV. Do not infer that changing a correlated feature will change outcomes.

## 10. Conclusion
Academic and financial indicators provide testable hypotheses for the modeling team. The EDA establishes data structure, imbalance, descriptive group differences and preparation needs. Whether one group predicts better, or adds value to the other, remains a cross-validated modeling question.

## Reference
Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students’ Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89. Dataset licensed CC BY 4.0. Variable meanings and label horizon: {SOURCE}.
'''
    (ROOT/'reports/person_b_eda_report.md').write_text(report)
    print(json.dumps(metadata,indent=2))
    print('\nACADEMIC MEDIANS\n',med.to_string())
    print('\nFINANCIAL RATES\n',fin.to_string(index=False))

if __name__=='__main__': main()
