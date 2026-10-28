# Held-out evaluation verification

Executed the repository evaluation script on 2026-10-28 using its saved model configuration and original dataset. No model parameters or thresholds were changed.

The three model-comparison rows and every ensemble confusion-matrix count match the README to its displayed precision. Ensemble accuracy is 0.7266, macro F1 is 0.6549, dropout precision is 0.7724, and dropout recall is 0.6690.

The evaluation script completed successfully, including its internal metric and row-integrity assertions. Software versions, input hashes, selected parameters, aggregate metrics, and error summaries accompany this report. Student-level predictions and row indices are retained locally and are not published in this repository. The original CSV-byte hash agrees with the EDA source hash (3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae).

To reproduce, run from the repository root:

```bash
python insy674_partb_result/part_f_results.py
```

This writes a local `part_f_results/` folder. This directory contains the archived outputs of the verified run. The evidence supports this internal held-out result; it does not establish future-cohort performance or intervention effectiveness.
