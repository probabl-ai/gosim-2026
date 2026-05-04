# Claude Live Demo — Pre-written Prompts

**Context:** These prompts are designed for the live demo portion of the
GOSIM presentation (Act 7). They assume the notebook has already been run
up to Act 6 and the variables `X`, `y`, `X_train`, `X_test`, `y_train`,
`y_test`, `report_gb`, `report_rf`, `report_lr` are available in the
kernel. (The train/test split happens in Act 1.)

**Tip:** Copy-paste these prompts into the Claude app. Each one should
produce a clean, working response in ~10–15 seconds.

---

## Prompt 1 — Evaluate a new model with skore (~30 seconds)

Purpose: Show that Claude understands skore's API and generates proper
evaluation code — not just training code.

```
Using skore, create a CrossValidationReport for an SVM classifier
(sklearn.svm.SVC with probability=True) on the wine training data.
Use X_train and y_train as the data, 5-fold cross-validation, and pos_label="top".
Then show the summary metrics and the ROC curve.
Use skrub's tabular_pipeline for preprocessing.
```

**Expected output:** Claude generates something like:

```python
from sklearn.svm import SVC

pipe_svm = tabular_pipeline(SVC(probability=True, random_state=42))
report_svm = CrossValidationReport(pipe_svm, X=X_train, y=y_train, splitter=5, pos_label="top")

display = report_svm.metrics.summarize()
display.frame()
```

```python
display = report_svm.metrics.roc()
display.plot()
```

---

## Prompt 2 — Diagnose and compare (~30 seconds)

Purpose: Show the scientific guardrail angle — Claude doesn't just train,
it evaluates properly.

```
Now compare this SVM with the three models we already trained
(report_gb, report_rf, report_lr) using skore's ComparisonReport.
Also run diagnose() on the SVM to check for any issues.
```

**Expected output:**

```python
comparison_with_svm = ComparisonReport(
    reports=[report_gb, report_rf, report_lr, report_svm]
)

display = comparison_with_svm.metrics.summarize()
display.frame()
```

```python
report_svm.diagnose()
```

---

## Prompt 3 — Push to the hub (~20 seconds)

Purpose: Show the full loop — from question to hub in one conversation.

```
Create an EstimatorReport for the SVM on the test set
(with X_train, y_train, X_test, y_test, pos_label="top"),
compute its permutation importance,
and push it to the skore hub project "gosim/digital-sommelier-final".
```

**Expected output:**

```python
pipe_svm = tabular_pipeline(SVC(probability=True, random_state=42))
pipe_svm.fit(X_train, y_train)

er_svm = EstimatorReport(
    pipe_svm,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    pos_label="top",
)
er_svm.inspection.permutation_importance()

project = skore.Project(name="gosim/digital-sommelier-final", mode="hub")
project.put("SVM", er_svm)
```

---

## Talking points after the demo

- "See? Claude understands skore's API. It generates proper evaluation
  code — CrossValidationReport, diagnose, ComparisonReport — not just
  model.fit() and model.predict()."

- "AI tools help you code faster. skore makes sure that faster code is
  also *correct* code. The two are complementary."

- "And because skore's API is structured and consistent, AI tools can
  reason about it reliably. This is what happens when you design a
  library for the data science workflow rather than just wrapping
  scikit-learn."

---

## Fallback

If the live demo has connectivity or timing issues, you can say:

> "I had a live demo prepared, but let me show you the output instead —
> I generated this earlier with Claude."

Then switch to a pre-rendered cell in the notebook that shows the SVM
evaluation. (Consider adding a hidden "backup" section at the bottom of
the notebook with pre-rendered SVM results.)
