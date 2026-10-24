**TITLE**
A Robust Deep Learning Framework for Interpreting Physician Handwritten Prescriptions and Extracting Medical Data to Prevent Fraudulent Medical Claims

**Description**
This project implements a two-module AI framework for the health insurance domain. Module 1 digitizes handwritten physician prescriptions using Qwen2-VL, a vision-language deep learning model, extracting medication names, dosages, frequencies, durations, and physician instructions into structured JSON — run under FP16 mixed-precision inference so it can operate on commonly available GPUs. Module 2 is an explainable machine-learning fraud-detection pipeline trained on real, anonymized health insurance claim data, using Logistic Regression and CatBoost classifiers with SHAP and LIME to audit which claim-level features drive each fraud prediction. The two modules were developed and evaluated independently and are presented as complementary components toward a future integrated system: prescription digitization reduces manual data-entry error and ambiguity, while the fraud-detection module flags suspicious claims for human review.

**Dataset Information**
- *Prescription interpretation (Module 1):* Handwritten and computer-generated prescription images gathered from multiple healthcare sources, including the "Handwritten Medical Prescriptions Collection" (Kaggle, 2024), spanning a range of handwriting styles, prescription layouts, and image-quality conditions. Ground-truth annotations for key medical entities (drug names, doses, frequencies, physician-related data) were manually curated for evaluation.
- *Fraud detection (Module 2):* A real outpatient claims register obtained from Tawasul HealthCare TPA (Pvt.) Ltd., comprising 20,287 medical claim records across 48 financial, socio-demographic, and policy-related attributes, collected from a live operational insurance environment and anonymized for research use. The dataset is not publicly released but is available from the corresponding author on request.

**Code Overview**
The framework integrates two independent pipelines:
1. Prescription Interpretation (Qwen2-VL)
   - Image acquisition from scanned/photographed prescriptions, followed by pre-processing (noise reduction, contrast enhancement, scaling, binarization, skew correction, normalization) and text-side tokenization/normalization/lemmatization.
   - A Qwen2-VL vision encoder (ViT-based patch embeddings) extracts visual structure and handwriting strokes from the pre-processed image.
   - Vision-language fusion maps visual embeddings into the language-model embedding space via Q-Former-style compression and cross-attention, run under FP16 (half-precision) computation to reduce GPU memory footprint and increase throughput.
   - The Qwen2-VL multimodal decoder performs autoregressive decoding over the fused representation, deciphering handwriting, medical acronyms, and shorthand.
   - Medical normalization expands abbreviations (e.g., "bd" → "twice daily") and standardizes dosage notation and brand/generic drug names.
   - Output is emitted as structured JSON (medication names, dosages, frequency, duration, instructions).
2. Fraud Detection (LR / CatBoost)
   - Data pre-processing: date-field parsing, zero-fill for numeric fields, UNKNOWN-category fill for categoricals, exact-duplicate removal.
   - Feature engineering: Claim Utilization Ratio (Claimed / Sum Insured), Sanction Gap (Claimed − Approved), Payment Gap (Approved − Paid), Hospital Days.
   - Logistic Regression (`max_iter=1000`, L2 regularization, balanced class weighting, `random_state=42`) and CatBoost (`iterations=500`, `depth=6`, `learning_rate=0.05`, `loss_function='Logloss'`, `auto_class_weights='Balanced'`, default `l2_leaf_reg=3.0`, `random_seed=42`) trained on an eight-feature set.
   - Evaluation via confusion matrices, accuracy, precision, recall, specificity, F1-score, ROC-AUC, and five-fold stratified cross-validation.
   - Explainability via SHAP (TreeExplainer, global and per-claim, on CatBoost's raw margin output) and LIME (local surrogate explanations on CatBoost's predicted probability), plus a feature-correlation heatmap against the fraud label.

All fraud-detection analysis is implemented in Python using scikit-learn and CatBoost, with SHAP and LIME for explainability and Matplotlib/Seaborn for plotting. The prescription-interpretation module is built on the Qwen2-VL vision-language architecture with FP16 inference.

**Usage Instructions**
1. Prescription Interpretation
   - Place prescription images in a working directory and point the Qwen2-VL pipeline at it; the model is run with FP16 precision to reduce memory requirements on commonly available GPUs.
   - Run the extraction pipeline end to end; output is written as per-prescription JSON files containing the extracted medical fields.
2. Fraud Detection
   - Place the claims extract in the working directory (Excel format, as obtained from the TPA source).
   - Open and run the fraud-detection notebook (data cleaning → feature engineering → LR/CatBoost training → evaluation → SHAP/LIME) end to end in Jupyter/Colab.
3. Outputs Generated
   - Structured JSON records of extracted prescription data
   - Cleaned, feature-engineered claims dataset with fraud labels
   - Classification reports and confusion matrices for LR and CatBoost
   - Five-fold cross-validation performance tables
   - SHAP summary/force plots and LIME local explanation plots
   - Feature-correlation heatmap

**Requirements**
| Library | Version (Recommended) |
|----------|-----------------------|
| Python | ≥ 3.9 |
| transformers | ≥ 4.45 (Qwen2-VL support) |
| torch | ≥ 2.1 (GPU build, FP16-capable) |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.25 |
| scikit-learn | ≥ 1.3 |
| catboost | ≥ 1.2 |
| shap | ≥ 0.44 |
| lime | ≥ 0.2 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.12 |
| xlrd | ≥ 2.0 |

Install dependencies:
```bash
pip install transformers torch pandas numpy scikit-learn catboost shap lime matplotlib seaborn xlrd
```

**Methodology Summary**
| Component | Description |
|------------|-------------|
| Prescription Module | Qwen2-VL vision-language model, FP16 mixed-precision inference, Q-Former-style vision-language fusion, autoregressive multimodal decoding |
| Fraud Module | Logistic Regression, CatBoost |
| Fraud Training Strategy | Supervised classification on a documented rule-derived fraud label, with balanced class weighting |
| Fraud Preprocessing | Date parsing, zero/UNKNOWN imputation, exact-duplicate removal, engineered financial/temporal features |
| Fraud Dataset Split | 80% training / 20% testing, stratified |
| Fraud Evaluation Metrics | Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC |
| Fraud Robustness Analysis | Five-fold stratified cross-validation |
| Interpretability | SHAP (global and local) and LIME, for the fraud-detection module |

**Performance Summary**
| Module | Metric | Result | Notes |
|--------|--------|--------|-------|
| Qwen2-VL Prescription Interpretation | Reported accuracy | 95.00% | Evaluated qualitatively/quantitatively on prescription samples of varying handwriting style and image quality; reported as an independent-module result, not integrated with the fraud-detection output |
| Logistic Regression (fraud, test set) | Accuracy / Precision / Recall / F1 (fraud class) | 96.01% / 0.17 / 0.85 / 0.29 | Interpretable linear baseline, high recall, lower precision |
| CatBoost (fraud, test set) | Accuracy / Precision / Recall / F1 (fraud class) | 99.68% / 0.75 / 1.00 / 0.86 | No fraudulent case missed on the held-out test set; far fewer false alarms than LR |
| CatBoost (5-fold CV) | Mean accuracy / mean ROC-AUC | 99.69% ± 0.06% / 99.93% ± 0.02% | — |
| Logistic Regression (5-fold CV) | Mean accuracy / mean ROC-AUC | 96.40% ± 0.33% / 97.66% ± 1.36% | — |

**Visualization Examples**
- Fig. Extraction Pipeline — Qwen2-VL framework for turning a prescription image into structured medical data
- Fig. Medical Data Extraction — example JSON output (drug names, dosage, frequency, duration)
- Fig. Multimodal Transformer handling variability in handwriting style and image quality
- Confusion Matrices — training and testing class-wise breakdown for LR and CatBoost
- SHAP Summary and Force Plots — global and per-claim feature attribution on CatBoost's raw score
- LIME Local Explanation Plot — model-agnostic cross-check on CatBoost's predicted probability
- Feature-Correlation Heatmap — correlation of each engineered feature with the fraud label

**Evaluation Environment**
Developed and tested on:
- OS: Windows 10 / Linux
- Platform: Google Colab (GPU-accelerated for Qwen2-VL FP16 inference)
- Frameworks: Transformers/PyTorch (Qwen2-VL), scikit-learn and CatBoost (fraud detection)
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, SHAP, LIME

**Results and Discussion**
- Qwen2-VL accurately interpreted handwritten prescriptions across a range of handwriting styles (cursive, irregular, abbreviated medical terminology) and low-quality scans, outperforming traditional OCR-based approaches in qualitative comparisons reported in the study, with an overall reported accuracy of 95.00%.
- CatBoost outperformed Logistic Regression on every reported fraud-detection metric, both on the held-out test set and under five-fold cross-validation, reflecting its ability to model non-linear relationships (e.g., same-day admission/discharge patterns) that a linear model cannot capture as cleanly.
- Logistic Regression's lower precision (0.17 on the test set) indicates that, even under balanced class weighting, a linear combination of the eight features cannot separate the fraud class without a high false-positive cost.
- SHAP and LIME attribution for CatBoost both highlighted Claim Utilization Ratio and Sanction Gap as dominant contributors to the fraud prediction, consistent with their correlation to the fraud label.
- The two modules were developed and evaluated independently; the manuscript's own comparison table notes this explicitly — the 95.00% (Qwen2-VL) and 96.01%/99.68% (LR/CatBoost) figures are not combined into a single end-to-end pipeline metric.

**Limitations**
- The two modules are not yet integrated into a single pipeline; prescription-extraction output does not currently feed the fraud-detection features, so the framework's full fraud-prevention value chain (prescription → structured data → fraud flag) is not end-to-end validated.
- The fraud label is a documented, rule-based heuristic rather than an investigator-adjudicated outcome, and several of the fraud model's engineered features are constitutive of that same rule, which constrains claims of independently discovered fraud signal beyond rule recovery.
- Prescription-interpretation evaluation was performed on sample images with varying handwriting/quality rather than a held-out benchmark with standardized metrics (e.g., Word/Character Error Rate) reported for this study's own model.
- All claims data originate from a single TPA and policy year, limiting generalizability to other insurers or claim types.
- Hyperparameters for both modules (CatBoost depth/learning rate/iterations, Logistic Regression max_iter, Qwen2-VL inference settings) were fixed rather than tuned via systematic search.
- Future work, as stated in the study, includes integrating both modules into a single operational pipeline, expanding the prescription dataset for better generalization, and pursuing a decoupled-feature protocol and investigator-verified labels for the fraud module.

**Conclusion**
The proposed dual-module framework delivers:
- A Qwen2-VL–based prescription-interpretation pipeline that converts handwritten prescriptions into structured, machine-readable JSON with a reported 95.00% accuracy, using FP16 inference for deployability on commonly available GPUs
- A fraud-detection pipeline (Logistic Regression + CatBoost) achieving up to 99.68% test accuracy and 0.86 F1-score on the fraud class, validated by five-fold stratified cross-validation
- Dual explainability (SHAP and LIME) for the fraud-detection module, and transparent, human-readable JSON output for the prescription module
- A disclosed, honest framing of both modules as independently evaluated components, with explicit next steps toward a single integrated, investigator-validated system

This framework represents a transparent first stage toward an end-to-end AI pipeline for digitizing physician handwriting and triaging health insurance claims fraud, with a clear, stated path — pipeline integration, expanded datasets, and verified labels — toward a validated, deployable system.
