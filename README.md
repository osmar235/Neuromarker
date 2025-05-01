# Grip Force Variability and Deep Learning for Parkinson's Disease Classification

This repository contains the full code and supplementary material for the manuscript:

**"Grip Force Variability and Deep Learning: Differentiating Parkinson’s Disease Patients from Healthy Older Adults."**

---

## 📄 Summary

We propose a deep learning approach to classify individuals with Parkinson’s Disease (PD) based on grip force variability measures. This repository includes:

- Preprocessing of raw data
- Stratified data splitting and StandardScaler fitting (no leakage)
- SMOTE oversampling during training folds
- GridSearchCV for hyperparameter optimization
- Final training with early stopping
- Evaluation using accuracy, AUC, sensitivity, specificity, etc.
- Permutation feature importance analysis with 95% bootstrap confidence intervals
- Comparison with classic ML models (SVM, RF, GBM, Stacking)

---

## 🧪 Reproducibility
All code is implemented in Python using TensorFlow, SciKeras, and Scikit-learn. Training logs, model performance, ROC curve, and feature importance data are exported in both `.csv` and `.png` formats.

## 📊 Requirements
Install all dependencies using:
pip install -r requirements.txt


## 📁 Contents

| File | Description |
|------|-------------|
| `main_analysis.ipynb` | Main notebook with DNN training, evaluation, and permutation importance |
| `roc_curve_test.png` | ROC curve plot on independent test set |
| `dnn_significant_feature_importance.png` | Significant features from permutation importance |
| `training_validation_loss_with_early_stopping.png` | Loss curve with early stopping |
| `training_history.csv` / `training_history.json` | Full training logs from SciKeras |
| `roc_test_curve.csv` | Raw FPR/TPR/threshold values |
| `feature_table.csv` | Feature matrix + encoded target |
| `perm_importance_annotated.png` | Annotated plot of all feature importances |

An anonymized dataset (`feature_table.csv`) is included for replication purposes. The data was collected using low-intensity grip force tests from a cohort of PD and HOA participants.


---

## 🧠 Dependencies

Tested with:
- `Python 3.10+`
- `TensorFlow 2.15.0`
- `Scikit-learn 1.3.2`
- `SciKeras 0.12.0`
- `Imbalanced-learn`
- `Matplotlib`, `Pandas`, `NumPy`

To install dependencies:

```bash
pip install -r requirements.txt
📊 How to Run
Clone the repo

Place your .csv data file inside the working directory

Update the FILE_PATH in the notebook

Run the cells in order

For reproducibility, random seeds are set for NumPy, Python, and TensorFlow.

📌 Notes
Data was scaled after splitting to prevent data leakage.

SMOTE was applied only to training data in each fold.

Feature importance is based on permutation with ROC AUC as the scoring metric.

Bootstrap CI was computed with 1,000 iterations.

## 📈 Results
Final DNN accuracy: **91.0%**  
Final ROC AUC: **92.2%**

📂 Citation
If you use this code, please cite the associated paper (once published).


## 📜 License
This project is licensed under the MIT License.

## 📬 Contact
For questions or collaborations, please contact Osmar Pinto Neto at osmarpintoneto@hotmail.com or osmar@csusm.edu

