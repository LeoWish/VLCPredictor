VCLPredictor (K-mer Based Sequence Classification Pipeline)
This project provides a robust machine learning pipeline for classifying biological sequences using k-mer features. It features an advanced feature selection strategy combining statistical scores, stability-based importance, and hierarchical clustering to handle high-dimensional data.
Project Structure
The code is organized into three main modules:
feature_select.py: Contains the stability_clustering_feature_selection_v2 function. It performs variance filtering, statistical pre-filtering (SelectKBest), stability-based fused scoring (F-Value, Mutual Information, Chi-Square, and Random Forest Importance), and redundancy reduction via hierarchical clustering.
models.py: Defines the dictionary of machine learning classifiers (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, etc.) and the evaluation metrics logic.
prediction.py: The main entry point. It handles data loading, preprocessing (scaling), orchestrates the feature selection, and executes the training/evaluation loop.

Workflow Description
1.Data Loading: Loads labels from an Excel file and k-mer frequency matrices from multiple CSV files.
2.Feature Selection:
Variance Threshold: Removes near-constant features.
Prefilter: Uses f_classif to narrow down the feature space.
Stability Scoring: Computes a weighted fusion of multiple importance metrics over several bootstrap rounds.
Clustering: Groups correlated features using hierarchical clustering and selects the best representative from each cluster.
3.Model Training: Trains 8 different classifiers on the optimized feature set.
4.Evaluation: Outputs Accuracy, Precision, Recall, F1-Score, Specificity, and AUC for each model.

Installation
Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:
Bash
pip install numpy pandas scikit-learn scipy xgboost lightgbm tqdm openpyxl torch
Data Requirements
The script expects the following directory structure:
Plaintext
.
├── data/
│   ├── label_67.xlsx        # Excel file with an 'index' and 'label' column
│   └── kmer_67/             # Folder containing .csv files (features as rows)
├── feature_select.py
├── models.py
└── prediction.py
Usage
To run the entire pipeline, simply execute the prediction.py script:
Bash
python prediction.py
Performance Metrics
The pipeline automatically calculates and prints the following metrics for every model:
AUC (Area Under Curve)
Accuracy
Precision / Recall / F1-Score
Specificity

Configuration
You can adjust the following parameters inside prediction.py:
test_size: Ratio of the dataset used for testing (default is 0.4).
top_ratio: Percentage of top features to keep before clustering.
n_rounds: Number of bootstrap rounds for stability scoring.
