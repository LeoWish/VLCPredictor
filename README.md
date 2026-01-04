# VLCPredictor

VLCPredictor is a k-mer–based strain-level genomic sequence classification pipeline designed for predicting the functional potential of probiotic candidate strains. The framework integrates annotation-independent genomic representation, multi-stage feature selection, and parallel machine learning model evaluation, and is particularly suited for small-sample, high-dimensional whole-genome datasets.

VLCPredictor was developed to support strain-level probiotic prioritization, rather than community-level microbiome or metagenomic analysis.

The core features of the framework include: multi-order k-mer representation of whole-genome sequences; a multi-round feature selection strategy combining statistical filtering, stability-aware feature scoring, and redundancy reduction; parallel evaluation of multiple machine learning algorithms; and reproducible performance assessment using repeated internal validation.

The repository is structured as follows:

VLCPredictor/
feature_select.py  
models.py  
prediction.py  
data/  
└── label_67.xlsx  
└── kmer_67/  
  └── *.csv  
README.md  

The file feature_select.py implements the stability-based multi-stage feature selection procedure, including variance threshold filtering, statistical pre-filtering, stability score fusion across multiple feature importance metrics, and correlation-based redundancy reduction. The file models.py defines the machine learning models used in the framework, including Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), Naïve Bayes (NB), k-Nearest Neighbor (KNN), Decision Tree (DT), XGBoost (XGB), and LightGBM (LGBM). The file prediction.py serves as the main entry point of the pipeline and orchestrates data loading, feature selection, model training, and performance evaluation.

The workflow of VLCPredictor consists of loading phenotypic labels and k-mer feature matrices, applying multi-stage feature selection to identify informative and non-redundant genomic signals, training multiple machine learning classifiers in parallel, and evaluating predictive performance using standard classification metrics.

The pipeline requires Python version 3.8 or later. Required dependencies can be installed using pip:

pip install numpy pandas scikit-learn scipy xgboost lightgbm tqdm openpyxl torch

Input data should follow the expected directory structure. The label file should be an Excel file containing sample identifiers and corresponding binary labels. The k-mer feature directory should contain CSV files with k-mer frequency matrices generated from whole-genome sequences.

To run the full VLCPredictor pipeline, execute:

python prediction.py

The script will automatically perform feature selection, train all specified classifiers, and report performance metrics including AUC, accuracy, precision, recall, F1-score, and specificity.

Key configuration parameters such as test set proportion, feature retention ratio, and the number of stability selection rounds can be adjusted directly within prediction.py.
