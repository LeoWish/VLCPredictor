import os
import numpy as np
import pandas as pd
import torch
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import from custom modules
from feature_select import stability_clustering_feature_selection_v2
from models import get_models, evaluate_model

warnings.filterwarnings("ignore")

# Constants
RANDOM_STATE = 42
RANDOM_STATE_DT = 3


def load_data(label_path, kmer_folder):
    """
    Load label and k-mer features from files.
    """
    label_df = pd.read_excel(label_path, sheet_name="Sheet1", index_col=0)
    y = np.array(label_df['label'])

    kmer_data = []
    print("Loading k-mer feature files...")
    file_list = [f for f in os.listdir(kmer_folder) if f.endswith('.csv')]

    for file in tqdm(file_list):
        file_path_k = os.path.join(kmer_folder, file)
        data = pd.read_csv(file_path_k)
        kmer_data.append(data.T)

    combined_kmer = np.array(pd.concat(kmer_data, axis=1)[1:])
    return combined_kmer, y


def main():
    # Set seeds
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    label_path = './data/label_67.xlsx'
    kmer_folder = './data/kmer_67'

    X_raw, y = load_data(label_path, kmer_folder)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.4, random_state=RANDOM_STATE_DT, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Feature Selection
    selected_features = stability_clustering_feature_selection_v2(
        X_train_scaled, y_train, RANDOM_STATE
    )

    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]

    # Model Training and Evaluation
    models = get_models(RANDOM_STATE)

    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        model.fit(X_train_selected, y_train)
        evaluate_model(model, X_test_selected, y_test)


if __name__ == "__main__":
    main()