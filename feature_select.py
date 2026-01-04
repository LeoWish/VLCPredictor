import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import f_classif, mutual_info_classif, VarianceThreshold, SelectKBest, chi2
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from concurrent.futures import ProcessPoolExecutor


def _compute_scores_extended(X_var, y_train, random_state):
    """
    Compute fused scores using F-value, mutual info, chi2, and Random Forest feature importance.
    """
    X_sample, y_sample = resample(X_var, y_train, n_samples=int(0.8 * len(y_train)), random_state=random_state)

    F_values, _ = f_classif(X_sample, y_sample)
    MI_values = mutual_info_classif(X_sample, y_sample, random_state=random_state)
    chi2_values, _ = chi2(np.abs(X_sample), y_sample)

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    rf_importance = rf.feature_importances_

    def norm01(arr):
        return (arr - arr.min()) / (arr.ptp() + 1e-8)

    F_norm = norm01(F_values)
    MI_norm = norm01(MI_values)
    chi2_norm = norm01(chi2_values)
    rf_norm = norm01(rf_importance)

    fused_score = 0.3 * F_norm + 0.3 * MI_norm + 0.2 * chi2_norm + 0.2 * rf_norm
    return fused_score


def stability_clustering_feature_selection_v2(X_train, y_train, random_state_base, top_ratio=0.8, n_rounds=5,
                                              corr_threshold=None, prefilter_k=10000):
    """
    Improved stability-based and clustering-enhanced feature selection.
    """
    print("Performing advanced stability-based and clustering-enhanced feature selection...")

    vt = VarianceThreshold(threshold=1e-5)
    X_var = vt.fit_transform(X_train)
    retained_var_idx = vt.get_support(indices=True)

    skb = SelectKBest(score_func=f_classif, k=min(prefilter_k, X_var.shape[1]))
    X_kbest = skb.fit_transform(X_var, y_train)
    retained_kbest_idx = skb.get_support(indices=True)
    retained_indices = retained_var_idx[retained_kbest_idx]

    fusion_scores_accum = np.zeros(X_kbest.shape[1])
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_compute_scores_extended, X_kbest, y_train, random_state_base + i) for i in
                   range(n_rounds)]
        for f in tqdm(futures):
            fusion_scores_accum += f.result()

    fusion_scores_mean = fusion_scores_accum / n_rounds
    sorted_idx = np.argsort(-fusion_scores_mean)
    top_k = int(top_ratio * len(sorted_idx))
    selected_idx_in_kbest = sorted_idx[:top_k]
    selected_idx_in_train = retained_indices[selected_idx_in_kbest]

    selected_features_matrix = X_train[:, selected_idx_in_train]
    corr_matrix = np.corrcoef(selected_features_matrix.T)
    np.fill_diagonal(corr_matrix, 0)

    if corr_threshold is None:
        corr_threshold = np.quantile(np.abs(corr_matrix), 0.99)
    print(f"Using correlation threshold for clustering: {corr_threshold:.3f}")

    distance_matrix = 1 - np.abs(corr_matrix)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    linked = linkage(squareform(distance_matrix), method='average')
    cluster_assignments = fcluster(linked, t=corr_threshold, criterion='distance')

    final_selected = []
    for cluster_id in np.unique(cluster_assignments):
        cluster_members = np.where(cluster_assignments == cluster_id)[0]
        cluster_scores = fusion_scores_mean[selected_idx_in_kbest[cluster_members]]
        best_member = cluster_members[np.argmax(cluster_scores)]
        final_selected.append(selected_idx_in_train[best_member])

    print(f"Final selected features: {len(final_selected)} out of {X_train.shape[1]}")
    return final_selected