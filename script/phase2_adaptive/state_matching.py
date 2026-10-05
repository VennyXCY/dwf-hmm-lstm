"""
state_matching.py   (environment: dwf-hmm)
=========================================
Match the local HMM states to the baseline archive using a combined
water-quality/flow + time-of-day cost matrix (weight alpha), solved with the
Hungarian algorithm, then re-index (standardize) the HMM to the archive's state
order. Matches whose confidence falls below the threshold are flagged as
Evolved States.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from hmmlearn import hmm

import config


def load_standard_archive(archive_csv, n_states):
    """Load the baseline archive: one row per baseline state, with the six
    physical feature means and the 24 hour-of-day frequencies."""
    df = pd.read_csv(archive_csv)
    time_cols = [f"Hour_{i}" for i in range(config.N_HOURS)]
    wq = df[config.FEATURE_COLS].values
    tm = df[time_cols].values
    if wq.shape != (n_states, config.N_FEATURES) or tm.shape != (n_states, config.N_HOURS):
        raise ValueError(f"Archive shape mismatch for n_states={n_states}: "
                         f"wq {wq.shape}, time {tm.shape}")
    return wq, tm


def get_state_mapping_combined(local_means, standard_means, local_time,
                               standard_time, alpha, n_states):
    """Return mapping[local_state] = archive_state and the combined distance
    matrix. Physical means are min-max normalized before comparison, and the two
    distance matrices are min-max normalized before the weighted combination."""
    sc = MinMaxScaler()
    ln = sc.fit_transform(local_means)
    bn = sc.transform(standard_means)
    d_wq = cdist(ln, bn, "euclidean")
    d_time = cdist(local_time, standard_time, "euclidean")
    d_total = (alpha * MinMaxScaler().fit_transform(d_wq)
               + (1 - alpha) * MinMaxScaler().fit_transform(d_time))
    row, col = linear_sum_assignment(d_total)
    mapping = np.zeros(n_states, dtype=int)
    mapping[row] = col
    return mapping, d_total


def check_matching_confidence(dist_matrix, mapping, threshold=config.CONFIDENCE_THRESHOLD):
    """Confidence = 1 / (1 + distance). Return ({archive_state: confidence} for
    matches below the threshold = the Evolved States, mean confidence over all
    matched pairs)."""
    evolved = {}
    confs = []
    for local_state, std_state in enumerate(mapping):
        conf = 1.0 / (1.0 + dist_matrix[local_state, std_state])
        confs.append(conf)
        if conf < threshold:
            evolved[int(std_state)] = float(conf)
    return evolved, float(np.mean(confs))


def standardize_hmm(local_model, mapping, n_states):
    """Re-index the local HMM so its state order matches the archive."""
    std = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                          random_state=config.SEED)
    std.startprob_ = local_model.startprob_[mapping]
    std.transmat_ = local_model.transmat_[mapping][:, mapping]
    std.means_ = local_model.means_[mapping]
    std.covars_ = local_model.covars_[mapping]
    return std
