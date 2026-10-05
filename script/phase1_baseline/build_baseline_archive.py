"""
build_baseline_archive.py
=========================
Phase 1 of the adaptive HMM-LSTM framework for dry weather flow (DWF) state
recognition and prediction: construction of the initial baseline state archive.

Pipeline
--------
1. Load high-frequency sensor data.
2. Split the record into continuous segments at time gaps.
3. Clean each segment (rolling-IQR outlier removal + linear interpolation,
   with site-specific hard caps).
4. For each segment: standardize -> PCA -> fit a Gaussian HMM whose number of
   hidden states is selected by BIC.
5. Extract state posteriors, physical state means, and hour-of-day
   distributions.
6. Build a per-state feature vector (water-quality/flow means + time-PCA),
   weight it by sample count, and merge states across segments by Ward
   hierarchical clustering. The number of merged states is chosen from the
   elbow of the merge-distance curve.
7. Save the baseline archive and the intermediate tables used downstream
   (Phase 2 rolling update and the figure scripts).

Environment
-----------
dwf-hmm  (hmmlearn, scikit-learn==1.7.2); see requirements-hmm.txt.

Data
----
The input file ``data/outlet_phase1.csv`` is the sensor record after the
manual quality-assurance steps of the cleaning procedure; this script then
performs the final automated cleaning (rolling-IQR outlier removal, linear
interpolation, and site-specific hard caps). Set SKIP_CLEANING = True
when the input is already cleaned. See data/README.md for provenance and how to
obtain the file. Outputs are written to ./results.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration (edit paths / parameters here)
# ---------------------------------------------------------------------------
INPUT_CSV = os.path.join("data", "outlet_phase1.csv")   # input sensor data
OUTPUT_DIR = "results"                                       # output directory

FEATURE_NAMES = ["Temp", "NH3", "Turb", "Cond", "pH", "Flow"]

# Segmentation and cleaning
GAP_HOURS = 3                              # start a new segment when the gap exceeds this
MIN_DURATION = pd.Timedelta("3 days")      # discard segments shorter than this
IQR_WINDOW = 60                            # rolling window (samples) for IQR outlier flagging
IQR_WHISKER = 1.5
HARD_THRESHOLDS = {"Temp": 28, "NH3": 500, "Cond": 5000}  # site-specific physical caps
SKIP_CLEANING = False                      # set True when the input is already cleaned
                                           # (e.g. the HydroShare cleaned dataset)

# HMM model selection (per segment)
HMM_MIN_STATES = 2
HMM_MAX_STATES = 6
HMM_N_ITER = 500
HMM_TOL = 1e-4
PCA_COMPONENTS = 3
RANDOM_STATE = 0

# State merging (hierarchical clustering)
TIME_PCA_COMPONENTS = 3
MAX_CLUSTERS = 8                           # upper bound; actual k from the elbow


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def rolling_iqr_smoothing(df_seg, window=IQR_WINDOW, whisker=IQR_WHISKER):
    """Flag outliers per feature with a centered rolling IQR (plus hard caps)
    and replace them by linear interpolation."""
    df_smooth = df_seg.copy()
    actual_window = min(window, len(df_seg))
    if actual_window < 2:
        return df_smooth

    for col in FEATURE_NAMES:
        series = df_seg[col]
        roll = series.rolling(window=actual_window, center=True,
                              min_periods=max(1, actual_window // 2))
        q1, q3 = roll.quantile(0.25), roll.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - whisker * iqr, q3 + whisker * iqr
        outlier_mask = (series < lower) | (series > upper)
        if col in HARD_THRESHOLDS:
            outlier_mask |= series > HARD_THRESHOLDS[col]
        df_smooth[col] = series.mask(outlier_mask).interpolate(
            method="linear", limit_direction="both")
    return df_smooth


# ---------------------------------------------------------------------------
# HMM model selection
# ---------------------------------------------------------------------------
def select_best_hmm_model(x_pca, min_states=HMM_MIN_STATES,
                          max_states=HMM_MAX_STATES, seg_id=None):
    """Fit Gaussian HMMs over a range of state counts and return the one with
    the lowest BIC, together with a table of scores."""
    best_model, best_bic, scores = None, np.inf, []
    for n in range(min_states, max_states + 1):
        try:
            model = GaussianHMM(n_components=n, covariance_type="full",
                                n_iter=HMM_N_ITER, tol=HMM_TOL,
                                random_state=RANDOM_STATE, verbose=False)
            model.fit(x_pca)
            log_l = model.score(x_pca)
            n_params = n * n + 2 * n * x_pca.shape[1] - 1
            aic = -2 * log_l + 2 * n_params
            bic = -2 * log_l + n_params * np.log(x_pca.shape[0])
            scores.append((n, model.monitor_.converged, log_l, aic, bic))
            if bic < best_bic:
                best_bic, best_model = bic, model
            print(f"[Segment {seg_id}] states={n} | converged="
                  f"{model.monitor_.converged} | BIC={bic:.1f}")
        except Exception as exc:  # singular covariance, etc.
            scores.append((n, False, None, None, None))
            print(f"[Segment {seg_id}] states={n} failed: {exc}")

    df_scores = pd.DataFrame(scores,
                             columns=["States", "Converged", "LogL", "AIC", "BIC"])
    if best_model is None:
        print(f"[Segment {seg_id}] no HMM selected; segment skipped.")
    return best_model, df_scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}. Download the dataset from "
            f"HydroShare (see data/README.md) and place it under ./data.")

    # --- Load and segment ---
    df_raw = pd.read_csv(INPUT_CSV, parse_dates=["Datetime"])
    df = df_raw[["Datetime"] + FEATURE_NAMES].dropna()
    df = df[df["Flow"] != 0].reset_index(drop=True)  # drop zero-flow (no discharge)

    delta_h = df["Datetime"].diff().dt.total_seconds() / 3600.0
    df["Segment"] = (delta_h > GAP_HOURS).cumsum()

    posterior_list, state_means_list = [], []
    cleaned_list, model_scores = [], {}

    # --- Segment-wise HMM modelling ---
    for seg_id, group in df.groupby("Segment"):
        if group.empty:
            continue
        span = group["Datetime"].max() - group["Datetime"].min()
        if span < MIN_DURATION or len(group) < IQR_WINDOW:
            print(f"Segment {seg_id} skipped (span={span}, n={len(group)}).")
            continue
        print(f"\n--- Segment {seg_id} (span={span}) ---")

        cleaned = group.copy() if SKIP_CLEANING else rolling_iqr_smoothing(group)
        cleaned["Datetime"] = group["Datetime"].values
        cleaned_list.append(cleaned)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(cleaned[FEATURE_NAMES])
        n_comp = min(PCA_COMPONENTS, *x_scaled.shape)
        if n_comp < 1:
            continue
        pca = PCA(n_components=n_comp)
        x_pca = pca.fit_transform(x_scaled)

        model, df_scores = select_best_hmm_model(x_pca, seg_id=seg_id)
        model_scores[seg_id] = df_scores
        if model is None:
            continue

        _, posteriors = model.score_samples(x_pca)
        post_df = pd.DataFrame(
            posteriors, columns=[f"State_{i}" for i in range(model.n_components)])
        post_df["Datetime"] = cleaned["Datetime"].values
        post_df["Segment"] = seg_id
        post_df["PredictedState"] = post_df[
            [f"State_{i}" for i in range(model.n_components)]].idxmax(axis=1)
        posterior_list.append(post_df)

        means_original = scaler.inverse_transform(pca.inverse_transform(model.means_))
        state_df = pd.DataFrame(means_original, columns=FEATURE_NAMES)
        state_df["Segment"] = seg_id
        state_df["State"] = [f"State_{i}" for i in range(model.n_components)]
        state_means_list.append(state_df)

    if not posterior_list:
        print("No segment met the minimum-duration criterion. Nothing to save.")
        return

    # --- Save cleaned data and intermediate tables ---
    pd.concat(cleaned_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "outlet_phase1_cleaned.csv"),
        index=False)
    posterior_all = pd.concat(posterior_list, ignore_index=True)
    state_means_all = pd.concat(state_means_list, ignore_index=True)
    posterior_all.to_csv(
        os.path.join(OUTPUT_DIR, "posterior_all_for_clustering.csv"), index=False)
    state_means_all.to_csv(
        os.path.join(OUTPUT_DIR, "state_means_all_for_clustering.csv"), index=False)

    # --- Per-state feature vector (means + hour-of-day distribution) ---
    posterior_all["Hour"] = pd.to_datetime(posterior_all["Datetime"]).dt.hour
    time_dist = (posterior_all.groupby(["Segment", "PredictedState"])["Hour"]
                 .value_counts(normalize=True).unstack(fill_value=0))
    for h in range(24):
        if h not in time_dist.columns:
            time_dist[h] = 0
    time_dist = time_dist.reindex(sorted(time_dist.columns), axis=1)
    time_cols = [f"Hour_{i}" for i in range(24)]
    time_dist.columns = time_cols

    state_feat = pd.merge(state_means_all, time_dist, how="left",
                          left_on=["Segment", "State"], right_index=True)
    state_feat[time_cols] = state_feat[time_cols].fillna(0)

    counts = (posterior_all.groupby(["Segment", "PredictedState"])
              .size().reset_index(name="SampleCount"))
    state_feat = pd.merge(state_feat, counts, how="left",
                          left_on=["Segment", "State"],
                          right_on=["Segment", "PredictedState"])
    state_feat["SampleCount"] = state_feat["SampleCount"].fillna(0)
    weights = state_feat["SampleCount"] / state_feat["SampleCount"].max()
    weights[weights == 0] = 0.01  # keep a small time-feature contribution

    # --- Time-distribution PCA ---
    n_time_pca = min(TIME_PCA_COMPONENTS, len(state_feat), len(time_cols))
    time_pca_cols = []
    if n_time_pca >= 1:
        pca_time = PCA(n_components=n_time_pca)
        time_pca = pca_time.fit_transform(state_feat[time_cols])
        time_pca_cols = [f"Time_PC{i + 1}" for i in range(n_time_pca)]
        state_feat = pd.concat(
            [state_feat, pd.DataFrame(time_pca, columns=time_pca_cols,
                                      index=state_feat.index)], axis=1)

    # --- Standardize, weight the time dimensions, and cluster ---
    feature_cols = FEATURE_NAMES + time_pca_cols
    x = state_feat[feature_cols]
    x_scaled = StandardScaler().fit_transform(x)
    if time_pca_cols:
        idx = [x.columns.get_loc(c) for c in time_pca_cols]
        x_scaled[:, idx] *= weights.values[:, np.newaxis]

    if len(x_scaled) > 1:
        linked = linkage(x_scaled, method="ward")
        # Number of merged states from the elbow of the merge-distance curve
        merge_d = linked[:, 2]
        elbow_idx = int(np.argmax(np.diff(merge_d)))
        suggested_k = len(merge_d) - elbow_idx
        n_clusters = max(1, min(suggested_k, MAX_CLUSTERS, len(x_scaled)))
        state_feat["MergedState_final"] = fcluster(
            linked, t=n_clusters, criterion="maxclust")
        print(f"\nElbow-suggested k = {suggested_k}; using {n_clusters} "
              f"merged baseline states.")
    else:
        state_feat["MergedState_final"] = 1

    # --- Save the per-(segment, state) table with cluster labels ---
    state_feat.to_csv(
        os.path.join(OUTPUT_DIR, "state_features_and_merged_clusters.csv"),
        index=False)

    # --- Aggregate merged states into the phase-2 baseline archive ---
    # One row per merged baseline state: the mean of the per-state physical
    # means and hour-of-day distributions. All data use consistent units
    # (Flow in m3/h), so no unit conversion is applied.
    merged_means = state_feat.groupby("MergedState_final")[FEATURE_NAMES].mean().sort_index()
    merged_time = state_feat.groupby("MergedState_final")[time_cols].mean().sort_index()
    archive = pd.merge(merged_means, merged_time,
                       left_index=True, right_index=True).round(3)
    archive.to_csv(
        os.path.join(OUTPUT_DIR, "merged_state_features_physical_and_time.csv"))
    print(f"Baseline archive: {len(archive)} merged states -> "
          "merged_state_features_physical_and_time.csv")

    # --- Save the model-selection scores ---
    try:
        with pd.ExcelWriter(
                os.path.join(OUTPUT_DIR, "all_segment_model_scores.xlsx")) as writer:
            for seg_id, df_scores in model_scores.items():
                df_scores.to_excel(writer, sheet_name=f"Segment_{seg_id}",
                                   index=False)
    except Exception as exc:
        print(f"Could not write model-score workbook: {exc}")

    print("\nBaseline archive constructed. Outputs written to "
          f"'{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
