"""
part1_hmm_matching.py   (environment: dwf-hmm)
=============================================
Stage 1 of an update cycle: fit the local HMM on the training window, match it
to the baseline archive, standardize it, and generate the LSTM training data.

Outputs (to results/HMM_LSTM_Model_<version>/):
    ann_training_data/{X_seq_data,R_prior_data,Y_target_data}.npy
    transmat.npy, scaler.joblib, evolved_map.joblib

Set the target cycle in config.ACTIVE_CYCLE, then run part1 -> part2 -> part3.
Run:  python script/phase2_adaptive/part1_hmm_matching.py
"""

import os
import numpy as np
import joblib

import config
import online_hmm as oh
import state_matching as sm


def run(cfg):
    outdir = config.cycle_dir(cfg["version"])
    ns = cfg["n_states"]

    # 1. training-window data + local standardizer
    df_std, df_phys, scaler = oh.load_and_prep_window(
        cfg["data_csv"], config.FEATURE_COLS, cfg["train_start"], cfg["train_end"])
    X, lengths = oh.get_data_chunks(df_std[config.FEATURE_COLS])

    # 2. local HMM + its physical fingerprints
    local = oh.fit_local_hmm(X, lengths, ns)
    local_means_phys = scaler.inverse_transform(local.means_)
    seq = local.predict(X, lengths)
    local_time = oh.extract_local_time_dists(df_phys, seq, ns)

    # 3. match to the archive + confidence
    arch_wq, arch_time = sm.load_standard_archive(cfg["archive_csv"], ns)
    mapping, dist = sm.get_state_mapping_combined(
        local_means_phys, arch_wq, local_time, arch_time, config.MATCHING_ALPHA, ns)
    evolved, conf_mean = sm.check_matching_confidence(dist, mapping)
    print(f"[{cfg['version']}] mapping (local->archive): {mapping.tolist()}")
    print(f"[{cfg['version']}] evolved states: {evolved} | mean confidence: {conf_mean:.3f}")

    # 4. standardize HMM + generate LSTM training data
    std_hmm = sm.standardize_hmm(local, mapping, ns)
    X_seq, R_prior, Y = oh.generate_ann_training_data(
        X, lengths, std_hmm, config.WINDOW_SIZE)

    # 5. save artifacts for stages 2 and 3
    ann = os.path.join(outdir, "ann_training_data")
    np.save(os.path.join(ann, "X_seq_data.npy"), X_seq)
    np.save(os.path.join(ann, "R_prior_data.npy"), R_prior)
    np.save(os.path.join(ann, "Y_target_data.npy"), Y)
    np.save(os.path.join(outdir, "transmat.npy"), std_hmm.transmat_)
    joblib.dump(scaler, os.path.join(outdir, "scaler.joblib"))
    joblib.dump(evolved, os.path.join(outdir, "evolved_map.joblib"))
    joblib.dump({"n_states": ns, "conf_mean_raw": conf_mean,
                 "baseline_used": os.path.basename(cfg["archive_csv"])},
                os.path.join(outdir, "cycle_meta.joblib"))

    print(f"Stage 1 done ({len(X_seq)} samples) -> {outdir}")
    print("Next: run part2_lstm_training.py in the dwf-ann environment.")


if __name__ == "__main__":
    run(config.CYCLES[config.ACTIVE_CYCLE])
