"""
part2_lstm_training.py   (environment: dwf-ann)
==============================================
Stage 2 of an update cycle: train the teacher-student LSTM on the data produced
by part1, and save the model.

Output: results/HMM_LSTM_Model_<version>/lstm_model.h5
Run:  python script/phase2_adaptive/part2_lstm_training.py
"""

import os
import numpy as np

import config
import lstm_model as lm


def run(cfg):
    outdir = config.cycle_dir(cfg["version"])
    ann = os.path.join(outdir, "ann_training_data")
    X_seq = np.load(os.path.join(ann, "X_seq_data.npy"))
    R_prior = np.load(os.path.join(ann, "R_prior_data.npy"))
    Y = np.load(os.path.join(ann, "Y_target_data.npy"))

    lm.set_seeds()
    model, _ = lm.train_lstm(
        X_seq, R_prior, Y, cfg["n_states"], config.WINDOW_SIZE, config.N_FEATURES)

    model_path = os.path.join(outdir, "lstm_model.h5")
    model.save(model_path)
    print(f"Stage 2 done -> {model_path}")
    print("Next: run part3_rolling_inference.py in the dwf-ann environment.")


if __name__ == "__main__":
    run(config.CYCLES[config.ACTIVE_CYCLE])
