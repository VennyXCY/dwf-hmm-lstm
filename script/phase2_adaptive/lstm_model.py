"""
lstm_model.py   (environment: dwf-ann)
=====================================
Teacher-student LSTM: a two-input network (sensor window + HMM prior) trained to
reproduce the HMM's standardized posterior. Architecture and training
configuration match the paper.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import config


def set_seeds(seed=config.SEED):
    """Fix all seeds for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_model(n_states, window_size, n_features):
    """Two-input model: LSTM(32) over the sensor window, Dense(16) over the HMM
    prior, concatenated -> Dense(16) -> softmax over the states."""
    input_seq = Input(shape=(window_size, n_features), name="Sensor_Window")
    lstm_out = LSTM(32, activation="relu", return_sequences=False)(input_seq)

    input_prior = Input(shape=(n_states,), name="HMM_Prior")
    prior_dense = Dense(16, activation="relu")(input_prior)

    concat = Concatenate()([lstm_out, prior_dense])
    dense_out = Dense(16, activation="relu")(concat)
    output = Dense(n_states, activation="softmax", name="HMM_Posterior_Proxy")(dense_out)

    model = Model(inputs=[input_seq, input_prior], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy", "kullback_leibler_divergence"])
    return model


def train_lstm(X_seq, R_prior, Y_target, n_states, window_size, n_features):
    """Train the LSTM (80/20 split, EarlyStopping + ReduceLROnPlateau)."""
    Xtr, Xva, Rtr, Rva, Ytr, Yva = train_test_split(
        X_seq, R_prior, Y_target, test_size=0.2, random_state=config.SEED, shuffle=True)
    model = build_lstm_model(n_states, window_size, n_features)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5,
                          min_lr=1e-6, verbose=1),
    ]
    history = model.fit([Xtr, Rtr], Ytr,
                        validation_data=([Xva, Rva], Yva),
                        epochs=100, batch_size=32, callbacks=callbacks, verbose=1)
    return model, history
