import sys
import os

# 🔧 FIX: Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import numpy as np
import joblib
from scipy.signal import welch

from alerts.email_alert import send_email_alert

# Load trained model
model = joblib.load("models/intent_model.pkl")

# EEG frequency bands
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

label_map = {0: "YES", 1: "NO", 2: "HELP", 3: "WATER"}

# -------- Feature extraction --------
def bandpower(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(psd[mask])

def extract_features(eeg):
    features = []

    for channel in eeg:
        mean = np.mean(channel)
        std = np.std(channel)
        var = np.var(channel)
        energy = np.sum(channel ** 2)

        freqs, psd = welch(channel, fs=128)

        delta = bandpower(freqs, psd, BANDS["delta"])
        theta = bandpower(freqs, psd, BANDS["theta"])
        alpha = bandpower(freqs, psd, BANDS["alpha"])
        beta  = bandpower(freqs, psd, BANDS["beta"])

        features.extend([
            mean, std, var, energy,
            delta, theta, alpha, beta
        ])

    return np.array(features).reshape(1, -1)

# -------- EEG simulation --------
def simulate_live_eeg(intent):
    time = np.linspace(0, 2, 256)

    intent_freqs = {
        "YES": [10, 3],
        "NO": [20, 6],
        "HELP": [6, 14],
        "WATER": [3, 18]
    }

    eeg = []
    for _ in range(8):
        signal = np.zeros(256)
        for f in intent_freqs[intent]:
            signal += np.sin(2 * np.pi * f * time)
        noise = np.random.normal(0, 0.3, 256)
        eeg.append(signal + noise)

    return np.array(eeg)

# -------- Streamlit UI --------
st.set_page_config(page_title="BCI Command Interface", layout="centered")

st.title("🧠 Brain-Computer Interface")
st.subheader("Command-Based Assistive System")

intent = st.selectbox(
    "Simulate Patient Thought:",
    ["YES", "NO", "HELP", "WATER"]
)

if st.button("🔍 Predict Command"):
    eeg = simulate_live_eeg(intent)
    features = extract_features(eeg)

    predicted_command = label_map[model.predict(features)[0]]
    st.success(f"🧾 Predicted Command: **{predicted_command}**")

    st.subheader("🛠️ Action Executed")

    if predicted_command in ["HELP", "WATER"]:
        result = send_email_alert(predicted_command)
        if "successfully" in result.lower():
            st.error(result) if predicted_command == "HELP" else st.success(result)
        else:
            st.warning(result)

    elif predicted_command == "YES":
        st.success("✅ Confirmation received.")

    elif predicted_command == "NO":
        st.info("❌ Request denied.")
