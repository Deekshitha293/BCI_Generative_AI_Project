import numpy as np
import joblib
from scipy.signal import welch

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

def bandpower(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(psd[mask])

def extract_features(eeg):
    feature_vector = []

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

        feature_vector.extend([
            mean, std, var, energy,
            delta, theta, alpha, beta
        ])

    return np.array(feature_vector).reshape(1, -1)

# -------- Simulate live EEG input --------
def simulate_live_eeg(intent="HELP"):
    time = np.linspace(0, 2, 256)

    intent_freqs = {
        "YES": [10, 3],
        "NO": [20, 6],
        "HELP": [6, 12],
        "WATER": [3, 15]
    }

    eeg = []
    for ch in range(8):
        signal = np.zeros(256)
        for f in intent_freqs[intent]:
            signal += np.sin(2 * np.pi * f * time)

        noise = np.random.normal(0, 0.5, 256)
        eeg.append(signal + noise)

    return np.array(eeg)

# -------- Run prediction --------
if __name__ == "__main__":
    print("🧠 Simulating live EEG input...")

    eeg = simulate_live_eeg(intent="HELP")  # Change intent to test
    features = extract_features(eeg)

    prediction = model.predict(features)[0]

    print("🧾 Predicted Command:", label_map[prediction])
