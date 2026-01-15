import numpy as np
import os
from scipy.signal import welch

RAW_EEG_DIR = "data/raw_eeg"

FEATURES = []
LABELS = []

# EEG frequency bands
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

label_map = {"YES": 0, "NO": 1, "HELP": 2, "WATER": 3}

def bandpower(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(psd[mask])

print("🧹 Extracting advanced EEG features...")

# Walk through directory safely
for root, dirs, files in os.walk(RAW_EEG_DIR):
    for file in files:
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(root, file)
        eeg = np.load(file_path)

        # Determine label from filename
        label_name = file.split("_")[0]
        label = label_map[label_name]

        feature_vector = []

        for channel in eeg:
            # Time-domain features
            mean = np.mean(channel)
            std = np.std(channel)
            var = np.var(channel)
            energy = np.sum(channel ** 2)

            # Frequency-domain features
            freqs, psd = welch(channel, fs=128)

            delta = bandpower(freqs, psd, BANDS["delta"])
            theta = bandpower(freqs, psd, BANDS["theta"])
            alpha = bandpower(freqs, psd, BANDS["alpha"])
            beta  = bandpower(freqs, psd, BANDS["beta"])

            feature_vector.extend([
                mean, std, var, energy,
                delta, theta, alpha, beta
            ])

        FEATURES.append(feature_vector)
        LABELS.append(label)

# Save features
os.makedirs("data", exist_ok=True)
np.save("data/features.npy", np.array(FEATURES))
np.save("data/labels.npy", np.array(LABELS))

print("✅ Feature extraction completed!")
print("📐 Feature shape:", np.array(FEATURES).shape)
