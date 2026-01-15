import numpy as np
import os

# Create folders
os.makedirs("data/raw_eeg", exist_ok=True)

# Simulation parameters
NUM_CHANNELS = 8
SAMPLES_PER_SIGNAL = 256
SAMPLING_RATE = 128
NUM_SAMPLES_PER_CLASS = 300

time = np.linspace(0, 2, SAMPLES_PER_SIGNAL)

# Intent definitions (distinct frequency patterns)
INTENTS = {
    "YES":    [10, 3],   # alpha + delta
    "NO":     [20, 6],   # beta + theta
    "HELP":   [6, 12],   # theta + alpha
    "WATER":  [3, 15],  # delta + beta
}

labels_map = {"YES": 0, "NO": 1, "HELP": 2, "WATER": 3}

all_labels = []

def generate_signal(freqs, channel_bias):
    signal = np.zeros(SAMPLES_PER_SIGNAL)

    for f in freqs:
        signal += np.sin(2 * np.pi * f * time)

    # Add noise
    noise = np.random.normal(0, 0.5, SAMPLES_PER_SIGNAL)

    return channel_bias * signal + noise

print("🧠 Generating high-quality EEG signals...")

for intent, freqs in INTENTS.items():
    intent_dir = f"data/raw_eeg/{intent}"
    os.makedirs(intent_dir, exist_ok=True)

    for i in range(NUM_SAMPLES_PER_CLASS):
        eeg = []

        for ch in range(NUM_CHANNELS):
            # Channel-specific behavior
            channel_bias = np.random.uniform(0.8, 1.5) if ch < 4 else np.random.uniform(0.3, 0.8)
            eeg.append(generate_signal(freqs, channel_bias))

        eeg = np.array(eeg)

        filename = f"{intent}_{i}.npy"
        np.save(os.path.join(intent_dir, filename), eeg)

        all_labels.append(labels_map[intent])

# Save labels
np.save("data/labels.npy", np.array(all_labels))

print("✅ EEG simulation completed successfully!")
print("📊 Samples per class:", NUM_SAMPLES_PER_CLASS)
print("📡 Channels:", NUM_CHANNELS)
print("💾 Data saved in data/raw_eeg/")
