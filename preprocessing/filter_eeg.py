import numpy as np
from scipy.signal import butter, filtfilt
import os

INPUT_DIR = "data/raw_eeg"
OUTPUT_DIR = "data/clean_eeg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filter settings
LOWCUT = 1.0
HIGHCUT = 40.0
FS = 256  # Sampling rate

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

# Process all EEG files
for file in os.listdir(INPUT_DIR):
    if file.endswith(".npy"):
        eeg = np.load(os.path.join(INPUT_DIR, file))
        filtered_eeg = np.zeros_like(eeg)

        for ch in range(eeg.shape[0]):
            filtered_eeg[ch] = bandpass_filter(
                eeg[ch], LOWCUT, HIGHCUT, FS
            )

        np.save(os.path.join(OUTPUT_DIR, file), filtered_eeg)

print("✅ EEG filtering completed successfully!")

