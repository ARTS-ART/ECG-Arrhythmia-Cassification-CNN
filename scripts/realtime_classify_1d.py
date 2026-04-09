import numpy as np
import torch
import wfdb
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from biosppy.signals import ecg
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.models1d import EcgResNet34  # Make sure path is correct

# --- Bandpass Filter ---
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

# --- Rhythm Logic ---
def rhythm_logic(rr_intervals, beat_labels, fs):
    hr = 60 / rr_intervals
    avg_hr = np.mean(hr)
    rr_std = np.std(rr_intervals)

    total = len(beat_labels)
    v_count = sum(1 for b in beat_labels if b == 'V')
    v_ratio = v_count / total if total > 0 else 0

    # Count longest streak of V-beats
    consec_v, max_consec_v = 0, 0
    for b in beat_labels:
        if b == 'V':
            consec_v += 1
            max_consec_v = max(max_consec_v, consec_v)
        else:
            consec_v = 0

    # --- VF Logic ---
    if rr_std > 0.3 and v_ratio > 0.6:
        return "Ventricular Fibrillation"

    # --- VT Logic ---
    if (max_consec_v >= 2 and avg_hr > 100) or (v_ratio > 0.3 and avg_hr > 110):
        return "Ventricular Tachycardia"

    # --- Normal Rhythm ---
    if v_ratio < 0.1 and rr_std < 0.15:
        return "Normal Sinus Rhythm"

    return "Unknown"


# --- Main Execution ---
def main(record_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load ECG
    rec = wfdb.rdrecord(record_path)
    signal = rec.p_signal[:, 0]
    fs = rec.fs
    filtered = bandpass_filter(signal, fs)

    # Ground-truth R-peaks from annotation
    ann = wfdb.rdann(record_path, 'atr')
    r_peaks = ann.sample
    print(f"✅ Detected {len(r_peaks)} R-peaks (from annotation)")

    rr = np.diff(r_peaks / fs)

    # Load Model
    model = EcgResNet34(num_classes=4).to(device)
    checkpoint = torch.load(
        "checkpoints/resnet34_clean_1d.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare Segments (360 samples around R-peaks)
    segments = []
    for peak in tqdm(r_peaks, disable=True):
        start = max(0, peak - 180)
        end = min(len(filtered), peak + 180)
        seg = filtered[start:end]
        if len(seg) < 360:
            seg = np.pad(seg, (0, 360 - len(seg)), mode='constant')
        segments.append(seg.copy())

    # Run Inference
    batch_tensor = torch.tensor(np.array(segments)).float().unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

    class_map = ['N', 'V', 'F', 'Q']
    beat_labels = [class_map[idx] for idx in predicted_labels]
    print("Beat Label Distribution:", dict((lbl, beat_labels.count(lbl)) for lbl in set(beat_labels)))

    rhythm = rhythm_logic(rr, beat_labels, fs)
    avg_hr = 60 / np.mean(rr)

    print(f"📊 Average HR: {avg_hr:.2f} BPM")
    print(f"🩺 Rhythm Classification: {rhythm}")

# Entry point
if __name__ == "__main__":
    main("../mit-bih-arrhythmia-database-1.0.0/100")
