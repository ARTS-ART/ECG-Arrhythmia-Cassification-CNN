import numpy as np
import wfdb
import os

def prepare_single_record_npz(record_name, record_dir, out_path='scripts/clean_single_record.npz'):
    # Initialize arrays
    X_data, y_data = [], []

    # Full paths
    record_path = os.path.join(record_dir, record_name)
    dat_path = record_path + '.dat'
    hea_path = record_path + '.hea'
    atr_path = record_path + '.atr'

    if not all(os.path.exists(p) for p in [dat_path, hea_path, atr_path]):
        raise FileNotFoundError("Missing .dat, .hea, or .atr files")

    # Load data
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]  # use channel 0
    fs = record.fs

    print(f"📥 Loaded record {record_name} | Length: {len(signal)} | FS: {fs}")

    # Symbol → Label Mapping
    symbol_to_label = {
        'N': 0,  # Normal
        'V': 1,  # Ventricular
        'F': 2,  # Fusion
        '!': 2,  # Treat flutter as fusion
        '[': 3, ']': 3  # Unknown
    }

    # Sliding window (360 samples, 50% overlap)
    window_size = 360
    for i in range(0, len(signal) - window_size + 1, window_size // 2):
        segment = signal[i:i+window_size]
        label = 3  # Default to Unknown
        for sample, symbol in zip(ann.sample, ann.symbol):
            if i <= sample < i + window_size:
                label = symbol_to_label.get(symbol, 3)
                break
        X_data.append(segment)
        y_data.append(label)

    # Save to .npz
    X_np = np.array(X_data)
    y_np = np.array(y_data)
    np.savez(out_path, X=X_np, y=y_np)
    print(f"✅ Saved {len(X_np)} segments → {out_path}")
    print(f"    ➤ X shape: {X_np.shape}, y shape: {y_np.shape}")

# ==== Run it ====
if __name__ == '__main__':
    record = '215'  # Change to any MIT-BIH record (e.g., '101', '118')
    directory = r'C:\Users\eshan\Downloads\ecg-classification-master\ecg-classification-master\mit-bih-arrhythmia-database-1.0.0'
    prepare_single_record_npz(record, directory)
