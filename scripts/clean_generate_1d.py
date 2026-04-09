import os
import os.path as osp
import wfdb
import numpy as np
from sklearn.preprocessing import scale
from glob import glob

# CONFIG
VALID_LABELS = ["N", "V", "F", "Q"]
SEGMENT_LENGTH = 128
OUTPUT_DIR = "clean_1D_dataset"  # This will be created

def process_record(ecg_file):
    name = osp.basename(ecg_file)
    try:
        record = wfdb.rdrecord(ecg_file)
        ann = wfdb.rdann(ecg_file, extension="atr")
    except Exception as e:
        print(f"❌ Error reading {name}: {e}")
        return

    for sig_name, signal in zip(record.sig_name, record.p_signal.T):
        signal = scale(signal)
        for label, peak in zip(ann.symbol, ann.sample):
            if label not in VALID_LABELS:
                continue

            left = peak - SEGMENT_LENGTH // 2
            right = peak + SEGMENT_LENGTH // 2
            if left < 0 or right > len(signal):
                continue

            segment = signal[left:right]
            out_dir = osp.join(OUTPUT_DIR, name, sig_name, label)
            os.makedirs(out_dir, exist_ok=True)
            out_file = osp.join(out_dir, f"{peak}.npy")
            np.save(out_file, segment)
        print(f"✅ {name} → {sig_name} done")

def main():
    input_dir = "../mit-bih-arrhythmia-database-1.0.0/*.hea"  # Change path if needed
    ecg_files = sorted([f[:-4] for f in glob(input_dir)])
    print(f"📦 Total records: {len(ecg_files)}")

    for i, record in enumerate(ecg_files):
        print(f"➡️  [{i+1}/{len(ecg_files)}] Processing {record}")
        process_record(record)

if __name__ == "__main__":
    main()
