import os
import os.path as osp
import wfdb
import numpy as np
from sklearn.preprocessing import scale

valid_labels = ["N", "V", "F", "Q"]
segment_length = 128
output_dir = "../data"

def process_record(ecg_file):
    name = osp.basename(ecg_file)
    record = wfdb.rdrecord(ecg_file)
    ann = wfdb.rdann(ecg_file, extension="atr")

    for sig_name, signal in zip(record.sig_name, record.p_signal.T):
        signal = scale(signal)
        for label, peak in zip(ann.symbol, ann.sample):
            if label not in valid_labels:
                continue

            left, right = peak - segment_length // 2, peak + segment_length // 2
            if left < 0 or right >= len(signal):
                continue

            segment = signal[left:right]
            out_path = osp.join(output_dir, "1D", name, sig_name, label)
            os.makedirs(out_path, exist_ok=True)
            np.save(osp.join(out_path, f"{peak}.npy"), segment)
