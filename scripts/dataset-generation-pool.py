

from glob import glob

from dataset_generation import process_record

input_dir = "../mit-bih-arrhythmia-database-1.0.0/*.hea"
ecg_data = [f[:-4] for f in glob(input_dir)]

for record in ecg_data:
    process_record(record)

