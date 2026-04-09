import wfdb
import matplotlib.pyplot as plt

record_path = "../CUBD/cu01"  # or mit-bih-arrhythmia-database-1.0.0/215
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

plt.figure(figsize=(15, 4))
plt.plot(record.p_signal[:, 0], label="ECG Signal")
plt.scatter(annotation.sample, record.p_signal[annotation.sample, 0], color='red', s=10, label="R-peaks (Expert)")
plt.legend()
plt.title(f"Annotated R-peaks for {record_path}")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.show()
