import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt
from biosppy.signals import ecg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.models1d import EcgResNet34
import time
import winsound

# --- Signal Filter ---
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
    consec_v, max_consec_v = 0, 0
    for b in beat_labels:
        if b == 'V':
            consec_v += 1
            max_consec_v = max(max_consec_v, consec_v)
        else:
            consec_v = 0
    if rr_std > 0.3 and v_ratio > 0.6:
        return "Ventricular Fibrillation"
    if (max_consec_v >= 2 and avg_hr > 100) or (v_ratio > 0.3 and avg_hr > 110):
        return "Ventricular Tachycardia"
    if v_ratio < 0.1 and rr_std < 0.15:
        return "Normal Sinus Rhythm"
    return "Unknown"

# --- GUI Class ---
class ECGGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Rhythm Classifier")
        self.root.attributes('-fullscreen', True)

        # --- Patient ID and File Load ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        tk.Label(top_frame, text="Patient ID:").grid(row=0, column=0, sticky="e")
        self.patient_id_entry = tk.Entry(top_frame, width=30)
        self.patient_id_entry.grid(row=0, column=1, columnspan=2, sticky="w")

        self.load_btn = tk.Button(top_frame, text="Load ECG Record", command=self.load_file)
        self.load_btn.grid(row=0, column=3, padx=10)

        self.monitor_btn = tk.Button(top_frame, text="Start Monitoring", command=self.start_monitoring)
        self.monitor_btn.grid(row=0, column=4, padx=10)

        # --- Rhythm Banner, HR, Plot ---
        self.rhythm_label = tk.Label(root, text="Rhythm: --", bg="lightgray", font=("Arial", 18, "bold"))
        self.rhythm_label.pack(fill=tk.X, pady=5)

        self.hr_label = tk.Label(root, text="Heart Rate: -- BPM", font=("Arial", 14))
        self.hr_label.pack()

        self.fig, self.ax = plt.subplots(figsize=(12, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # --- Log and Timer ---
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(bottom_frame, text="Event Log:").pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(bottom_frame, width=100, height=15, state='disabled')
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # Exit button
        self.exit_btn = tk.Button(root, text="Exit Fullscreen", command=self.exit_fullscreen)
        self.exit_btn.pack(pady=5)

        self.file_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("WFDB ECG Record", "*.dat")])
        if self.file_path:
            self.log(f"Loaded file: {os.path.basename(self.file_path)}")

    def start_monitoring(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please load an ECG record first.")
            return

        rec_path = self.file_path[:-4]
        record = wfdb.rdrecord(rec_path)
        signal = bandpass_filter(record.p_signal[:, 0], record.fs)
        ann = wfdb.rdann(rec_path, 'atr')
        r_peaks = ann.sample
        fs = record.fs

        self.ax.clear()
        self.ax.plot(signal, label="ECG Signal")
        self.ax.scatter(r_peaks, signal[r_peaks], color='red', label="R-peaks")
        self.ax.set_title(f"Annotated R-peaks for {os.path.basename(rec_path)}")
        self.ax.set_xlabel("Sample index")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.canvas.draw()

        rr = np.diff(r_peaks / fs)
        avg_hr = 60 / np.mean(rr)
        self.hr_label.config(text=f"Heart Rate: {avg_hr:.2f} BPM")

        model = EcgResNet34(num_classes=4).to(self.device)
        checkpoint = torch.load("scripts/checkpoints/resnet34_clean_1d.pth", map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()

        segments = []
        for peak in r_peaks:
            start = max(0, peak - 180)
            end = min(len(signal), peak + 180)
            seg = signal[start:end]
            if len(seg) < 360:
                seg = np.pad(seg, (0, 360 - len(seg)), mode='constant')
            segments.append(seg)

        inputs = torch.tensor(segments).float().unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        class_map = ['N', 'V', 'F', 'Q']
        beat_labels = [class_map[i] for i in preds]

        rhythm = rhythm_logic(rr, beat_labels, fs)
        self.rhythm_label.config(text=f"Rhythm: {rhythm}", bg=self.get_rhythm_color(rhythm))
        self.log(f"Rhythm detected: {rhythm}")
        self.log(f"{len(r_peaks)} R-peaks, Avg HR: {avg_hr:.2f} BPM")

        if rhythm in ["Ventricular Fibrillation", "Ventricular Tachycardia"]:
            self.trigger_alert(rhythm)

    def get_rhythm_color(self, rhythm):
        return {
            "Normal Sinus Rhythm": "lightgreen",
            "Ventricular Tachycardia": "orange",
            "Ventricular Fibrillation": "red",
            "Unknown": "gray"
        }.get(rhythm, "lightgray")

    def trigger_alert(self, rhythm):
        self.log(f"⚠️ ALERT: {rhythm} detected!")
        winsound.Beep(1000, 1000)  # 1 kHz beep for 1 second

    def log(self, message):
        self.log_box.configure(state='normal')
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_box.insert(tk.END, timestamp + message + "\n")
        self.log_box.configure(state='disabled')
        self.log_box.yview(tk.END)

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)

# Run the GUI
if __name__ == '__main__':
    root = tk.Tk()
    app = ECGGui(root)
    root.mainloop()

