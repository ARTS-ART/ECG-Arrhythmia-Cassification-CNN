import os
import sys
import csv
import time
import numpy as np
import torch
import wfdb
import traceback
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
from biosppy.signals import ecg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QMessageBox, QFrame, QToolBar
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTime, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import winsound
from models.models1d import EcgResNet34

# Signal Processing
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


# VF Detection
def detect_vf(signal, fs):
    segment = signal - np.mean(signal)
    freqs = np.fft.fftfreq(len(segment), d=1/fs)
    fft_vals = np.abs(fft(segment)) ** 2
    vf_band = (freqs >= 4) & (freqs <= 10)
    vf_power = np.sum(fft_vals[vf_band])
    total_power = np.sum(fft_vals)
    vf_ratio = vf_power / total_power if total_power > 0 else 0
    zero_crossings = ((segment[:-1] * segment[1:]) < 0).sum()
    return vf_ratio > 0.25 and zero_crossings > 100

# VT Detection
def detect_vt(signal, fs):
    try:
        out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        r_peaks = out['rpeaks']
        if len(r_peaks) < 4:
            return False
        rr = np.diff(r_peaks / fs)
        hr = 60 / rr
        avg_hr = np.mean(hr)
        vt_candidates = sum(hr > 100)
        return vt_candidates >= 3
    except Exception:
        return False

# HR Calculation for CUDB
def calculate_hr(signal, fs, rhythm):
    if rhythm == "Ventricular Fibrillation":
        return None  # HR unreliable for VF
    try:
        out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        r_peaks = out['rpeaks']
        if len(r_peaks) < 2:
            return None
        rr = np.diff(r_peaks / fs)
        avg_hr = 60 / np.mean(rr)
        return avg_hr if 30 <= avg_hr <= 300 else None  # Valid HR range
    except Exception:
        return None

# Original Rhythm Logic (for MIT-BIH)
def rhythm_logic(rr_intervals, beat_labels, fs):
    hr = 60 / rr_intervals
    avg_hr = np.mean(hr)
    rr_std = np.std(rr_intervals)
    v_indices = [i for i, b in enumerate(beat_labels) if b == 'V']
    v_count = len(v_indices)
    v_ratio = v_count / len(beat_labels)

    consec_v, max_consec_v, vt_start_index = 0, 0, -1
    for i, b in enumerate(beat_labels):
        if b == 'V':
            consec_v += 1
            if consec_v == 3:
                vt_start_index = i - 2
            max_consec_v = max(max_consec_v, consec_v)
        else:
            consec_v = 0

    if rr_std > 0.3 and v_ratio > 0.6:
        return "Ventricular Fibrillation", vt_start_index
    if (max_consec_v >= 3 and avg_hr > 100) or (v_ratio > 0.025 and avg_hr > 110):
        return "Ventricular Tachycardia", vt_start_index
    if v_ratio < 0.1 and rr_std < 0.15:
        return "Normal Sinus Rhythm", -1
    return "Unknown", -1

class ECGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI ECG Rhythm Classifier Dashboard")
        self.central = QWidget()
        self.setCentralWidget(self.central)

        # Initialize state variables
        self.file_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self.scroll_plot)
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_flash)
        self.sms_timer = QTimer()
        self.sms_timer.timeout.connect(self.clear_sms_message)
        self.time_left = 3600
        self.flash_on = False
        self.flash_duration = 0
        self.golden_timer_started = False
        self.plot_index = 0
        self.signal_data = None
        self.fs = None
        self.r_peaks = None
        self.rhythm_annotations = None
        self.cudb_path = "./CUBD"
        self.current_rhythm = None
        self.current_hr = None

        self.build_ui()
        self.log("GUI initialized successfully")

    def build_ui(self):
        layout = QVBoxLayout()

        # Toolbar with Exit button
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.exit_btn = QPushButton("✖ Exit")
        self.exit_btn.clicked.connect(self.exit_application)
        toolbar.addWidget(self.exit_btn)
        toolbar.setStyleSheet("QToolBar { border: 0px; }")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # SMS Message Label
        self.sms_label = QLabel("")
        self.sms_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.sms_label.setAlignment(Qt.AlignCenter)
        self.sms_label.setStyleSheet("color: red; padding: 5px;")
        layout.addWidget(self.sms_label)

        top = QHBoxLayout()
        self.patient_label = QLabel("Patient ID:")
        self.patient_label.setFont(QFont("Segoe UI", 12))
        self.patient_id = QLabel("--")
        self.patient_id.setFont(QFont("Segoe UI", 12, QFont.Bold))
        top.addWidget(self.patient_label)
        top.addWidget(self.patient_id)
        top.addStretch()
        self.timer_display = QLabel("Golden Hour: 60:00")
        self.timer_display.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.timer_display.setVisible(False)
        top.addWidget(self.timer_display)
        layout.addLayout(top)

        self.rhythm_label = QLabel("Rhythm: --")
        self.rhythm_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.rhythm_label.setAlignment(Qt.AlignCenter)
        self.rhythm_label.setStyleSheet("background-color: lightgray; padding: 10px; border-radius: 10px;")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 4)
        self.rhythm_label.setGraphicsEffect(shadow)
        layout.addWidget(self.rhythm_label)
        
        # --- NEW: Confidence Label ---
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Segoe UI", 14))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        # --- END NEW ---

        self.hr_label = QLabel("Heart Rate: -- BPM")
        self.hr_label.setFont(QFont("Segoe UI", 16))
        self.hr_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hr_label)

        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Load ECG File")
        self.load_btn.clicked.connect(self.load_file)
        self.load_cudb_btn = QPushButton("📂 Load from CUBD Folder")
        self.load_cudb_btn.clicked.connect(self.load_cudb_file)
        self.monitor_btn = QPushButton("▶ Start Monitoring")
        self.monitor_btn.clicked.connect(self.start_monitoring)
        self.stop_btn = QPushButton("⏹ Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.export_btn = QPushButton("📤 Export Events to CSV")
        self.export_btn.clicked.connect(self.export_events)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.load_cudb_btn)
        btn_layout.addWidget(self.monitor_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)

        self.log_box = QTextEdit()
        self.log_box.setFont(QFont("Consolas", 12))
        self.log_box.setReadOnly(True)
        layout.addWidget(QLabel("Event Log:", font=QFont("Segoe UI", 12)))
        layout.addWidget(self.log_box)

        self.central.setLayout(layout)
        self.log("Application started")

    def update_timer(self):
        self.time_left -= 1
        mins, secs = divmod(self.time_left, 60)
        self.timer_display.setText(f"Golden Hour: {mins:02}:{secs:02}")
        if self.time_left <= 0:
            self.timer.stop()

    def toggle_flash(self):
        if self.flash_duration > 5:
            self.flash_timer.stop()
            self.central.setStyleSheet("")
            return
        self.flash_on = not self.flash_on
        color = "background-color: rgba(255,0,0,80);" if self.flash_on else ""
        self.central.setStyleSheet(color)
        self.flash_duration += 0.5

    def clear_sms_message(self):
        self.sms_label.setText("")
        self.sms_timer.stop()

    def scroll_plot(self):
        if self.signal_data is None:
            self.log("No signal data for plotting")
            return
        window_size = self.fs * 5
        if self.plot_index + window_size >= len(self.signal_data):
            self.scroll_timer.stop()
            self.log("Reached end of signal data")
            return
        self.ax.clear()
        self.ax.plot(self.signal_data[self.plot_index:self.plot_index + window_size])
        if self.rhythm_annotations:
            for sample, symbol in self.rhythm_annotations:
                if self.plot_index <= sample < self.plot_index + window_size and symbol in ['VF', 'VFL', 'VT']:
                    self.ax.axvline(sample - self.plot_index, color='red', linestyle='--', linewidth=1.5)
        self.ax.set_title("Live Scrolling ECG")
        self.canvas.draw()
        self.plot_index += self.fs // 2

    def log(self, msg):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_box.append(f"{timestamp} {msg}")

    def export_events(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", "events.csv", "CSV files (*.csv)")
        if path:
            try:
                # Save CSV
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time", "Event"])
                    for line in self.log_box.toPlainText().splitlines():
                        writer.writerow(line.split(' ', 1))
                self.log(f"Events exported to CSV: {path}")
                
                # Save plot image
                img_path = None
                if self.signal_data is not None and self.current_rhythm is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    record_name = os.path.basename(self.file_path) if self.file_path else "unknown"
                    img_path = f"{path[:-4]}{self.current_rhythm.replace(' ', '')}_{timestamp}.png"
                    self.fig.savefig(img_path, bbox_inches='tight')
                    self.log(f"Plot image saved as: {img_path}")
                QMessageBox.information(self, "Exported", f"Event log saved as {path}\nPlot image saved as {img_path if img_path else 'N/A'}")
            except Exception as e:
                self.log(f"Error exporting events or image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
        else:
            self.log("Export canceled")

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open ECG File", "", "WFDB Record (*.dat)")
        if path:
            self.file_path = path[:-4]
            self.patient_id.setText(os.path.basename(self.file_path))
            self.log(f"Loaded file: {self.file_path}")
        else:
            self.log("No file selected")

    def load_cudb_file(self):
        if not os.path.exists(self.cudb_path):
            QMessageBox.critical(self, "Error", f"CUBD folder not found at {self.cudb_path}")
            self.log(f"CUBD folder {self.cudb_path} not found")
            return
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open CUBD ECG File", self.cudb_path, "WFDB Record (*.dat)")
            if path:
                if not os.path.isfile(path) or not os.path.isfile(path[:-4] + '.atr'):
                    QMessageBox.critical(self, "Error", "Invalid or missing .dat/.atr files")
                    self.log("Invalid or missing .dat/.atr files")
                    return
                self.file_path = path[:-4]
                self.patient_id.setText(os.path.basename(self.file_path))
                self.log(f"Loaded CUBD file: {self.file_path}")
            else:
                self.log("No CUBD file selected")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CUBD file: {e}")
            self.log(f"Error loading CUDB file: {e}")

    def stop_monitoring(self):
        self.log("Stop monitoring clicked")
        self.scroll_timer.stop()
        self.flash_timer.stop()
        self.timer.stop()
        self.sms_timer.stop()
        self.signal_data = None
        self.fs = None
        self.r_peaks = None
        self.rhythm_annotations = None
        self.plot_index = 0
        self.current_rhythm = None
        self.current_hr = None
        self.golden_timer_started = False
        self.timer_display.setVisible(False)
        self.rhythm_label.setText("Rhythm: --")
        self.rhythm_label.setStyleSheet("background-color: lightgray; padding: 10px; border-radius: 10px;")
        self.hr_label.setText("Heart Rate: -- BPM")
        self.confidence_label.setText("Confidence: --") # Reset confidence
        self.patient_id.setText("--")
        self.sms_label.setText("")
        self.log_box.clear()  # Clear event log
        self.ax.clear()
        self.ax.set_title("Live Scrolling ECG")
        self.canvas.draw()
        self.log("Monitoring stopped, event log cleared, ready to load new record")

    def exit_application(self):
        self.log("Exit button clicked")
        try:
            self.scroll_timer.stop()
            self.flash_timer.stop()
            self.timer.stop()
            self.sms_timer.stop()
            QApplication.quit()
            self.log("Application quit successfully")
        except Exception as e:
            self.log(f"Error during exit: {e}")

    def start_monitoring(self):
        try:
            self.log("Starting monitoring")
            if not self.file_path:
                QMessageBox.critical(self, "Error", "Load ECG file first.")
                self.log("No file path set")
                return

            record = wfdb.rdrecord(self.file_path)
            self.fs = record.fs
            self.signal_data = bandpass_filter(record.p_signal[:, 0], self.fs)
            self.log(f"Loaded record, fs={self.fs}, signal length={len(self.signal_data)} samples")

            # Check if file is CUDB
            record_name = os.path.basename(self.file_path)
            is_cudb = record_name.startswith('cu') and record_name[2:].isdigit() and 1 <= int(record_name[2:]) <= 35
            self.log(f"Processing {'CUBD' if is_cudb else 'MIT-BIH'} record: {record_name}")

            # Load rhythm annotations
            try:
                ann = wfdb.rdann(self.file_path, 'atr')
                self.rhythm_annotations = [(sample, aux.strip()) for sample, aux in zip(ann.sample, ann.aux_note) if aux.strip()]
                self.log(f"Loaded {len(self.rhythm_annotations)} rhythm annotations")
            except:
                self.rhythm_annotations = []
                self.log("No rhythm annotations found")

            # CUDB processing
            if is_cudb:
                vt_detected, vf_detected = False, False
                window_size = self.fs * 10  # 10-second window
                step = self.fs * 5  # 5-second step
                rhythm = "Unknown"
                rhythm_samples = []

                # Check rhythm annotations first
                rhythm_counts = {}
                for sample, symbol in self.rhythm_annotations:
                    rhythm_counts[symbol] = rhythm_counts.get(symbol, 0) + 1
                total_count = sum(rhythm_counts.values())
                vf_vt_ratio = sum(rhythm_counts.get(s, 0) for s in ['VF', 'VFL', 'VT']) / total_count if total_count > 0 else 0

                if vf_vt_ratio > 0.3:
                    rhythm = "Ventricular Fibrillation"
                    rhythm_samples = [sample for sample, symbol in self.rhythm_annotations if symbol in ['VF', 'VFL']]
                elif vf_vt_ratio > 0.1:
                    rhythm = "Ventricular Tachycardia"
                    rhythm_samples = [sample for sample, symbol in self.rhythm_annotations if symbol in ['VT']]

                # Additional VF/VT detection
                for start in range(0, len(self.signal_data) - window_size, step):
                    segment = self.signal_data[start:start + window_size]
                    if not vf_detected and detect_vf(segment, self.fs):
                        minsec = time.strftime('%M:%S', time.gmtime(start / self.fs))
                        self.log(f"Ventricular Fibrillation detected at {minsec}, sample {start}")
                        vf_detected = True
                        rhythm = "Ventricular Fibrillation"
                        rhythm_samples.append(start)
                    if not vt_detected and detect_vt(segment, self.fs):
                        minsec = time.strftime('%M:%S', time.gmtime(start / self.fs))
                        self.log(f"Ventricular Tachycardia detected at {minsec}, sample {start}")
                        vt_detected = True
                        if rhythm != "Ventricular Fibrillation":
                            rhythm = "Ventricular Tachycardia"
                            rhythm_samples.append(start)
                    if vf_detected and vt_detected:
                        break

                if not vf_detected and not vt_detected and not rhythm_samples:
                    rhythm = "Unknown"
                    self.log(f"Unknown rhythm detected, samples {len(self.signal_data)}")

                # Calculate HR for CUDB
                self.current_rhythm = rhythm
                self.current_hr = calculate_hr(self.signal_data[:self.fs * 60], self.fs, rhythm)
                if self.current_hr is None:
                    self.hr_label.setText("Heart Rate: -- BPM")
                    self.log("Heart Rate: Not calculated (unreliable or VF)")
                else:
                    self.hr_label.setText(f"Heart Rate: {self.current_hr:.2f} BPM")
                    self.log(f"Heart Rate: {self.current_hr:.2f} BPM")

                # Update GUI
                self.rhythm_label.setText(f"Rhythm: {rhythm}")
                self.confidence_label.setText("Confidence: N/A (Classical Method)") # No confidence for classical
                color = {
                    "Normal Sinus Rhythm": "#DFFFD6",
                    "Ventricular Tachycardia": "#FFD700",
                    "Ventricular Fibrillation": "#FF4C4C",
                    "Unknown": "#C0C0C0"
                }.get(rhythm, "#DDDDDD")
                self.rhythm_label.setStyleSheet(f"background-color: {color}; padding: 10px; border-radius: 10px;")

                if rhythm in ["Ventricular Tachycardia", "Ventricular Fibrillation"]:
                    if not self.golden_timer_started:
                        self.timer_display.setVisible(True)
                        self.timer.start(1000)
                        self.golden_timer_started = True
                        self.log("Started golden hour timer")
                    for sample in rhythm_samples:
                        minsec = time.strftime('%M:%S', time.gmtime(sample / self.fs))
                        self.log(f"{rhythm} detected at {minsec}, sample {sample}")
                        self.sms_label.setText("SMS Sent to Doctor")
                        self.sms_timer.start(5000)
                    winsound.Beep(1000, 500)
                    self.flash_duration = 0
                    self.flash_timer.start(500)
                else:
                    self.log(f"{rhythm} detected, samples {len(self.signal_data)}")

                self.plot_index = 0
                self.scroll_timer.start(500)
                return

            # MIT-BIH processing
            ann = wfdb.rdann(self.file_path, 'atr')
            self.r_peaks = ann.sample
            self.log(f"Loaded {len(self.r_peaks)} R-peaks")

            rr = np.diff(self.r_peaks / self.fs)
            avg_hr = 60 / np.mean(rr) if len(rr) > 0 else 0
            self.current_rhythm = "Unknown"
            self.current_hr = avg_hr
            self.hr_label.setText(f"Heart Rate: {avg_hr:.2f} BPM")
            self.log(f"Heart Rate: {avg_hr:.2f} BPM")

            model = EcgResNet34(num_classes=4).to(self.device)
            checkpoint = torch.load("scripts/checkpoints/resnet34_clean_1d.pth", map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            self.log("Loaded model")

            segments = []
            for peak in self.r_peaks:
                start = max(0, peak - 180)
                end = min(len(self.signal_data), peak + 180)
                seg = self.signal_data[start:end]
                if len(seg) < 360:
                    seg = np.pad(seg, (0, 360 - len(seg)), mode='constant')
                segments.append(seg)
            self.log(f"Prepared {len(segments)} segments")

            inputs = torch.tensor(segments).float().unsqueeze(1).to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
                
                # --- NEW: Calculate Confidence ---
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probabilities, dim=1)
                avg_confidence = max_probs.mean().item()
                self.confidence_label.setText(f"Confidence: {avg_confidence:.2%}")
                self.log(f"Model average confidence: {avg_confidence:.2%}")
                # --- END NEW ---
                
                preds = preds.cpu().numpy()
            self.log("Model inference completed")

            class_map = ['N', 'V', 'F', 'Q']
            beat_labels = [class_map[i] for i in preds]
            rhythm, vt_index = rhythm_logic(rr, beat_labels, self.fs)
            self.current_rhythm = rhythm
            self.rhythm_label.setText(f"Rhythm: {rhythm}")
            self.log(f"{rhythm} detected, R-peaks {len(self.r_peaks)}, HR {avg_hr:.2f} BPM")

            color = {
                "Normal Sinus Rhythm": "#DFFFD6",
                "Ventricular Tachycardia": "#FFD700",
                "Ventricular Fibrillation": "#FF4C4C",
                "Unknown": "#C0C0C0"
            }.get(rhythm, "#DDDDDD")
            self.rhythm_label.setStyleSheet(f"background-color: {color}; padding: 10px; border-radius: 10px;")

            if rhythm in ["Ventricular Tachycardia", "Ventricular Fibrillation"]:
                if not self.golden_timer_started:
                    self.timer_display.setVisible(True)
                    self.timer.start(1000)
                    self.golden_timer_started = True
                    self.log("Started golden hour timer")
                if vt_index != -1:
                    vt_sample = self.r_peaks[vt_index]
                    minsec = time.strftime('%M:%S', time.gmtime(vt_sample / self.fs))
                    self.log(f"{rhythm} detected at {minsec}, sample {vt_sample}")
                    self.sms_label.setText("SMS Sent to Doctor")
                    self.sms_timer.start(5000)
                    self.ax.axvline(vt_sample, color='red', linestyle='--', linewidth=1.5)
                winsound.Beep(1000, 500)
                self.flash_duration = 0
                self.flash_timer.start(500)
            else:
                self.log(f"{rhythm} detected, R-peaks {len(self.r_peaks)}, HR {avg_hr:.2f} BPM")

            self.plot_index = 0
            self.scroll_timer.start(500)

        except Exception as e:
            traceback.print_exc(file=open("error_log.txt", "a"))
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")
            self.log(f"Error in start_monitoring: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ECGApp()
    window.showMaximized()
    sys.exit(app.exec_())
