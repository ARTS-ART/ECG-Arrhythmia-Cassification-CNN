# Auto Diagnosis of ECG Using Advanced AI Technology

This repository contains an ECG diagnosis workflow built with advanced AI techniques in PyTorch. The project is focused on training and using a 1D CNN model for ECG arrhythmia classification, along with supporting tools for preprocessing, inference, evaluation, visualization, and rhythm monitoring.

## Project Highlights

- Train ECG classification models from JSON configuration files
- Run inference and end-to-end ECG processing pipelines
- Focus on a trained 1D CNN model for ECG classification
- Generate training datasets and annotation files from ECG records
- Evaluate trained checkpoints with confusion matrix and ROC visualizations
- Launch a desktop GUI for ECG monitoring and rhythm event logging
- Includes local data folders and generated outputs used during experimentation

## Repository Structure

- `train.py`: starts model training from a training config
- `inference.py`: runs inference from an inference config
- `pipeline.py`: executes the configured ECG processing pipeline
- `evaluate_model.py`: evaluates a trained 1D model and saves metrics plots
- `gui_pyqt5.py`: PyQt5 dashboard for ECG rhythm monitoring
- `scripts/`: dataset generation, annotation generation, cleaning, and helper scripts
- `configs/`: training, inference, and pipeline JSON/YAML configuration files
- `models/`: 1D and 2D CNN model definitions
- `datasets/`: dataset loader implementations
- `trainers/`, `runners/`, `pipelines/`: core execution logic
- `utils/`: shared utilities
- `notebooks/`: Jupyter notebooks for experimentation
- `etc/`: project images and documentation assets
- `data/`, `CUBD/`, `mit-bih-arrhythmia-database-1.0.0/`: ECG datasets and generated samples
- `experiments/`: experiment outputs, checkpoints, and result artifacts

## Model Used

This project is centered on a 1D CNN-based ECG classification workflow. The repository includes multiple legacy configs, but the trained and used approach in this project is the 1D CNN pipeline.

## Installation

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. If you plan to use the GUI, ensure system support for `PyQt5` and audio/Windows-specific modules is available.
4. Download or place the ECG datasets in the expected project directories before training or running the pipeline.

## Usage

### Train the 1D CNN model

```bash
python train.py --config configs/training/EcgResNet34.json
```

### Run inference

```bash
python inference.py --config configs/inference/config.json
```

### Run the ECG pipeline

```bash
python pipeline.py --config configs/pipelines/config.json
```

### Generate datasets and annotations

Examples from the `scripts/` folder:

```bash
python scripts/dataset-generation-pool.py
python scripts/annotation-generation-1d.py
python scripts/annotation-generation-2d.py
```

### Evaluate a trained model

```bash
python evaluate_model.py
```

This generates outputs such as `confusion_matrix.png` and `roc_curve_enhanced.png`.

### Launch the GUI dashboard

```bash
python gui_pyqt5.py
```

## Data Notes

This repository currently includes large local datasets and generated files, including:

- MIT-BIH arrhythmia data
- CUBD rhythm data
- Generated `.npy` and `.npz` training assets
- Experiment outputs and plots

If you plan to publish this repository to GitHub, consider whether all dataset and environment folders should remain versioned. Large folders such as `data/`, `ecg_env/`, and dataset dumps may be better excluded with `.gitignore` or Git LFS depending on your publishing goal.

## Requirements Review

The current `requirements.txt` is largely based on an older PyTorch stack and may need cleanup for a fresh setup.

Notable points:

- `torch==1.1.0` and `torchvision==0.3.0` are very old
- `tensorboard==1.14` is tied to an older TensorFlow-era release cycle
- `numpy==1.21.0` may not be compatible with every pinned legacy package here
- `scikit_learn==0.22.2.post1` uses the package name `scikit_learn`, while many environments expect `scikit-learn`
- `PyQt5` is used by `gui_pyqt5.py` but is not listed in `requirements.txt`
- `biosppy` is imported by `gui_pyqt5.py` but is not listed in `requirements.txt`
- `wfdb==2.2.1` is included and is required for waveform record handling
- `Pillow<7` is pinned tightly and may conflict with newer environments

A safer next step before publishing is to test installation in a fresh environment and then update dependency pins based on the Python version you want to support.

## License

This project includes an MIT-style license in `LICENCE`.
