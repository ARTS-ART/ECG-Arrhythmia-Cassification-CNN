# ECG Arrhythmia Classification CNN

End-to-end deep learning system for classifying ECG signals into arrhythmia categories using a CNN-based architecture.

An ECG diagnosis project built in Python and PyTorch, focused on a trained 1D CNN pipeline for automatic arrhythmia classification. The repository also includes tools for inference, evaluation, dataset preparation, and a desktop monitoring GUI.

## Problem

Manual ECG interpretation can be slow, repetitive, and error-prone when large numbers of recordings need to be reviewed.

## Solution

This project uses a 1D convolutional neural network to classify ECG signals automatically and support faster rhythm analysis.

## Dataset

- MIT-BIH Arrhythmia Dataset: https://physionet.org/content/mitdb/1.0.0/

## What This Repo Includes

- `train.py` for model training from config files
- `inference.py` for running trained-model predictions
- `pipeline.py` for end-to-end ECG processing
- `evaluate_model.py` for confusion matrix and ROC analysis
- `gui_pyqt5.py` for a desktop ECG monitoring interface
- `configs/` for training, inference, and pipeline configuration
- `models/` for model definitions
- `datasets/`, `scripts/`, `trainers/`, `runners/`, and `utils/` for supporting code

## How to Run

```bash
python train.py --config configs/training/EcgResNet34.json
python inference.py --config configs/inference/config.json
python pipeline.py --config configs/pipelines/config.json
```

## Quick Start

```bash
pip install -r requirements.txt
```

## Example Workflow

```text
1. Prepare ECG data
2. Train the 1D CNN model
3. Run inference on new ECG signals
4. Evaluate predictions with ROC and confusion matrix plots
5. Use the GUI for rhythm monitoring and event review
```

## Results

Evaluation run used:
- `scripts/clean_single_record.npz`
- `scripts/checkpoints/resnet34_clean_1d.pth`
- CPU execution
- 3,610 test samples

Actual metrics from the bundled evaluation script:
- Accuracy: `85.68%`
- Weighted F1 score: `88.21%`

Class-wise results:
- Normal: precision `0.95`, recall `0.90`, F1 `0.92`
- Ventricular: precision `0.01`, recall `0.04`, F1 `0.02`
- Fusion: precision `0.00`, recall `0.00`, F1 `0.00`
- Unknown: precision `0.00`, recall `0.00`, F1 `0.00`

Evaluation outputs include confusion matrix and ROC curve visualizations.

## Visuals

Add these images to `assets/` for a stronger GitHub presentation:

- ECG waveform screenshot
- Prediction output screenshot
- Confusion matrix plot

## How It Works

- ECG signals are prepared and converted into model-ready samples
- The 1D CNN learns rhythm patterns from the training data
- The trained model predicts ECG classes on new inputs
- Evaluation scripts generate plots and summary metrics
- The GUI provides a simple way to inspect ECG rhythm behavior

## Project Structure

```text
ecg-classification-master/
├── configs/
├── datasets/
├── models/
├── scripts/
├── trainers/
├── runners/
├── utils/
├── train.py
├── inference.py
├── pipeline.py
├── evaluate_model.py
├── gui_pyqt5.py
├── requirements.txt
└── README.md
```


## License

This project is distributed under the MIT-style license in `LICENCE`.
