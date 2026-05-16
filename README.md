# Event-Based Eye Tracking and Blink Detection

This repository combines high-precision eye tracking and blink phase classification using both Spiking Neural Networks (SNN). It leverages the temporal precision of event-based sensors for safety-critical applications like driver monitoring.

## Overview

The project integrates two core functionalities:
1.  **Blink Phase Classification**: Identifying four physiological states (Open, Closing, Closed, Opening) using RSNN (Recurrent Spiking Neural Network) and CRNN (Convolutional Recurrent Neural Network) architectures.
2.  **Eye Tracking**: High-precision coordinate regression using an event-driven SNN with Parametric LIF (PLIF) neurons and soft-argmax decoding.

--- 

## Project Structure

```text
├── dataset_eyeblinking.py  # Loader for blink detection datasets (.pt tensors)
├── dataset_eyetracking.py  # Loader for eye tracking datasets (events + labels)
├── model_blink.py          # SNN architecture for blink detection
├── model_blink_ann.py      # ANN (CRNN) baseline for blink detection
├── model_track.py          # SNN architecture for eye tracking
├── utils_eyetracking.py    # Utilities for tracking, heatmaps, and visualization
├── visualise.py            # Main entry point for inference, evaluation, and visualization
├── blink.pth               # Pre-trained SNN weights (Blink)
├── blink_ann.pth           # Pre-trained ANN weights (Blink)
├── track.pth               # Pre-trained SNN weights (Tracking)
├── DV008.h5                # Sample event-based data file
└── DV008.mp4               # Sample input/output video
```

---

## Requirements

-   **Python**: 3.8+
-   **Core Libraries**:
    -   PyTorch 2.0+
    -   OpenCV (`opencv-python`)
    -   NumPy
    -   h5py
    -   tqdm
    -   Scikit-learn
    -   Scipy

---

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd Eyetracking-and-Eye-blinking-detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install torch torchvision numpy opencv-python h5py tqdm scikit-learn scipy
    ```

3.  **Data Preparation**:
    Ensure you have the required `.h5` event files and pre-trained weights (`.pth`) in the root directory or specify their paths via command-line arguments.

---

## Usage

### Inference and Visualization
The `visualise.py` script runs both models simultaneously on an event stream, evaluates performance if labels are provided, and generates a visualization video.

```bash
# Basic inference with visualization
python visualise.py --events DV008.h5 --visualise --output result.mp4

# Inference with ground truth evaluation
python visualise.py --events DV008.h5 --labels labels.txt --visualise
```

**Key Arguments**:
-   `--events`: Path to the input `.h5` event file.
-   `--labels`: Path to the ground truth labels (optional).
-   `--blink_weights`: Path to blink model weights (default: `blink.pth`).
-   `--track_weights`: Path to tracking model weights (default: `track.pth`).
-   `--visualise`: Flag to enable video generation.
-   `--slow_down`: Slow motion factor for the output video (default: 3).

---
