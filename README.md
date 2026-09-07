# CG-Grasp

Code for the paper *Learning to Cluster: Improving the Accuracy and Convergence of Grasp Detection Models for Robotic Manipulation*.

A deep-learning based robotic grasp detection framework built on PyTorch, providing training, evaluation and inference pipelines for RGB-D grasp prediction.

## Project layout

```
├── train_network.py          # Train a grasp network
├── evaluate.py               # Evaluate a trained network
├── models/                   # Training-side network architectures
├── inference/
│   ├── models/               # Network registry used at train/inference time
│   ├── grasp_generator.py    # Generate grasps from network output
│   ├── post_process.py       # Convert raw output to grasp predictions
├── utils/
│   ├── data/                 # Dataset loaders
│   ├── dataset_processing/   # Dataset processing & evaluation helpers
│   ├── visualisation/        # Visualisation helpers
│   └── get_*.sh              # Dataset download scripts
└── hardware/                 # Camera capture and calibration
```

## Installation

- Python 3.9+
- [PyTorch](https://pytorch.org/) (with CUDA recommended)
- Other requirements: `numpy`, `opencv-python`, `tensorboardX`, `torchsummary`, `scipy`

```bash
pip install torch torchvision tensorboardX torchsummary numpy opencv-python scipy
```

## Usage

### Train

```bash
python train_network.py --network grconvnet3 --dataset cornell \
    --dataset-path /path/to/Cornell --use-depth 1 --use-rgb 1
```

See `python train_network.py --help` for all options (dataset split, batch size, epochs, learning rate scheduler, etc.).

### Evaluate

```bash
python evaluate.py --network grconvnet3 --dataset cornell \
    --dataset-path /path/to/Cornell --iou-threshold 0.25
```
