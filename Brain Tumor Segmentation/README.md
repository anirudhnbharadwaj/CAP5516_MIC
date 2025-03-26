# CAP 5516 Assignment #2: 3D U-Net for Brain Tumor Segmentation

## Overview
This repository contains the implementation of a 3D U-Net model for brain tumor segmentation on the BraTS dataset, as part of CAP 5516: Medical Image Computing. The model is trained with 5-fold cross-validation, achieving an average WT Dice score of 0.6181. The submission includes the code, report, and instructions to reproduce the results.

## Repository Structure
- `src/`: Source code.
  - `models.py`: 3D U-Net implementation.
  - `main.py`: Main script for training and evaluation.
  - `utils.py`: Utilities for metrics and visualization.
  - `data.py`: Data loading and preprocessing.
  - `train_and_evaluate.py`: Training and evaluation logic.
- `config/hyperparams.json`: Configuration file.
- `run_braTS.slurm`: SLURM script for cluster execution.
- `output/`: Output directory (logs, visualizations, plots, results table) - Just name it wahtever in the hyperparams.json and it will be created automatically.
- `report.pdf`: Detailed report with methodology, results, and discussion.
- `README.md`: This file.
- `requirements.txt`: Dependency Requirements.
- `Task01_BrainTumour.zip`: BraTS Dataset.
- `Task01_BrainTumour/`: Unzipped BraTS Dataset. [Make sure to change the data_path in hyperparams.json to match the dataset directory name]
  - `imagesTr`: Image data for training.
  - `labelsTr`: Ground truth labels for training.
  - `dataset.jsom`: Metadata file.

`Note`: Dataset has to be downloaded from https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 

## Requirements
- Python 3.8+
- PyTorch (with CUDA support)
- MONAI (`pip install monai`)
- NumPy, scikit-learn, matplotlib, seaborn, pandas, nibabel, tqdm

Install dependencies:
```bash
pip install torch torchvision monai numpy scikit-learn matplotlib seaborn pandas nibabel tqdm tensorboard
```
## Directions
Root directory contains the code and dataset. 
Update the `run_braTS.slurm` script before submitting the job. 
`main.py` is used to train and evaluate the model.
Make sure to change the `data_path` and `output_path` in the `config/hyperparams.json` file to match your local directory structure.
