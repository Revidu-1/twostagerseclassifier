# AST-Based Two-Stage Respiratory Sound Classifier

A deep learning system for classifying respiratory sounds using Audio Spectrogram Transformer (AST) with a two-stage hierarchical approach, as described in the report.

## Overview

This system implements the approach described in the report for respiratory sound classification using the ICBHI 2017 Challenge dataset. It employs a two-stage classification:

1. **Stage 1**: Binary classification (Normal vs Abnormal sounds)
2. **Stage 2**: Multi-class classification of abnormal sounds (Crackle, Wheeze, Both)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the Dataset

```bash
python icbhi_preprocessing.py --data_dir ./ICBHI_final_database --split_file split.txt --output_dir ./icbhi_data
```

### 2. Train and evaluate the Model

```bash
python main.py --data_dir ./icbhi_data --output_dir ./ast_two_stage_icbhi_model
```

Results will be saved in `./ast_two_stage_icbhi_model/` including:
- Trained models for both stages
- Performance metrics and confusion matrices
- Training history and visualizations

This implementation reproduces the approach detailed in the accompanying report.