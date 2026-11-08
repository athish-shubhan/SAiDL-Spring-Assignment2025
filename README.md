
# SAiDL Spring 2025 Induction Assignment

## 📋 Overview
This repository contains my submission for the Society for Artificial Intelligence and Deep Learning (SAiDL) Spring 2025 Induction Assignment. I have completed the **Core ML task** (mandatory) and the **Multi-Modality domain-specific task**.

## 📁 Repository Structure

### 1. 🧠 coreml_files
This directory contains my implementation of the Core ML task focusing on robustness with noisy labels:

- **CoreML.py**: Main implementation file for normalized losses and APL framework
- **analysis.py**: Analysis scripts for evaluating model performance
- **Bonus_Task.py**: Implementation of the bonus task with asymmetric noise
- **CoreML Task.pdf**, **Bonus_Task.pdf**: Reports on the methodology and results
- **results/**: Directory containing result images from main experiments
- **results_bonus/**: Directory containing result images from bonus experiments
- **plots/**: Directory containing performance comparison plots
- **tables/**: Directory containing CSV files with experimental results

### 2. 📊 Task-1 (Multi-Modality)
This directory contains the EDA implementation for the Multi-Modality task:

- **entities&triples.py**: Script for analyzing entities and triple relationships
- **images.py**: Script for processing and analyzing image data
- **eda_final.py**: Comprehensive EDA implementation
- **dmg777k_dataset/**: not included in the repository. It is produced locally from the `kgbench` package by `entities&triples.py` and `images.py` (triples, entities and images).
- **eda_results/**: Directory containing EDA visualizations and findings

### 3. 🔗 Task-2&3 (Multi-Modality)
This directory contains the GNN implementation for node classification:

- **clip_gnn.py**: Integration of CLIP features with GNN
- **config.py**: Configuration settings for the models
- **customencoder.py**: Custom encoding implementations
- **gnn.py**: Graph Neural Network implementation
- **huggingface.py**: Integration with HuggingFace models
- **run_focused.py**: For running task 2 in sequence
- **sfeatures.py**: Simple Feature Extraction
- **subset_strategy.py**: Implementation of dataset subset creation
- **train_utils.py**: Utilities for training
- **train.py**: Main training script
- **ablations.py**: Ablation experiments
- **finetune.py**: Bonus task code

### 4. 📝 Report
`Athish_SaiDL_Assignment (1).pdf` documents the approach, methodology, results and analyses for all tasks.

## 🔍 Task Overview

### Core ML Task ✅
Implemented robustness techniques for handling noisy labels in machine learning:
- 🔄 Normalized loss functions (NCE, NFL)
- 🛡️ Active-Passive Loss (APL) framework
- 📉 Performance comparison under different noise rates
- 🎯 Bonus task with asymmetric noise

### Multi-Modality Task ✅
Developed a multi-modal Graph Neural Network that:
- 📊 Performs Exploratory Data Analysis on the dmg777k dataset
- 🔄 Combines vision and language modalities for node classification
- 🤖 Uses pre-trained models from HuggingFace to embed nodal information
- 🧪 Implements and evaluates various feature extraction techniques

## 🛠️ Dependencies

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- torch_geometric
- transformers
- pillow
- scikit-learn
- optuna (for hyperparameter optimization)
- kgbench (dmg777k dataset loader), tqdm

## 🚀 Running the Code

### Core ML
```bash
cd coreml_files
python CoreML.py
python analysis.py
python Bonus_Task.py
```

### Multi-Modality dataset and EDA
The scripts expect `dmg777k_dataset/` (entities, triples, images) in the directory they are run from. Create it once in each of `Task-1/` and `Task-2&3/` (or copy it), then run the EDA:
```bash
cd Task-1
python "entities&triples.py"   # writes dmg777k_dataset/entities.json and triples.txt
python images.py               # extracts images into dmg777k_dataset/images
python eda_final.py
```

### Multi-Modality GNN
```bash
cd "Task-2&3"
python run_focused.py
python finetune.py
```

## 📈 Results
Results and analysis are in the PDF report and in the results directories (`coreml_files/results*`, `coreml_files/tables`, `Task-1/eda_results`).


