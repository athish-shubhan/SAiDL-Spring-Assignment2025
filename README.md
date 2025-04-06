# SAiDL-Spring-Assignment2025
# 🚀 SAiDL-Spring-Assignment2025 🚀

## 📋 Overview
This repository contains my submission for the Society for Artificial Intelligence and Deep Learning (SAiDL) Spring 2025 Induction Assignment. I have completed the **Core ML task** (mandatory) and the **Multi-Modality domain-specific task**.

## 📁 Repository Structure

### 1. 🧠 coreml_files
This directory contains my implementation of the Core ML task focusing on robustness with noisy labels:

- **CoreML.py**: Main implementation file for normalized losses and APL framework
- **analysis.py**: Analysis scripts for evaluating model performance
- **Bonus_Task.py**: Implementation of the bonus task with asymmetric noise
- **CoreML.pdf**: Detailed report of methodology and results
- **results/**: Directory containing result images from main experiments
- **results_bonus/**: Directory containing result images from bonus experiments
- **plots/**: Directory containing performance comparison plots
- **tables/**: Directory containing CSV files with experimental results

### 2. 📊 Task 1 - Multi-Modality
This directory contains the EDA implementation for the Multi-Modality task:

- **entities&triples.py**: Script for analyzing entities and triple relationships
- **images.py**: Script for processing and analyzing image data
- **eda_final.py**: Comprehensive EDA implementation
- **dmg777k_dataset/**: Dataset directory containing:
  - **images/**: Folder with image files
  - **triples.txt**: Triple relationships data
  - **entities.json**: Entities information
- **eda_results/**: Directory containing EDA visualizations and findings

### 3. 🔗 Task 2&3 - Multi-Modality
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
- **train_with_optuna.py**: Hyperparameter optimization with Optuna
- **finetune.py**-Bonus Task Code(Not Implemented - Only code)
- **results/**: Directory containing training results and model performance

### 4. 📝 LaTeX Report
A comprehensive LaTeX report documenting my approach, methodology, results, and analyses for all implemented tasks.

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

## 🚀 Running the Code

### Core ML
```bash
cd coreml_files
python CoreML.py
python analysis.py
python Bonus_Task.py
```

### Multi-Modality EDA
```bash
cd "Task 1"
python eda_final.py
```

### Multi-Modality GNN
```bash
cd "Task 2"
python run_focused.py
```

## 📈 Results
Detailed results and analysis can be found in the LaTeX report and in the respective results directories. The implementation demonstrates the effectiveness of normalized losses and the APL framework in handling noisy labels, as well as the capabilities of multi-modal GNNs in processing heterogeneous graph data with visual and textual information.

## 📝 LaTeX Report Structure
The LaTeX report follows this structure:
- **Introduction**: Overview of the tasks and objectives
- **Core ML Task**: Methodology, implementation details, experiments, and results
- **Multi-Modality Task**: 
  - EDA analysis and findings
  - GNN architecture and implementation
  - Experimental setup and results
- **Conclusion**: Summary of findings and potential future improvements
- **References**: Citations for all resources used
