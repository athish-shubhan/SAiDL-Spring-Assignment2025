import torch
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration for the multimodal tasks"""
    # Dataset
    root_dir: str = "dmg777k_dataset/"
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    
    # Model parameters
    hidden_dim: int = 256
    dropout_rate: float = 0.2
    gnn_type: str = "gat"  # Options: "gat", "gcn", "sage", "transformer"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else \
              "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Results storage
    results_dir: str = "training_results"
    save_metrics: bool = True
    save_model_checkpoints: bool = True
    
    def __post_init__(self):
        # Create results directory if needed
        os.makedirs(self.results_dir, exist_ok=True)

