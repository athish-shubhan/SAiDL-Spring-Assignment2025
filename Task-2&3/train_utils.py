import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from config import Config

class ResultTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.results_dir, exist_ok=True)
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(cfg.results_dir, f"exp_{self.exp_id}")
        os.makedirs(self.save_path, exist_ok=True)
        
    def save_metrics(self, metrics: dict, epoch: int):
        path = os.path.join(self.save_path, f"metrics_epoch{epoch}.json")
        with open(path, 'w') as f:
            json.dump(metrics, f)
            
    def save_model(self, model: nn.Module, epoch: int):
        path = os.path.join(self.save_path, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
