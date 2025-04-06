import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph
from transformers import CLIPProcessor, CLIPModel
import os
import json
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime

def get_gnn_layer(gnn_type, in_channels, out_channels, heads=4, dropout=0.2, edge_dim=None):
    """Get GNN layer based on specified type"""
    if gnn_type == 'gat':
        return GATConv(in_channels, out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
    elif gnn_type == 'gcn':
        return GCNConv(in_channels, out_channels)
    elif gnn_type == 'sage':
        return SAGEConv(in_channels, out_channels)
    elif gnn_type == 'transformer':
        return TransformerConv(in_channels, out_channels, edge_dim=edge_dim)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")

class JointFineTunedModel(nn.Module):
    def __init__(self, num_classes: int, gnn_type: str = 'gat', hidden_dim: int = 256,
                 dropout_rate: float = 0.2, edge_attr_dim: int = 0):
        """
        Joint fine-tuned model combining CLIP with GNN
        Args:
            num_classes: Number of output classes
            gnn_type: Type of GNN layer to use ('gat', 'gcn', 'sage', 'transformer')
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout rate
            edge_attr_dim: Dimension of edge attributes
        """
        super(JointFineTunedModel, self).__init__()
        
        # Load CLIP model for joint fine-tuning
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze most of CLIP, unfreeze only last transformer layers
        for name, param in self.clip.named_parameters():
            if 'visual.transformer.resblocks.11' in name or 'text_model.encoder.layers.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Projections for CLIP outputs
        self.image_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Store GNN type for reference
        self.gnn_type = gnn_type
        
        # GNN layers using the helper function
        self.gnn_layers = nn.ModuleList([
            get_gnn_layer(gnn_type, hidden_dim * 2, hidden_dim, heads=4, dropout=dropout_rate, edge_dim=edge_attr_dim),
            get_gnn_layer(gnn_type, hidden_dim * (4 if gnn_type == 'gat' else 1), hidden_dim * 2, heads=4, dropout=dropout_rate, edge_dim=edge_attr_dim),
            get_gnn_layer(gnn_type, hidden_dim * 2, hidden_dim, heads=4, dropout=dropout_rate, edge_dim=edge_attr_dim)
        ])
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bn_classifier = nn.BatchNorm1d(num_classes)

    def forward(self, images: List[torch.Tensor], texts: List[str],
                edge_indices: List[torch.Tensor], edge_attrs: List[Optional[torch.Tensor]],
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model"""
        device = next(self.parameters()).device
        
        # Process images with CLIP
        with torch.no_grad():
            image_inputs = self.clip_processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)
            image_features = self.clip.get_image_features(**image_inputs)
        
        # Process text with CLIP
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
                return_attention_mask=True
            ).to(device)
            text_features = self.clip.get_text_features(**text_inputs)
        
        # Project features
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)
        
        # Concatenate modalities
        x = torch.cat([image_proj, text_proj], dim=1)
        
        # Apply GNN layers to individual subgraphs
        node_features = []
        for i, (edge_index, edge_attr) in enumerate(zip(edge_indices, edge_attrs)):
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            
            # Apply each GNN layer
            node_x = x[i].unsqueeze(0)
            for conv in self.gnn_layers:
                node_x = conv(node_x, edge_index, edge_attr=edge_attr)
                node_x = F.relu(node_x)
            
            node_features.append(node_x.squeeze(0))
        
        # Stack node features
        if node_features:
            x = torch.stack(node_features)
        
        # Dropout and pooling
        x = F.dropout(x, p=0.2, training=self.training)
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        x = self.bn_classifier(x)
        
        return x, image_features, text_features

class MultiModalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, temperature: float = 0.07):
        """
        Multi-modal loss combining classification and contrastive losses
        Args:
            alpha: Weight for classification loss
            beta: Weight for contrastive loss
            temperature: Temperature for contrastive loss
        """
        super(MultiModalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Contrastive loss
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity scores
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        # Labels for contrastive loss (diagonal entries should match)
        contrastive_labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        
        # Compute losses in both directions
        loss_i = self.ce_loss(logits_per_image, contrastive_labels)
        loss_t = self.ce_loss(logits_per_text, contrastive_labels)
        contrastive_loss = (loss_i + loss_t) / 2
        
        # Combined loss
        return self.alpha * cls_loss + self.beta * contrastive_loss

class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, edge_index, edge_attr, labels, num_hops=1, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.labels = labels
        self.num_hops = num_hops
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and process image
        if self.image_paths[idx] and os.path.exists(self.image_paths[idx]):
            try:
                image = Image.open(self.image_paths[idx]).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)
        
        # Extract subgraph around node
        node_idx = torch.tensor([idx])
        subset, sub_edge_index, sub_edge_attr, _ = k_hop_subgraph(
            node_idx,
            num_hops=self.num_hops,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            relabel_nodes=True
        )
        
        return {
            'image': image,
            'text': self.texts[idx],
            'edge_index': sub_edge_index,
            'edge_attr': sub_edge_attr,
            'label': self.labels[idx]
        }

def create_fine_tuning_dataloaders(dataset_path: str, batch_size: int = 16,
                                  num_hops: int = 1) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create dataloaders for fine-tuning
    Args:
        dataset_path: Path to dataset
        batch_size: Batch size
        num_hops: Number of hops for subgraph extraction
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_predicates: Number of unique predicates (edge types)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory {dataset_path} does not exist")
    
    # Load data
    with open(os.path.join(dataset_path, 'entities.json')) as f:
        entities = json.load(f)
    
    triples = pd.read_csv(os.path.join(dataset_path, 'triples.txt'),
                         sep='\t', names=['subject', 'predicate', 'object'])
    
    # Create mappings
    entity_id_to_idx = {str(e['id']): i for i, e in enumerate(entities)}
    predicate_to_idx = {p: i for i, p in enumerate(triples['predicate'].unique())}
    
    # Build edge index and attributes
    edge_index = []
    edge_attr = []
    for _, row in triples.iterrows():
        if str(row['subject']) in entity_id_to_idx and str(row['object']) in entity_id_to_idx:
            edge_index.append([entity_id_to_idx[str(row['subject'])],
                             entity_id_to_idx[str(row['object'])]])
            edge_attr.append(predicate_to_idx[row['predicate']])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    # Prepare labels and paths
    labels = torch.tensor([e.get('label', 0) for e in entities])
    image_paths = []
    texts = []
    for entity in entities:
        img_path = os.path.join(dataset_path, 'images', entity['image']) \
                  if 'image' in entity and entity['image'] else None
        image_paths.append(img_path)
        texts.append(entity.get('description', ''))
    
    # Create dataset
    dataset = FineTuningDataset(image_paths, texts, edge_index, edge_attr, labels, num_hops=num_hops)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, len(predicate_to_idx)

def finetune_model(model: JointFineTunedModel, train_loader: DataLoader, val_loader: DataLoader,
                  num_epochs: int, learning_rate: float, device: torch.device,
                  gnn_type: str = 'gat', feature_type: str = 'simple',
                  results_dir: str = 'finetuning_results', patience: int = 10) -> None:
    """
    Fine-tune the model
    Args:
        model: Model to fine-tune
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        gnn_type: Type of GNN used in model
        feature_type: Type of features used
        results_dir: Directory to save results
        patience: Early stopping patience
    """
    os.makedirs(results_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model configuration
    config_file = os.path.join(run_dir, f"{gnn_type}_{feature_type}_config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'gnn_type': gnn_type,
            'feature_type': feature_type,
            'hidden_dim': model.gnn_layers[0].in_channels // 2,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Initialize optimizer, criterion, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MultiModalLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Initialize metrics tracking
    best_val_acc = 0
    trigger_times = 0
    metrics_history = {
        'epochs': [],
        'train_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': [],
        'gnn_type': gnn_type,
        'feature_type': feature_type
    }
    
    # Use mixed precision for CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training loop
    print(f"Starting fine-tuning {gnn_type} model with {feature_type} features for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # Prepare data
            images = batch['image'].to(device)
            texts = batch['text']
            edge_indices = batch['edge_index']
            edge_attrs = batch['edge_attr']
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, img_feats, txt_feats = model(
                        images, texts, edge_indices, edge_attrs,
                        torch.zeros(len(images), dtype=torch.long).to(device)
                    )
                    
                    loss = criterion(logits, labels, img_feats, txt_feats)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                logits, img_feats, txt_feats = model(
                    images, texts, edge_indices, edge_attrs,
                    torch.zeros(len(images), dtype=torch.long).to(device)
                )
                
                loss = criterion(logits, labels, img_feats, txt_feats)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Prepare data
                images = batch['image'].to(device)
                texts = batch['text']
                edge_indices = batch['edge_index']
                edge_attrs = batch['edge_attr']
                labels = batch['label'].to(device)
                
                # Forward pass
                logits, _, _ = model(
                    images, texts, edge_indices, edge_attrs,
                    torch.zeros(len(images), dtype=torch.long).to(device)
                )
                
                # Compute accuracy
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                # Store predictions for F1 calculation
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = correct / total
        
        # Calculate F1 score
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Update metrics history
        metrics_history['epochs'].append(epoch + 1)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['val_f1'].append(val_f1)
        metrics_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Save metrics to file
        metrics_file = os.path.join(run_dir, f"{gnn_type}_{feature_type}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # Create metrics plot
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(metrics_history['epochs'], metrics_history['train_loss'])
            plt.title(f'Training Loss - {gnn_type} with {feature_type} features')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(2, 2, 2)
            plt.plot(metrics_history['epochs'], metrics_history['val_acc'], label='Accuracy')
            plt.plot(metrics_history['epochs'], metrics_history['val_f1'], label='F1')
            plt.title(f'Validation Metrics - {gnn_type} with {feature_type} features')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(metrics_history['epochs'], metrics_history['learning_rates'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'{gnn_type}_{feature_type}_metrics.png'))
            plt.close()
        except Exception as e:
            print(f"Failed to create metrics plot: {e}")
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, f'{gnn_type}_{feature_type}_best_model.pth'))
            trigger_times = 0
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(run_dir, f'{gnn_type}_{feature_type}_final_model.pth'))
    
    # Print training summary
    print(f"Fine-tuning completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to {run_dir}")

def run_finetuning(dataset_path: str, gnn_type: str = 'gat', feature_type: str = 'simple',
                  batch_size: int = 16, num_epochs: int = 50, learning_rate: float = 1e-5, 
                  hidden_dim: int = 256, patience: int = 10, output_dir: Optional[str] = None) -> None:
    """
    Run full fine-tuning pipeline
    Args:
        dataset_path: Path to dataset
        gnn_type: Type of GNN to use ('gat', 'gcn', 'sage', 'transformer')
        feature_type: Type of features ('simple', 'huggingface', 'intermediate')
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        hidden_dim: Hidden dimension size
        patience: Early stopping patience
        output_dir: Directory to save results
    """
    # Set up output directory with GNN and feature type info
    if output_dir is None:
        output_dir = f"finetuning_results/{gnn_type}_{feature_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else
                        'cpu')
    
    print(f"Using device: {device}")
    print(f"Fine-tuning with GNN type: {gnn_type}, Feature type: {feature_type}")
    
    # Create dataloaders
    train_loader, val_loader, num_predicates = create_fine_tuning_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size
    )
    
    print(f"Created dataloaders - Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")
    
    # Create model
    model = JointFineTunedModel(
        num_classes=10,
        gnn_type=gnn_type,
        hidden_dim=hidden_dim,
        edge_attr_dim=num_predicates
    ).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model created with {hidden_dim} hidden dimensions")
    print(f"GNN type: {gnn_type}, Feature type: {feature_type}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Save model information
    with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
        json.dump({
            'gnn_type': gnn_type,
            'feature_type': feature_type,
            'hidden_dim': hidden_dim,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Fine-tune model
    finetune_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        gnn_type=gnn_type,
        feature_type=feature_type,
        device=device,
        results_dir=output_dir,
        patience=patience
    )

if __name__ == '__main__':
    run_finetuning(
        dataset_path='dmg777k_dataset',
        batch_size=16,
        num_epochs=50,
        learning_rate=1e-5
    )
