import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
import ssl
from contextlib import nullcontext
from gnn import MultiModalGNN, GCN, GraphSAGE
from subset_strategy import DynamicSubsetHandler

# Parse arguments
parser = argparse.ArgumentParser(description="Train GNN for DMG777K")
parser.add_argument("--gnn_type", type=str, default="gcn",
                    choices=["gat", "gcn", "sage", "transformer", "clip_gnn"],
                    help="GNN architecture to use")
parser.add_argument("--feature_type", type=str, default="simple",
                    choices=["simple", "huggingface", "intermediate"],
                    help="Feature extraction method")
parser.add_argument("--output_dir", type=str, default="training_results",
                    help="Directory to save results")
parser.add_argument("--epochs", type=int, default=20,
                    help="Maximum number of training epochs")
parser.add_argument("--virtual_batch_size", type=int, default=8,
                    help="Virtual batch size for gradient accumulation")
args = parser.parse_args()

# Fixed hyperparameters (previously tuned with Optuna)
HYPERPARAMS = {
    "lr": 3e-4,
    "dropout": 0.3,
    "hidden_dim": 128
}

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

def check_dataset_balance(dataset, using_kgbench=False):
    """Check class distribution and non-zero feature counts"""
    try:
        if using_kgbench and hasattr(dataset, 'num_entities') and hasattr(dataset, 'training'):
            print(f"KGBench dataset with {dataset.num_entities} entities")
            # Get label distribution
            all_labels = []
            for i in range(len(dataset.training)):
                all_labels.append(dataset.training[i, 1].item())
            for i in range(len(dataset.withheld)):
                all_labels.append(dataset.withheld[i, 1].item())
            if all_labels:
                unique_labels, counts = np.unique(all_labels, return_counts=True)
                print(f"Class distribution: {dict(zip(unique_labels, counts))}")
        elif hasattr(dataset, 'data') and hasattr(dataset.data, 'y'):
            labels = dataset.data.y
            label_counts = torch.bincount(labels, minlength=10)
            print(f"Class distribution: {label_counts}")
        elif hasattr(dataset, 'data_list') and len(dataset.data_list) > 0:
            data = dataset.data_list[0]
            if hasattr(data, 'y'):
                label_counts = torch.bincount(data.y, minlength=10)
                print(f"Class distribution: {label_counts}")
                
            # Feature statistics if available
            if hasattr(data, 'x'):
                # Split features in half for image and text
                mid_point = data.x.shape[1] // 2
                img_features = data.x[:, :mid_point]
                txt_features = data.x[:, mid_point:]
                nonzero_img = (img_features.sum(dim=1) != 0).sum().item()
                nonzero_txt = (txt_features.sum(dim=1) != 0).sum().item()
                print(f"Non-zero image features: {nonzero_img}/{len(data.x)}")
                print(f"Non-zero text features: {nonzero_txt}/{len(data.x)}")
    except Exception as e:
        print(f"Error checking dataset balance: {e}")

def convert_kgbench_to_pyg(kg_dataset, features=None):
    """Convert KGBench dataset to PyTorch Geometric Data format"""
    device = torch.device('cpu') # Use CPU for conversion to save GPU memory
    
    # Create edge index from triples
    edge_index = []
    for s, p, o in kg_dataset.triples:
        s_val = s.item() if hasattr(s, 'item') else s
        o_val = o.item() if hasattr(o, 'item') else o
        edge_index.append([s_val, o_val])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Use provided features or create empty ones
    if features is not None and isinstance(features, dict) and 'image_features' in features and 'text_features' in features:
        try:
            # Check for dimension mismatch and handle it
            if features['image_features'].size(0) != features['text_features'].size(0):
                print(f"Feature dimension mismatch: {features['image_features'].size(0)} vs {features['text_features'].size(0)}")
                min_size = min(features['image_features'].size(0), features['text_features'].size(0))
                features['image_features'] = features['image_features'][:min_size]
                features['text_features'] = features['text_features'][:min_size]
            
            # Concatenate features
            x = torch.cat([features['image_features'], features['text_features']], dim=1)
        except Exception as e:
            print(f"Error concatenating features: {e}")
            print("Using random features instead")
            x = torch.randn(kg_dataset.num_entities, 1024)
    else:
        print(f"Features not provided or invalid. Using random features for {kg_dataset.num_entities} entities")
        x = torch.randn(kg_dataset.num_entities, 1024)
    
    # Create labels tensor for all nodes
    y = torch.full((kg_dataset.num_entities,), -1, dtype=torch.long)
    
    # Fill in known labels from training data
    for i in range(len(kg_dataset.training)):
        node_idx = kg_dataset.training[i, 0].item()
        label = kg_dataset.training[i, 1].item()
        if node_idx < kg_dataset.num_entities:  # Add validation check
            y[node_idx] = label
    
    # Fill in known labels from withheld data
    for i in range(len(kg_dataset.withheld)):
        node_idx = kg_dataset.withheld[i, 0].item()
        label = kg_dataset.withheld[i, 1].item()
        if node_idx < kg_dataset.num_entities:  # Add validation check
            y[node_idx] = label
    
    # Create train/test masks
    train_mask = torch.zeros(kg_dataset.num_entities, dtype=torch.bool)
    test_mask = torch.zeros(kg_dataset.num_entities, dtype=torch.bool)
    
    # Set train mask for training nodes
    for i in range(len(kg_dataset.training)):
        node_idx = kg_dataset.training[i, 0].item()
        if node_idx < kg_dataset.num_entities:  # Add validation check
            train_mask[node_idx] = True
    
    # Set test mask for withheld nodes
    for i in range(len(kg_dataset.withheld)):
        node_idx = kg_dataset.withheld[i, 0].item()
        if node_idx < kg_dataset.num_entities:  # Add validation check
            test_mask[node_idx] = True
    
    # Create the PyG data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data

def train_model(model, train_loader, optimizer, criterion, device, subset_handler=None):
    """Train model for one epoch"""
    accelerator_active = False
    try:
        from kaggle import accelerate
        if hasattr(model, 'module'):
            model.module = accelerate(model.module, precision='mixed')
        else:
            model = accelerate(model, precision='mixed')
        accelerator_active = True
        print("✓ Kaggle accelerator enabled with mixed precision")
    except (ImportError, Exception) as e:
        print(f"Kaggle accelerator not available: {e}")
        if torch.cuda.is_available():
            print("Using PyTorch native mixed precision instead")
    
    model.train()
    total_loss = 0
    start_time = time.time()
    all_preds = []
    all_labels = []
    
    # Use smaller batch size for gradient accumulation if needed
    virtual_batch_size = args.virtual_batch_size
    actual_batch_size = train_loader.batch_size
    accumulation_steps = max(1, actual_batch_size // virtual_batch_size)
    print(f"Using gradient accumulation: {accumulation_steps} steps (virtual batch: {virtual_batch_size})")
    
    optimizer_steps = 0
    
    # Memory-efficient mixed precision
    use_mixed_precision = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
    
    for i, data in enumerate(train_loader):
        # Zero gradients only at the beginning of accumulation
        if i % accumulation_steps == 0:
            optimizer.zero_grad()
        
        data = data.to(device)
        
        # Handle edge index dimensions
        if data.edge_index.dim() == 1:
            data.edge_index = data.edge_index.view(2, -1)
        elif data.edge_index.numel() == 0:
            data.edge_index = torch.zeros(2, 1, dtype=torch.long, device=device)
        
        try:
            # Mixed precision path for CUDA
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    out = model(data.x, data.edge_index)
                    loss = criterion(out, data.y) / accumulation_steps
                    
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_steps += 1
            else:
                # Standard path for CPU/MPS
                out = model(data.x, data.edge_index)
                loss = criterion(out, data.y) / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer_steps += 1
                    
            total_loss += loss.item() * accumulation_steps * data.num_graphs
            
            # Collect predictions
            pred = out.max(1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    epoch_time = time.time() - start_time
    
    # Calculate metrics
    if len(all_preds) > 0:
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            print(f"Train metrics - Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            f1 = 0.0
    else:
        f1 = 0.0
    
    # Dynamic subset handling based on epoch time
    if subset_handler is not None and epoch_time > 120:
        prev_size = len(subset_handler.current_subset)
        new_subset = subset_handler.create_subset(model, epoch_time)
        new_size = len(subset_handler.current_subset)
        strategy_name = subset_handler.strategy.strategy_names[
            subset_handler.strategy.strategy_history[-1]
        ]
        print(f"⏱️ Epoch time {epoch_time:.1f}s > 120s")
        print(f"Strategy: {strategy_name}")
        print(f"Reduced subset: {prev_size} → {new_size} samples")
    
    return total_loss / max(1, len(train_loader.dataset)), epoch_time, f1

def evaluate_model(model, loader, device):
    """Evaluate model on test data"""
    # Clean up GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Handle edge index dimensions
            if data.edge_index.dim() == 1:
                data.edge_index = data.edge_index.view(2, -1)
            elif data.edge_index.numel() == 0:
                data.edge_index = torch.zeros(2, 1, dtype=torch.long, device=device)
            
            # Forward pass
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                logits = model(data.x, data.edge_index)
                pred = logits.max(1)[1]
            
            # Update metrics
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    try:
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    print(f"Evaluation results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Create results directory
    output_dir = args.output_dir
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{args.gnn_type}_{args.feature_type}_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load dataset with proper KGBench handling
    try:
        import kgbench as kg
        dataset = kg.load('dmg777k', torch=True)
        using_kgbench = True
        print(f"Loaded KGBench dataset with {dataset.num_entities} entities and {len(dataset.triples)} triples")
    except Exception as e:
        print(f"Error loading with KGBench: {e}")
        try:
            from torch_geometric.datasets import DMG777KDataset
            dataset = DMG777KDataset(root='./dmg777k_dataset')
            using_kgbench = False
            print("Using PyTorch Geometric dataset")
        except Exception as e2:
            print(f"Error loading with PyTorch Geometric: {e2}")
            raise RuntimeError("Could not load dataset with either KGBench or PyTorch Geometric")
    
    # Check dataset characteristics
    check_dataset_balance(dataset, using_kgbench)
    
    # Load features based on feature type
    if args.feature_type:
        features_path = f"extracted_features/{args.feature_type}_features.pt"
        if os.path.exists(features_path):
            print(f"Loading {args.feature_type} features")
            features = torch.load(features_path, weights_only=True)  # Added weights_only=True for security
            current_features = features
        else:
            print(f"Features file not found: {features_path}")
            current_features = None
    else:
        current_features = None
    
    # Convert KGBench dataset to PyTorch Geometric format if needed
    if using_kgbench:
        data = convert_kgbench_to_pyg(dataset, current_features)
    else:
        # Handle PyG dataset
        if hasattr(dataset, 'process'):
            data_list = dataset.process()
            data = data_list[0] if isinstance(data_list, list) else data_list
        else:
            # Dataset is already a Data object or similar
            if hasattr(dataset, 'data'):
                data = dataset.data
            elif isinstance(dataset, list):
                data = dataset[0]
            else:
                data = dataset
    
    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Validate and fix edge index
    max_node_idx = data.x.size(0) - 1
    edge_index = data.edge_index
    edge_mask = (edge_index[0] <= max_node_idx) & (edge_index[1] <= max_node_idx)
    
    if edge_mask.sum() < 100:
        # Create simple connected structure to ensure graph connectivity
        print(f"Too few valid edges ({edge_mask.sum()}), adding minimal connectivity")
        # Create a simple chain connecting sequential nodes
        new_edges = torch.stack([
            torch.arange(0, min(1000, max_node_idx), device=edge_index.device),
            torch.arange(1, min(1001, max_node_idx + 1), device=edge_index.device)
        ], dim=0)
        # Update edge_index with valid edges
        edge_index = new_edges
    else:
        edge_index = edge_index[:, edge_mask]
    
    print(f"Edge index contains {edge_index.size(1)} valid edges")
    data.edge_index = edge_index
    
    # Validate and fix masks - CRITICAL FIX FOR KeyError
    max_idx = data.x.size(0) - 1
    
    # Fix train mask - ensure no indices exceed max_idx
    train_mask_indices = torch.where(data.train_mask)[0]
    valid_train_indices = train_mask_indices[train_mask_indices <= max_idx]
    new_train_mask = torch.zeros_like(data.train_mask)
    new_train_mask[valid_train_indices] = True
    data.train_mask = new_train_mask
    
    # Fix test mask similarly
    test_mask_indices = torch.where(data.test_mask)[0]
    valid_test_indices = test_mask_indices[test_mask_indices <= max_idx]
    new_test_mask = torch.zeros_like(data.test_mask)
    new_test_mask[valid_test_indices] = True
    data.test_mask = new_test_mask
    
    print(f"Fixed masks: train={valid_train_indices.size(0)}/{train_mask_indices.size(0)}, test={valid_test_indices.size(0)}/{test_mask_indices.size(0)}")
    
    # Get dimensions for model
    num_nodes = data.x.size(0)
    input_dim = data.x.size(1)
    image_dim = min(512, input_dim // 2)  # Handle different feature sizes
    text_dim = input_dim - image_dim
    
    # Get number of classes
    num_classes = 5  # Default for DMG777K
    if using_kgbench:
        unique_classes = torch.unique(dataset.training[:, 1])
        num_classes = len(unique_classes)
    elif hasattr(dataset, 'num_classes'):
        num_classes = dataset.num_classes
    
    print(f"Task has {num_classes} classes")
    print(f"Input dimensions: total={input_dim}, image={image_dim}, text={text_dim}")
    
    # Create model
    model = MultiModalGNN(
        image_dim=image_dim,
        text_dim=text_dim,
        hidden_dim=HYPERPARAMS["hidden_dim"],
        num_classes=num_classes,
        gnn_type=args.gnn_type,
        dropout_rate=HYPERPARAMS["dropout"]
    ).to(device)
    
    print(f"Created {args.gnn_type} model with {HYPERPARAMS['hidden_dim']} hidden dimensions")
    
    # Setup subset handler
    subset_handler = DynamicSubsetHandler(data, device)
    initial_subset = subset_handler.create_subset(model, 0.0)
    
    # Move data to device
    data = data.to(device)
    
    # Create data loaders - FIXED INDICES FOR THE KEYERROR
    train_indices = torch.where(data.train_mask)[0].tolist()
    # Filter any indices that might be out of bounds
    train_indices = [idx for idx in train_indices if idx < data.x.size(0)]
    train_subset = torch.utils.data.Subset(data, train_indices)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    
    # Create test loader similarly
    test_indices = torch.where(data.test_mask)[0].tolist()
    # Filter any indices that might be out of bounds
    test_indices = [idx for idx in test_indices if idx < data.x.size(0)]
    test_subset = torch.utils.data.Subset(data, test_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # Training loop
    best_acc = 0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    metrics_history = []
    
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss, epoch_time, train_f1 = train_model(
            model, train_loader, optimizer, criterion, device, subset_handler
        )
        
        # Evaluation
        metrics = evaluate_model(model, test_loader, device)
        test_acc = metrics['accuracy']
        
        # Update scheduler
        scheduler.step()
        
        # Save metrics for this epoch
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": metrics['f1'],
            "test_precision": metrics['precision'],
            "test_recall": metrics['recall'],
            "epoch_time": epoch_time
        }
        
        metrics_history.append(epoch_metrics)
        
        with open(os.path.join(run_dir, f"metrics_epoch{epoch+1}.json"), "w") as f:
            json.dump(epoch_metrics, f, indent=2)
        
        # Save model checkpoint if improved
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            print(f"New best accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final metrics
    final_metrics = {
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "hyperparameters": HYPERPARAMS,
        "gnn_type": args.gnn_type,
        "feature_type": args.feature_type,
        "all_metrics": metrics_history
    }
    
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"Training complete! Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Results saved to {run_dir}")

if __name__ == "__main__":
    main()
