import torch
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from torch_geometric.utils import to_networkx
from collections import defaultdict
from typing import List, Dict, Any, Optional

class DynamicSubsetStrategies:
    def __init__(self, data: Any, device: torch.device):
        """
        Dynamic subset strategies for handling large datasets
        Args:
            data: Graph data
            device: Computation device
        """
        self.data = data
        self.device = device
        self.strategy_history = []
        self.grad_history = defaultdict(list)
        
        # Initialize graph structure from data
        self.G = to_networkx(data, to_undirected=True)
        self.node_degrees = dict(self.G.degree())
        
        # Available strategies
        self.strategies = [
            self.adaptive_khop_sampling,
            self.gradient_aware_coreset,
            self.modality_curriculum
        ]
        
        self.strategy_names = [
            "Adaptive k-Hop Sampling",
            "Gradient-Aware Coreset",
            "Modality Curriculum"
        ]
        
        # Initialize contrastive scores
        self.contrastive_scores = self._init_contrastive_scores()
        
    def _init_contrastive_scores(self) -> np.ndarray:
        """Initialize contrastive scores between modalities"""
        # Extract features
        image_features = self.data.x[:, :512].cpu().numpy()
        text_features = self.data.x[:, 512:].cpu().numpy()
        
        # Normalize features
        image_features_norm = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)
        text_features_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
        
        # Simple score based on feature magnitude
        return np.sum(np.abs(image_features_norm), axis=1) + np.sum(np.abs(text_features_norm), axis=1)
        
    def adaptive_khop_sampling(self, current_size: int, time_exceeded: float) -> List[int]:
        """
        Sample a diverse set of nodes with k-hop connectivity
        Args:
            current_size: Target subset size
            time_exceeded: Seconds exceeding threshold
        Returns:
            List of selected node indices
        """
        # Dynamic k based on time constraints
        k = 2 if time_exceeded < 300 else 1
        
        # Cluster nodes using multimodal features
        kmeans = KMeans(n_clusters=5).fit(self.data.x.cpu().numpy())
        
        # Select representative nodes from each cluster
        subset = []
        for cluster_id in range(5):
            cluster_nodes = np.where(kmeans.labels_ == cluster_id)[0]
            if len(cluster_nodes) > 0:
                scores = self.contrastive_scores[cluster_nodes]
                top_indices = cluster_nodes[np.argsort(scores)[-max(1, current_size // 5):]]
                subset.extend(top_indices.tolist())
                
        # Add k-hop neighbors for connectivity
        neighbors = set()
        for node in subset:
            # Use networkx descendants at distance k
            try:
                neighbors.update(nx.descendants_at_distance(self.G, node, k))
            except:
                # Fallback if node not in graph
                continue
                
        # Combine and limit size
        return list(set(subset + list(neighbors)))[:current_size]
        
    def gradient_aware_coreset(self, model: torch.nn.Module, current_size: int) -> List[int]:
        """
        Sample nodes based on gradient magnitudes
        Args:
            model: Current model
            current_size: Target subset size
        Returns:
            List of selected node indices
        """
        model.train()
        gradients = []
        
        # Compute gradient for each node
        for idx in range(min(len(self.data.x), 1000)):  # Limit computation
            x = self.data.x[idx].unsqueeze(0).to(self.device)
            edge_index = self.data.edge_index.to(self.device)
            model.zero_grad()
            output = model(x, edge_index)
            
            # Use cross-entropy loss for the gradient
            loss = torch.nn.functional.cross_entropy(
                output,
                self.data.y[idx].unsqueeze(0).to(self.device)
            )
            
            loss.backward()
            
            # Record gradient norm
            gradients.append(torch.norm(x.grad).item())
            
        # Select nodes with highest gradients
        top_indices = np.argsort(gradients)[-current_size:]
        return top_indices.tolist()
        
    def modality_curriculum(self, current_size: int, time_exceeded: float = 0.0) -> List[int]:
        """
        Progressive modality introduction strategy
        Args:
            current_size: Target subset size
        Returns:
            List of selected node indices
        """
        # Define stage thresholds
        stages = [
            (0.8, 0.8),  # Both modalities
            (0.6, 0.6),  # Either modality
            (0.4, 0.4)   # Any nodes
        ]
        
        # Determine current stage based on history
        stage = min(len(self.strategy_history) // 3, 2)
        img_thresh, txt_thresh = stages[stage]
        
        # Score nodes by modality completeness
        scores = []
        for idx in range(len(self.data.x)):
            img_score = 1 if self.data.x[idx, :512].sum() > 0 else 0
            txt_score = 1 if self.data.x[idx, 512:].sum() > 0 else 0
            scores.append(img_score + txt_score)
            
        # Sort and select based on scores
        sorted_indices = np.argsort(scores)[::-1]
        subset = sorted_indices[:int(current_size * (img_thresh + txt_thresh) / 2)]
        
        # Add random nodes if needed
        remaining = current_size - len(subset)
        if remaining > 0:
            candidates = list(set(range(len(self.data.x))) - set(subset))
            if candidates:
                random_indices = np.random.choice(candidates,
                                                min(remaining, len(candidates)),
                                                replace=False)
                subset = np.concatenate([subset, random_indices])
                
        return subset[:current_size].tolist()
        
    def update_strategy(self, model: torch.nn.Module, current_size: int,
                       epoch_time: float) -> List[int]:
        """
        Dynamically select and apply a subset strategy
        Args:
            model: Current model
            current_size: Target subset size
            epoch_time: Time taken for last epoch
        Returns:
            List of selected node indices
        """
        # Select strategy based on current conditions
        if epoch_time > 120:
            # Use simpler strategy when training is too slow
            strategy = self.adaptive_khop_sampling
            strategy_idx = 0
        elif len(self.strategy_history) < 5:
            # Use curriculum in early training
            strategy = self.modality_curriculum
            strategy_idx = 2
        else:
            # Select based on gradient history
            strategy_idx = np.argmin([
                np.mean(self.grad_history[0]) if len(self.grad_history[0]) > 0 else float('inf'),
                np.mean(self.grad_history[1]) if len(self.grad_history[1]) > 0 else float('inf'),
                np.mean(self.grad_history[2]) if len(self.grad_history[2]) > 0 else float('inf')
            ])
            strategy = self.strategies[strategy_idx]
            
        # Apply selected strategy
        if strategy_idx == 1:  # Gradient-aware needs model
            subset = self.gradient_aware_coreset(model, current_size)
        else:
            subset = strategy(current_size, epoch_time)
            
        # Record strategy
        self.strategy_history.append(strategy_idx)
        print(f"Selected strategy: {self.strategy_names[strategy_idx]} with subset size {current_size}")
        
        # Save strategy information
        self._save_strategy_info(strategy_idx, current_size, epoch_time)
        
        return subset
        
    def _save_strategy_info(self, strategy_idx: int, subset_size: int, epoch_time: float) -> None:
        """Save information about the selected strategy"""
        import os
        import json
        from datetime import datetime
        
        os.makedirs("subset_strategies", exist_ok=True)
        
        # Create or load strategy history
        history_file = "subset_strategies/strategy_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            history = {
                "strategy_selections": [],
                "strategy_names": self.strategy_names
            }
            
        # Add new strategy selection
        history["strategy_selections"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": self.strategy_names[strategy_idx],
            "strategy_idx": strategy_idx,
            "subset_size": subset_size,
            "epoch_time": epoch_time
        })
        
        # Save updated history
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

class DynamicSubsetHandler:
    def __init__(self, dataset: Any, device: torch.device):
        """
        Handler for dynamic subset creation
        Args:
            dataset: Full dataset
            device: Computation device
        """
        self.full_dataset = dataset
        self.current_subset = list(range(len(dataset)))
        
        # Create strategy object
        self.strategy = DynamicSubsetStrategies(dataset, device)
        
        # Create sequence of decreasing sizes
        self.subset_sizes = self._generate_size_sequence(len(dataset))
        self.current_size_idx = 0
        
        # Store timing and performance data
        self.epoch_times = []
        self.subset_history = []
        
    def _generate_size_sequence(self, initial_size: int) -> List[int]:
        """Generate geometric sequence of subset sizes"""
        sizes = [initial_size]
        while sizes[-1] > 1000:
            sizes.append(int(sizes[-1] * 0.6))
        return sizes
        
    def create_subset(self, model: torch.nn.Module, epoch_time: float) -> torch.utils.data.Subset:
        """
        Create new subset based on current conditions
        Args:
            model: Current model
            epoch_time: Time taken for last epoch
        Returns:
            Subset of the dataset
        """
        self.epoch_times.append(epoch_time)
        
        # Reduce subset size if training is too slow
        if epoch_time > 120 and self.current_size_idx < len(self.subset_sizes) - 1:
            self.current_size_idx += 1
            
        # Get target size and create subset
        target_size = self.subset_sizes[self.current_size_idx]
        new_subset = self.strategy.update_strategy(model, target_size, epoch_time)
        
        # Save state and return subset
        self.current_subset = new_subset
        self.subset_history.append((len(new_subset), self.strategy.strategy_history[-1]))
        
        print(f"Created new subset with size {len(new_subset)}")
        return torch.utils.data.Subset(self.full_dataset, new_subset)
        
    def get_history(self) -> Dict[str, List]:
        """Get history of subset changes and strategies"""
        return {
            'epoch_times': self.epoch_times,
            'subset_sizes': [s[0] for s in self.subset_history],
            'strategies': [s[1] for s in self.subset_history],
            'strategy_names': self.strategy.strategy_names
        }
        
    def save_history(self) -> None:
        """Save subset history to disk"""
        import os
        import json
        
        os.makedirs("subset_strategies", exist_ok=True)
        history = self.get_history()
        
        with open("subset_strategies/subset_history.json", "w") as f:
            json.dump(history, f, indent=2)
