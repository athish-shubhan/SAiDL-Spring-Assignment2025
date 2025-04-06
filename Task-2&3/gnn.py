import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, BatchNorm1d, Dropout
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv, global_mean_pool
from typing import Dict, Tuple, List, Union, Optional

# Add standalone implementations for GCN and GraphSAGE
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class CrossModalAttention(torch.nn.Module):
    def __init__(self, dim: int):
        """
        Cross-modal attention mechanism
        
        Args:
            dim: Dimension of input features
        """
        super(CrossModalAttention, self).__init__()
        self.query = Linear(dim, dim)
        self.key = Linear(dim, dim)
        self.value = Linear(dim, dim)
        self.scale = dim ** -0.5
        self.dropout = Dropout(0.1)
        self.bn_fusion = BatchNorm1d(dim)
        
    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, 
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of cross-modal attention"""
        # Simple fusion if no edge index provided
        if edge_index is None or edge_index.numel() == 0:
            combined = image_emb + text_emb
            return self.dropout(F.relu(self.bn_fusion(combined)))
        
        # Attention mechanism
        query = self.query(image_emb)
        key = self.key(text_emb)
        value = self.value(text_emb)
        
        # Compute attention scores and weights
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention weights
        attended_text = torch.matmul(attn_probs, value)
        
        return attended_text

class MultiModalGNN(torch.nn.Module):
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int, 
                 num_classes: int, gnn_type: str = 'gat', dropout_rate: float = 0.2, 
                 num_gnn_layers: int = 2, edge_attr_dim: int = 0):
        """
        Multi-modal GNN for node classification
        
        Args:
            image_dim: Dimension of image features
            text_dim: Dimension of text features
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            gnn_type: Type of GNN layer ('gat', 'gcn', 'sage', 'transformer')
            dropout_rate: Dropout rate
            num_gnn_layers: Number of GNN layers
            edge_attr_dim: Dimension of edge attributes (0 if none)
        """
        super(MultiModalGNN, self).__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers for each modality
        self.image_proj = Linear(image_dim, hidden_dim)
        self.text_proj = Linear(text_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(hidden_dim)
        self.bn_fusion = BatchNorm1d(hidden_dim)
        self.dropout = Dropout(dropout_rate)
        
        # GNN layers
        self.gnn_type = gnn_type
        self.convs = ModuleList()
        
        for i in range(num_gnn_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim
            
            if gnn_type == 'gat':
                self.convs.append(GATConv(in_channels, out_channels, heads=4, 
                                         dropout=dropout_rate, edge_dim=edge_attr_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_channels, out_channels))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_channels, out_channels))
            elif gnn_type == 'transformer':
                self.convs.append(TransformerConv(in_channels, out_channels, edge_dim=edge_attr_dim))
            else:
                raise ValueError(f"Invalid GNN type: {gnn_type}")
        
        # Classifier
        self.classifier = Linear(hidden_dim, num_classes)
        self.bn_classifier = BatchNorm1d(num_classes)
    
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model"""
        device = next(self.parameters()).device
        
        # Handle edge_index issues
        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.zeros(2, 1, dtype=torch.long, device=device)
        elif edge_index.dim() == 1:
            edge_index = edge_index.view(2, -1)
        
        edge_index = edge_index.to(device)
        
        # Handle input features
        if isinstance(x, torch.Tensor) and x.dim() == 2:
            # Input is a concatenated tensor - split into modalities
            image_x = x[:, :self.image_dim].to(device)
            text_x = x[:, self.image_dim:].to(device)
        else:
            # Input is already split
            image_x, text_x = [t.to(device) for t in x]
        
        # Project features
        x_dtype = x.dtype
        self.image_proj.weight.data = self.image_proj.weight.data.to(x_dtype)
        self.image_proj.bias.data = self.image_proj.bias.data.to(x_dtype)
        image_emb = F.relu(self.image_proj(image_x))
        text_emb = F.relu(self.text_proj(text_x))
        
        # Cross-modal attention
        attended_text = self.cross_attention(image_emb, text_emb, edge_index)
        
        # Combine modalities
        x = image_emb + attended_text
        x = F.relu(self.bn_fusion(x))
        x = self.dropout(x)
        
        # Apply GNN layers
        for conv in self.convs:
            if self.gnn_type in ['gat', 'transformer'] and edge_attr is not None:
                x = F.relu(conv(x, edge_index, edge_attr=edge_attr))
            else:
                x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        
        # Global pooling if batch indices provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        if x.size(0) > 1:  # Apply batch norm only for batches
            x = self.bn_classifier(x)
        
        return x
