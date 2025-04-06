import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from transformers import CLIPProcessor, CLIPModel

class CLIPGNN(nn.Module):
    def __init__(self, hidden_dim, num_classes, gnn_type='gcn'):
        super(CLIPGNN, self).__init__()
        
        # Load pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embedding_cache = {}

        
        # Freeze CLIP parameters (use CLIP as a feature extractor)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # CLIP embedding dimension is typically 512
        self.clip_dim = self.clip.projection_dim
        
        # GNN layers to process the graph structure
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(self.clip_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:  # sage
            self.conv1 = SAGEConv(self.clip_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            
        # Output classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Temperature parameter (like in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
        
    def extract_image_features(self, images):
        """Extract image features using CLIP's image encoder"""
        # Check cache first
        cache_key = str(images)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        # Process and cache result
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.clip.device)
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                features = self.clip.get_image_features(**inputs)
            self.embedding_cache[cache_key] = features
            return features

        
    def extract_text_features(self, texts):
        """Extract text features using CLIP's text encoder"""
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.clip.device)
            features = self.clip.get_text_features(**inputs)
        return features
    
    def forward(self, x, edge_index):
        """
        Forward pass for standard node classification
        Args:
            x: Node features (CLIP image embeddings)
            edge_index: Graph connectivity
        """
        # Process features through GNN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def zero_shot_classify(self, x, edge_index, class_descriptions):
        """
        Zero-shot classification using natural language, following CLIP's approach
        Args:
            x: Node features (CLIP image embeddings)
            edge_index: Graph connectivity
            class_descriptions: List of text descriptions for each class
        """
        # Process node features with GNN
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        # Get text features for class descriptions
        text_features = self.extract_text_features(class_descriptions)
        
        # Optional projection layer to match dimensions if needed
        if h.shape[1] != text_features.shape[1]:
            projection = nn.Linear(h.shape[1], text_features.shape[1]).to(h.device)
            h = projection(h)
        
        # Normalize features (cosine similarity)
        h = h / h.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Apply temperature scaling like CLIP does
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(h, text_features.t())
        
        return logits
