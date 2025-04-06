import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import ssl
from typing import Tuple, Dict, Any, List, Optional
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import torch.cuda.amp

class CustomImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 256):
        """
        Custom image encoder based on ResNet
        Args:
            output_dim: Dimension of output embeddings
        """
        super(CustomImageEncoder, self).__init__()
        
        # Disable SSL verification for development
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"Error loading pretrained ResNet34: {e}")
            print("Initializing ResNet with random weights")
            resnet = models.resnet34(weights=None)
            
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x)

class CustomTextEncoder(nn.Module):
    def __init__(self, input_dim=1000, output_dim=256):
        """
        Custom text encoder with dynamic input adaptation
        """
        super(CustomTextEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Add flexible input adapter for handling dimension mismatches
        self.input_adapter = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input dimensions and create adapter if needed
        if x.shape[1] != self.input_dim:
            if self.input_adapter is None or self.input_adapter.in_features != x.shape[1]:
                print(f"Creating input adapter: {x.shape[1]} → {self.input_dim}")
                self.input_adapter = nn.Linear(x.shape[1], self.input_dim).to(x.device)
            
            # Apply the adapter
            x = self.input_adapter(x)
        
        # Process through main encoder
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

class IntermediateFeatureExtractor:
    def __init__(self, simple_extractor: Any):
        """
        Intermediate feature extractor using custom encoders with optimized processing
        Args:
            simple_extractor: Base feature extractor
        """
        # Initialize timing variables
        self.start_time = time.time()
        self.phase_start_time = time.time()
        
        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        self.simple_extractor = simple_extractor
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        
        print(f"Using device: {self.device}")
        
        # Initialize encoders
        self.print_step_header("Step 1: Initializing image encoder")
        try:
            self.image_encoder = CustomImageEncoder().to(self.device)
            print(f"✓ Image encoder initialized successfully ({time.time() - self.phase_start_time:.2f}s)")
        except Exception as e:
            print(f"✗ Error initializing image encoder: {e}")
            print("  Using simpler model")
            self.image_encoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            ).to(self.device)
        
        # Initialize text encoder
        self.phase_start_time = time.time()
        self.print_step_header("Step 2: Initializing text encoder")
        
        # Determine input dimension for text features
        text_input_dim = 1000  # Default fallback
        if hasattr(simple_extractor, 'text_vectorizer'):
            text_input_dim = simple_extractor.text_vectorizer.max_features
        
        try:
            self.text_encoder = CustomTextEncoder(input_dim=text_input_dim).to(self.device)
            print(f"✓ Text encoder initialized successfully ({time.time() - self.phase_start_time:.2f}s)")
        except Exception as e:
            print(f"✗ Error initializing text encoder: {e}")
            print("  Using simpler model")
            self.text_encoder = nn.Sequential(
                nn.Linear(text_input_dim, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            ).to(self.device)
        
        # Create directories for saving results
        os.makedirs("feature_extraction_results", exist_ok=True)
        os.makedirs("extracted_features", exist_ok=True)
        os.makedirs("intermediate_cache", exist_ok=True)
    
    def print_step_header(self, step_name: str):
        """Print a formatted step header"""
        print(f"\n{'=' * 20} {step_name} {'=' * 20}")
    
    def print_progress(self, current, total, phase="Processing", every=0.0001):
        """Print progress with ETA at specified intervals"""
        # Print progress updates at regular intervals
        if current % max(1, int(total * every)) == 0 or current == total:
            percentage = (current / total) * 100
            elapsed = time.time() - self.phase_start_time
            eta = (elapsed / max(current, 1)) * (total - current) if current > 0 else 0
            
            sys.stdout.write(f"\r{phase}: {percentage:.2f}% ({current}/{total}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            sys.stdout.flush()
            
            # Add new line at percentage milestones
            if current % max(1, int(total * 0.01)) == 0:
                sys.stdout.write("\n")
    
    def extract_features(self) -> Dict[str, torch.Tensor]:
        """Extract enhanced features using custom encoders with detailed progress tracking"""
        # Check if features are already extracted
        features_path = "extracted_features/intermediate_features.pt"
        if os.path.exists(features_path):
            print(f"Loading intermediate features from {features_path}")
            return torch.load(features_path)
        
        # Set up timing
        self.overall_start_time = time.time()
        
        # Step 3: Extract base features from simple extractor
        self.print_step_header("Step 3: Extracting base features")
        self.phase_start_time = time.time()
        
        try:
            # Get raw features from simple extractor
            simple_features = self.simple_extractor.extract_features()
            print(f"✓ Base features extracted successfully ({time.time() - self.phase_start_time:.2f}s)")
        except Exception as e:
            print(f"✗ Error extracting simple features: {e}")
            print("  Using random features instead")
            
            # Create random features as fallback
            if hasattr(self.simple_extractor, 'kg_data'):
                n_samples = self.simple_extractor.kg_data.num_entities
            elif hasattr(self.simple_extractor, 'entities'):
                n_samples = len(self.simple_extractor.entities)
            else:
                n_samples = 10000  # Default fallback
                
            print(f"Generating random features for {n_samples} samples")
            return {
                'image_features': torch.randn(n_samples, 256),
                'text_features': torch.randn(n_samples, 256),
                'edge_index': torch.zeros(2, 1, dtype=torch.long),
                'labels': torch.zeros(n_samples, dtype=torch.long)
            }
        
        # Step 4: Process image features
        self.print_step_header("Step 4: Enhancing image features")
        self.phase_start_time = time.time()
        
        # Move tensors to device
        image_features = simple_features['image_features'].to(self.device)
        
        # Use batching for processing large datasets with caching
        batch_size = 64  # Larger batch size for faster processing
        num_samples = image_features.size(0)
        
        # Check cache for image features
        image_cache_file = "intermediate_cache/image_features.pt"
        if os.path.exists(image_cache_file):
            print(f"Loading cached image features")
            try:
                enhanced_image_features = torch.load(image_cache_file)
                if len(enhanced_image_features) == num_samples:
                    print(f"✓ Loaded {len(enhanced_image_features)} cached image features")
                else:
                    print(f"✗ Cache size mismatch. Reprocessing all features.")
                    enhanced_image_features = []
            except Exception as e:
                print(f"✗ Error loading cache: {e}")
                enhanced_image_features = []
        else:
            enhanced_image_features = []
        
        # Process image features if cache wasn't valid
        if len(enhanced_image_features) != num_samples:
            enhanced_image_features = []
            total_batches = (num_samples + batch_size - 1) // batch_size
            
            # Process in batches with detailed progress tracking
            for i in range(0, num_samples, batch_size):
                batch_indices = slice(i, min(i + batch_size, num_samples))
                batch = image_features[batch_indices]
                
                # Calculate and display progress with ETA
                current_batch = i // batch_size + 1
                percentage = (current_batch / total_batches) * 100
                elapsed = time.time() - self.phase_start_time
                eta = elapsed / max(current_batch, 1) * (total_batches - current_batch) if current_batch > 0 else 0
                
                print(f"\rImage feature enhancement: {percentage:.2f}% ({i+batch.size(0)}/{num_samples}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
                
                # Process batch with mixed precision if available
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                    with torch.no_grad():
                        enhanced = self.image_encoder(batch)
                        enhanced_image_features.append(enhanced.cpu())
                
                # Save intermediate results every 50 batches
                if i > 0 and i % (batch_size * 50) == 0:
                    intermediate = torch.cat(enhanced_image_features, dim=0)
                    torch.save(intermediate, image_cache_file)
                    print(f"\nSaved intermediate image features ({len(intermediate)}/{num_samples})")
            
            # Combine all batches
            enhanced_image_features = torch.cat(enhanced_image_features, dim=0)
            
            # Save final cache
            torch.save(enhanced_image_features, image_cache_file)
            print(f"\n✓ Image feature enhancement completed in {time.time() - self.phase_start_time:.2f}s")
        
        # Step 5: Process text features
        self.print_step_header("Step 5: Enhancing text features")
        self.phase_start_time = time.time()
        
        # Move text features to device
        text_features = simple_features['text_features'].to(self.device)
        
        # Check cache for text features
        text_cache_file = "intermediate_cache/text_features.pt"
        if os.path.exists(text_cache_file):
            print(f"Loading cached text features")
            try:
                enhanced_text_features = torch.load(text_cache_file)
                if len(enhanced_text_features) == num_samples:
                    print(f"✓ Loaded {len(enhanced_text_features)} cached text features")
                else:
                    print(f"✗ Cache size mismatch. Reprocessing all features.")
                    enhanced_text_features = []
            except Exception as e:
                print(f"✗ Error loading cache: {e}")
                enhanced_text_features = []
        else:
            enhanced_text_features = []
        
        # Process text features if cache wasn't valid
        if len(enhanced_text_features) != num_samples:
            enhanced_text_features = []
            total_batches = (num_samples + batch_size - 1) // batch_size
            
            # Process in batches with detailed progress tracking
            for i in range(0, num_samples, batch_size):
                batch_indices = slice(i, min(i + batch_size, num_samples))
                batch = text_features[batch_indices]
                
                # Calculate and display progress with ETA
                current_batch = i // batch_size + 1
                percentage = (current_batch / total_batches) * 100
                elapsed = time.time() - self.phase_start_time
                eta = elapsed / max(current_batch, 1) * (total_batches - current_batch) if current_batch > 0 else 0
                
                print(f"\rText feature enhancement: {percentage:.2f}% ({i+batch.size(0)}/{num_samples}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
                
                # Use mixed precision for faster processing if available
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                    with torch.no_grad():
                        enhanced = self.text_encoder(batch)
                        enhanced_text_features.append(enhanced.cpu())
                
                # Save intermediate results every 50 batches
                if i > 0 and i % (batch_size * 50) == 0:
                    intermediate = torch.cat(enhanced_text_features, dim=0)
                    torch.save(intermediate, text_cache_file)
                    print(f"\nSaved intermediate text features ({len(intermediate)}/{num_samples})")
            
            # Combine all batches
            enhanced_text_features = torch.cat(enhanced_text_features, dim=0)
            
            # Save final cache
            torch.save(enhanced_text_features, text_cache_file)
            print(f"\n✓ Text feature enhancement completed in {time.time() - self.phase_start_time:.2f}s")
        
        # Step 6: Create result dictionary
        self.print_step_header("Step 6: Building final feature set")
        self.phase_start_time = time.time()
        
        # Create result dictionary with all tensors
        result = {
            'image_features': enhanced_image_features,
            'text_features': enhanced_text_features,
            'edge_index': simple_features['edge_index'],
            'labels': simple_features['labels']
        }
        
        # Save features
        torch.save(result, features_path)
        
        # Save feature statistics
        self._save_feature_statistics(enhanced_image_features.numpy(), enhanced_text_features.numpy())
        
        total_time = time.time() - self.overall_start_time
        print(f"\n✓ Feature building completed in {time.time() - self.phase_start_time:.2f}s")
        print(f"\nTotal processing time: {total_time:.2f}s")
        
        return result
    
    def _save_feature_statistics(self, image_features: np.ndarray, text_features: np.ndarray) -> None:
        """Save statistics about extracted features"""
        print("Calculating feature statistics...")
        
        # Calculate statistics
        img_norms = [np.linalg.norm(feat) for feat in image_features]
        txt_norms = [np.linalg.norm(feat) for feat in text_features]
        
        # Plot feature norm distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(img_norms, bins=50)
        plt.title('Intermediate Image Feature Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(txt_norms, bins=50)
        plt.title('Intermediate Text Feature Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('feature_extraction_results/intermediate_feature_norms.png')
        plt.close()
        
        # Save statistics to file
        with open('feature_extraction_results/intermediate_feature_stats.txt', 'w') as f:
            f.write(f"Total entities: {len(image_features)}\n")
            f.write(f"Image feature dimension: {image_features[0].shape[0]}\n")
            f.write(f"Text feature dimension: {text_features[0].shape[0]}\n")
            f.write(f"Avg image feature norm: {np.mean(img_norms):.4f}\n")
            f.write(f"Avg text feature norm: {np.mean(txt_norms):.4f}\n")
            f.write(f"Entities with zero image features: {sum(1 for norm in img_norms if norm < 1e-6)}\n")
            f.write(f"Entities with zero text features: {sum(1 for norm in txt_norms if norm < 1e-6)}\n")
            
        print(f"✓ Statistics saved to feature_extraction_results/")
