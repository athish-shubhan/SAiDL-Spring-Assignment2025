import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import random
from typing import Dict, List, Any, Optional, Tuple, Set
import matplotlib.pyplot as plt
import ssl
import time
import sys
import re
import base64
import io
import json

class AdaptiveMultimodalSampler:
    def __init__(self, initial_rate=0.05, modality_weights=None, learn_rate=True):
        """
        Initialize the adaptive multimodal sampler.
        
        Args:
            initial_rate: Initial sampling rate (0.0-1.0)
            modality_weights: Initial weights for different modalities
            learn_rate: Whether to adaptively learn sampling weights
        """
        self.sampling_rate = initial_rate
        self.modality_weights = modality_weights or {'text': 0.4, 'image': 0.4, 'spatial': 0.2}
        self.learn_rate = learn_rate
        self.node_importance = defaultdict(float)
        self.history = []
        self.sampled_nodes_history = []
        
        # GRAPES-specific parameters (Graph Representation learning with Adaptive Propagation)
        self.alpha = 0.15  # Restart probability
        self.eps = 1e-5    # Convergence threshold
        
        # EMO-GCN parameters
        self.info_gain_threshold = 0.05
        
        # Track image vs text influence
        self.modality_performance = {'text': 0.5, 'image': 0.5, 'spatial': 0.5}
        
    def sample_entities(self, entities, modality_map, connections=None, features=None):
        """
        Adaptively sample entities based on multimodal importance.
        
        Args:
            entities: List of all entities
            modality_map: Map of entities to their available modalities
            connections: Graph structure information (optional)
            features: Existing features (optional)
            
        Returns:
            Tuple of (sampled entities, sampled indices)
        """
        if len(self.history) > 5 and self.learn_rate:
            # Adjust sampling based on previous iterations
            self._adjust_sampling_weights()
        
        # Calculate importance scores
        print("Calculating entity importance...")

        scores = self._calculate_entity_importance(entities, modality_map, connections, features)
        print(f"Calculated scores for {len(scores)} entities")

        # Sample based on importance scores using GRAPES approach
        num_to_sample = max(int(len(entities) * self.sampling_rate), 100)
        print("Sampling based on importance...")

        sampled_indices = self._sample_by_importance(scores, num_to_sample, connections)
        print(f"Sampling complete: selected {len(sampled_indices)} entities")

        # Track sampled nodes
        self.sampled_nodes_history.append(set(sampled_indices))
        
        return [entities[i] for i in sampled_indices], sampled_indices
    
    def _calculate_entity_importance(self, entities, modality_map, connections=None, features=None):
        """
        Calculate importance scores for each entity based on multimodal information.
        Implements insights from EMO-GCN to prioritize info-rich nodes.
        
        Args:
            entities: List of all entities
            modality_map: Map of entities to their available modalities
            connections: Graph connections (optional)
            features: Existing features (optional)
        
        Returns:
            List of importance scores
        """
        scores = []
        
        # Track modality statistics
        modality_counts = {'text': 0, 'image': 0, 'spatial': 0, 'none': 0}
        
        # Get neighborhood information if connections available
        neighborhood_richness = defaultdict(float)
        if connections is not None:
            for s, _, o in connections:
                s_val = s.item() if hasattr(s, 'item') else s
                o_val = o.item() if hasattr(o, 'item') else o
                neighborhood_richness[s_val] += 1
                neighborhood_richness[o_val] += 1
        
        # Calculate per-entity scores
        for i, entity in enumerate(entities):
            # Get modality information
            has_text = modality_map.get(i, {}).get('text', False)
            has_image = modality_map.get(i, {}).get('image', False)
            has_spatial = modality_map.get(i, {}).get('spatial', False)
            
            # Update counts
            if has_text: modality_counts['text'] += 1
            if has_image: modality_counts['image'] += 1
            if has_spatial: modality_counts['spatial'] += 1
            if not (has_text or has_image or has_spatial): modality_counts['none'] += 1
            
            # Calculate base score from modality presence, weighted by importance
            score = (has_text * self.modality_weights['text'] + 
                     has_image * self.modality_weights['image'] + 
                     has_spatial * self.modality_weights['spatial'])
            
            # Add node importance from history (memory effect)
            score += self.node_importance.get(i, 0) * 0.3
            
            # Add neighborhood richness (from GRAPES)
            score += min(1.0, neighborhood_richness.get(i, 0) / 10) * 0.2
            
            # If features are available, use feature variance to estimate information content (EMO-GCN)
            if features is not None and i < len(features):
                feat = features[i]
                if isinstance(feat, torch.Tensor) and feat.numel() > 0:
                    # Use feature variance as a proxy for information content
                    var = torch.var(feat).item()
                    score += min(0.5, var * 10)  # Cap influence
            
            # Prioritize nodes that weren't sampled recently (exploration)
            if len(self.sampled_nodes_history) > 0:
                last_sampled = any(i in nodes for nodes in self.sampled_nodes_history[-min(3, len(self.sampled_nodes_history)):])
                if not last_sampled:
                    score += 0.2  # Boost for exploration
            
            scores.append(score)
        
        # Print modality statistics
        print(f"Modality stats: Text={modality_counts['text']}, Image={modality_counts['image']}, Spatial={modality_counts['spatial']}, None={modality_counts['none']}")
        
        return scores
    
    def _sample_by_importance(self, scores, num_samples, connections=None):
        """
        Implementation of GRAPES sampling approach with adaptations for multimodal data.
        Uses both importance-based sampling and graph structure preservation.
        
        Args:
            scores: Importance scores for each entity
            num_samples: Number of entities to sample
            connections: Graph connections (optional)
            
        Returns:
            List of sampled indices
        """
        # Ensure minimum probability for exploration (avoid zero probabilities)
        min_prob = 0.01
        max_prob = 0.95
        
        # Normalize scores to probabilities with temperature parameter from GRAPES
        temperature = 0.3
        if sum(scores) > 0:
            scores_exp = [np.exp(s / temperature) for s in scores]
            sum_exp = sum(scores_exp)
            probs = [min_prob + (max_prob - min_prob) * (s / sum_exp) for s in scores_exp]
        else:
            probs = [1.0/len(scores)] * len(scores)

        phase_start_time = time.time()
        
        # Weighted sampling without replacement
        sampled_indices = []
        remaining_indices = list(range(len(scores)))
        
        # First phase: Sample a core set based on importance
        core_size = int(num_samples * 0.6)  # 60% of samples based on importance
        
        while len(sampled_indices) < core_size and remaining_indices:
            if len(sampled_indices) % max(1, min(100, int(core_size * 0.01))) == 0:
                percentage = (len(sampled_indices) / core_size) * 100
                elapsed = time.time() - phase_start_time
                eta = (elapsed / max(len(sampled_indices), 1)) * (core_size - len(sampled_indices)) if len(sampled_indices) > 0 else 0
                print(f"\rPhase 1: {percentage:.2f}% ({len(sampled_indices)}/{core_size}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="", flush=True)
      
            remaining_probs = [probs[i] for i in remaining_indices]
            prob_sum = sum(remaining_probs)
            if prob_sum <= 0:
                # Fall back to random sampling if all probabilities are zero
                idx = random.choice(remaining_indices)
            else:
                normalized_probs = [p/prob_sum for p in remaining_probs]
                idx = random.choices(remaining_indices, weights=normalized_probs, k=1)[0]
            
            sampled_indices.append(idx)
            remaining_indices.remove(idx)
        print(f"\nPhase 1 completed in {time.time() - phase_start_time:.2f}s")
        phase_start_time = time.time()
        print("Phase 2: Adding connected nodes for graph connectivity...")
        
        # Second phase: Add connected nodes to ensure graph connectivity (GRAPES approach)
        if connections is not None:
            # Convert connections to an adjacency list
            neighbors = defaultdict(list)
            for s, _, o in connections:
                s_val = s.item() if hasattr(s, 'item') else s
                o_val = o.item() if hasattr(o, 'item') else o
                if s_val < len(scores) and o_val < len(scores):
                    neighbors[s_val].append(o_val)
                    neighbors[o_val].append(s_val)
            
            # Add neighbors of sampled nodes based on connection structure
            connectivity_samples = set()
            for idx in sampled_indices:
                for neighbor in neighbors.get(idx, []):
                    if neighbor in remaining_indices and len(connectivity_samples) < (num_samples - core_size):
                        connectivity_samples.add(neighbor)
            
            # Add connectivity samples
            for idx in connectivity_samples:
                if idx in remaining_indices:
                    sampled_indices.append(idx)
                    remaining_indices.remove(idx)
        
        # Third phase: Fill remaining slots randomly if needed
        remaining_slots = num_samples - len(sampled_indices)
        if remaining_slots > 0 and remaining_indices:
            random_samples = random.sample(remaining_indices, min(remaining_slots, len(remaining_indices)))
            sampled_indices.extend(random_samples)
        
        self.history.append((sampled_indices, scores))
        print(f"Sampling complete: selected {len(sampled_indices)} entities")
        return sampled_indices
    
    def _adjust_sampling_weights(self):
        """
        Update modality weights based on historical performance.
        Implements adaptive weight adjustment from EMO-GCN.
        """
        if len(self.history) < 2:
            return
        
        # Analyze which modalities contribute most to chosen nodes
        chosen_importance = defaultdict(list)
        
        # Extract performance metrics from history (simplified)
        # In a real implementation, this would incorporate model performance metrics
        
        # Simple version: increase the weight of modalities that appear more in selected nodes
        total_adjustment = 0.1  # Total adjustment budget
        weights_adjustment = {'text': 0, 'image': 0, 'spatial': 0}
        
        # Apply adjustment based on perceived performance
        for modality in self.modality_weights:
            if self.modality_performance[modality] > 0.5:
                # Increase weight for high-performing modalities
                weights_adjustment[modality] = total_adjustment * (self.modality_performance[modality] - 0.5) * 2
            else:
                # Decrease weight for low-performing modalities
                weights_adjustment[modality] = -total_adjustment * (0.5 - self.modality_performance[modality]) * 2
        
        # Apply adjustments
        for modality in self.modality_weights:
            self.modality_weights[modality] += weights_adjustment[modality]
        
        # Normalize weights to ensure they sum to 1
        total = sum(self.modality_weights.values())
        self.modality_weights = {k: v/total for k, v in self.modality_weights.items()}
        
        # Log the adjustment
        print(f"Adjusted modality weights: {self.modality_weights}")


class SimpleFeatureExtractor:
    def __init__(self, root_dir: str, use_tfidf: bool = True,
                 edge_attr: bool = True, augment_text: bool = False,
                 sample_rate: float = 0.05):
        """
        Extract simple features from graph nodes.
        
        Args:
            root_dir: Dataset root directory.
            use_tfidf: Whether to use TF-IDF for text.
            edge_attr: Whether to extract edge attributes.
            augment_text: Whether to augment text data.
            sample_rate: Fraction of nodes to process (0.0-1.0).
        """
        self.start_time = time.time()
        self.phase_start_time = time.time()
        
        # Initialize the sampler
        self.sampler = AdaptiveMultimodalSampler(initial_rate=sample_rate)
        
        # Disable SSL verification for development
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Try loading with KGBench
        try:
            import kgbench as kg
            self.kg_data = kg.load('dmg777k', torch=True)
            self.using_kgbench = True
            print(f"Loaded dataset using KGBench - Sample rate: {self.sampler.sampling_rate}")
            self.print_dataset_stats()
        except Exception as e:
            print(f"Error loading with KGBench: {e}")
            self.using_kgbench = False
            print("Using direct file loading instead")
            
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Directory {root_dir} does not exist")
            self.root_dir = root_dir
            
            # load entities
            entities_file = os.path.join(root_dir, 'entities.json')
            if not os.path.exists(entities_file):
                raise FileNotFoundError(f"Entities file {entities_file} not found")
            with open(entities_file) as f:
                self.entities = json.load(f)
                
            # load triples (edges)
            triples_file = os.path.join(root_dir, 'triples.txt')
            if not os.path.exists(triples_file):
                raise FileNotFoundError(f"Triples file {triples_file} not found")
            self.triples = pd.read_csv(triples_file, sep='\t',
                                     names=['subject', 'predicate', 'object'])
                
            # create entity mappings from loaded entities
            self.e2i = {str(e['id']): i for i, e in enumerate(self.entities)}
            self.i2e = {i: str(e['id']) for i, e in enumerate(self.entities)}
        
        # Set device
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Image transformation pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained ResNet
        print("Loading ResNet model...")
        try:
            self.cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove the final fully-connected layer
            self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.children())[:-1])
            self.cnn_model.eval().to(self.device)
        except Exception as e:
            print(f"Error loading ResNet model: {e}")
            print("Using random features instead")
            self.cnn_model = None
            
        # Text vectorizer settings
        self.use_tfidf = use_tfidf
        self.edge_attr = edge_attr
        self.augment_text = augment_text
        print("Initializing text vectorizer...")
        self.text_vectorizer = TfidfVectorizer(max_features=1000) if self.use_tfidf else CountVectorizer(max_features=1000)
        
        # Fit on domain-specific initial text for DMG777k
        init_texts = [
            "monument historical building architecture heritage cultural landmark structure",
            "dutch nederland netherlands gebouw building huis house amsterdam rotterdam",
            "construction built design architect style century historical cultural heritage",
            "description title name location address street city province region",
            "castle church cathedral museum palace house hall tower bridge fort"
        ]
        self.text_vectorizer.fit(init_texts)
        
        if not self.using_kgbench:
            self.predicate_to_idx = {pred: i for i, pred in enumerate(self.triples['predicate'].unique())}
        else:
            self.predicate_to_idx = {i: i for i in range(self.kg_data.num_relations)}
            
        if self.augment_text:
            try:
                from textaugment import EDA
                self.text_augmenter = EDA()
                print("Text augmentation enabled")
            except ImportError:
                print("Warning: textaugment package not found. Text augmentation disabled.")
                self.augment_text = False
                
        os.makedirs("feature_extraction_results", exist_ok=True)
        os.makedirs("extracted_features", exist_ok=True)
        
        # Modality mapping for the sampler
        self.modality_map = {}
    
    def print_dataset_stats(self):
        """Print basic statistics about the dataset"""
        if self.using_kgbench:
            print("\n=== DMG777K Dataset Statistics ===")
            print(f"Entities: {self.kg_data.num_entities}")
            print(f"Relations: {self.kg_data.num_relations}")
            print(f"Triples: {len(self.kg_data.triples)}")
            print(f"Classes: {self.kg_data.num_classes}")
            print(f"Training Examples: {len(self.kg_data.training)}")
            print(f"Validation Examples: {len(self.kg_data.withheld)}")
            
            # Count literal types
            text_count = 0
            image_count = 0
            spatial_count = 0
            
            if hasattr(self.kg_data, 'datatype_g2l'):
                try:
                    text_indices = self.kg_data.datatype_g2l('http://www.w3.org/2001/XMLSchema#string')
                    text_count = len(text_indices)
                except:
                    pass
                
                try:
                    image_indices = self.kg_data.datatype_g2l('http://kgbench.info/dt#base64Image')
                    image_count = len(image_indices)
                except:
                    pass
                    
                try:
                    spatial_indices = self.kg_data.datatype_g2l('http://www.opengis.net/ont/geosparql#wktLiteral')
                    spatial_count = len(spatial_indices)
                except:
                    pass
                    
            print(f"Text Literals: {text_count}")
            print(f"Image Literals: {image_count}")
            print(f"Spatial Literals: {spatial_count}")
            print("================================\n")
    
    def print_progress(self, current, total, phase="Processing", every=0.0001):
        """Print progress updates with ETA"""
        if current % max(1, int(total * every)) == 0 or current == total:
            percentage = (current / total) * 100
            elapsed = time.time() - self.phase_start_time
            eta = (elapsed / max(current, 1)) * (total - current) if current > 0 else 0
            
            sys.stdout.write(f"\r{phase}: {percentage:.2f}% ({current}/{total}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            sys.stdout.flush()
            
            if current % max(1, int(total * 0.01)) == 0:
                sys.stdout.write("\n")
                
            if abs(percentage - 25) < 0.5 or abs(percentage - 50) < 0.5 or abs(percentage - 75) < 0.5:
                print(f"\n--- {percentage:.0f}% MILESTONE REACHED ---")
    
    def _map_entity_modalities(self):
        """Create a mapping of entities to their available modalities"""
        if not self.using_kgbench:
            return
            
        print("\n=== Mapping Entity Modalities ===")
        self.phase_start_time = time.time()
        
        # Get the indices for different literal types
        text_indices_set = set()
        image_indices_set = set()
        spatial_indices_set = set()
        
        if hasattr(self.kg_data, 'datatype_g2l'):
            try:
                text_indices = self.kg_data.datatype_g2l('http://www.w3.org/2001/XMLSchema#string')
                text_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in text_indices])
                print(f"Found {len(text_indices_set)} text literal indices")
            except:
                pass
                
            try:
                image_indices = self.kg_data.datatype_g2l('http://kgbench.info/dt#base64Image')
                image_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in image_indices])
                print(f"Found {len(image_indices_set)} image literal indices")
            except:
                pass
                
            try:
                spatial_indices = self.kg_data.datatype_g2l('http://www.opengis.net/ont/geosparql#wktLiteral')
                spatial_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in spatial_indices])
                print(f"Found {len(spatial_indices_set)} spatial literal indices")
            except:
                pass
        
        # Predicates for different modalities
        text_predicates = [9, 10, 57, 33, 21]  # Based on previous analysis
        image_predicates = []  # Would need to identify these
        spatial_predicates = []  # Would need to identify these
        
        # Map entities to their modalities
        for s, p, o in tqdm(self.kg_data.triples, desc="Mapping entities to modalities"):
            s_val = s.item() if hasattr(s, 'item') else s
            p_val = p.item() if hasattr(p, 'item') else p
            o_val = o.item() if hasattr(o, 'item') else o
            
            if s_val not in self.modality_map:
                self.modality_map[s_val] = {'text': False, 'image': False, 'spatial': False}
                
            # Check text connection
            if p_val in text_predicates and o_val in text_indices_set:
                self.modality_map[s_val]['text'] = True
                
            # Check image connection (if we knew image predicates)
            if o_val in image_indices_set:
                self.modality_map[s_val]['image'] = True
                
            # Check spatial connection (if we knew spatial predicates)
            if o_val in spatial_indices_set:
                self.modality_map[s_val]['spatial'] = True
        
        # Count entities with each modality
        text_count = sum(1 for v in self.modality_map.values() if v['text'])
        image_count = sum(1 for v in self.modality_map.values() if v['image'])
        spatial_count = sum(1 for v in self.modality_map.values() if v['spatial'])
        
        print(f"Entities with text: {text_count} ({text_count/len(self.modality_map)*100:.2f}%)")
        print(f"Entities with images: {image_count} ({image_count/len(self.modality_map)*100:.2f}%)")
        print(f"Entities with spatial: {spatial_count} ({spatial_count/len(self.modality_map)*100:.2f}%)")
        print(f"Modality mapping completed in {time.time()-self.phase_start_time:.2f}s")
    
    def extract_text_features_kgbench(self) -> np.ndarray:
        """Extract text features using known text predicates"""
        print("\n=== Text Extraction for DMG777K using Predicate Mapping ===")
        self.phase_start_time = time.time()
        
        num_entities = self.kg_data.num_entities
        
        # Get the text literal indices
        text_indices = self.kg_data.datatype_g2l('http://www.w3.org/2001/XMLSchema#string')
        text_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in text_indices])
        print(f"Found {len(text_indices_set)} text literal indices")
        
        # Use the text predicates identified in debugging
        text_predicates = [9, 10, 57, 33, 21] # From debug output
        print(f"Using predicates {text_predicates} to connect entities to text")
        
        # Map entities to text literals
        entity_text_map = {}
        for s, p, o in tqdm(self.kg_data.triples, desc="Mapping entities to text"):
            s_val = s.item() if hasattr(s, 'item') else s
            p_val = p.item() if hasattr(p, 'item') else p
            o_val = o.item() if hasattr(o, 'item') else o
            
            if p_val in text_predicates and o_val in text_indices_set:
                entity_text_map[s_val] = o_val
                
        print(f"Found {len(entity_text_map)} entities mapped to text literals")
        
        # Categorize entities by connection to text
        connected_entity_count = len(entity_text_map)
        connected_percentage = (connected_entity_count / num_entities) * 100
        print(f"Entities with text: {connected_entity_count} ({connected_percentage:.2f}%)")
        
        # Create text features - using custom text for different entity types
        texts = []
        monument_text = "dutch monument historical building heritage architecture cultural landmark"
        
        for i in range(num_entities):
            if i in entity_text_map:
                # We found a text literal for this entity
                # For now, we use a more specific text for these entities
                text = monument_text + " with description"
            else:
                text = monument_text
                
            texts.append(text)
            
        # Vectorize text features
        print("Vectorizing text features...")
        features = self.text_vectorizer.transform(texts).toarray()
        
        print(f"Text extraction completed in {time.time()-self.phase_start_time:.2f}s")
        return features
    
    def extract_image_features_kgbench(self, entity_indices) -> List[np.ndarray]:
        """
        Extract image features for specified entities.
        
        Args:
            entity_indices: List of entity indices to process
            
        Returns:
            List of feature arrays
        """
        print("\n=== Image Feature Extraction ===")
        self.phase_start_time = time.time()
        
        # Find image nodes
        image_features = {}
        processed_count = 0
        
        if hasattr(self.kg_data, 'datatype_g2l'):
            image_type = 'http://kgbench.info/dt#base64Image'
            try:
                img_indices = self.kg_data.datatype_g2l(image_type)
                img_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in img_indices])
                print(f"Found {len(img_indices_set)} image nodes")
            except Exception as e:
                print(f"Error getting image indices: {e}")
                img_indices_set = set()
        else:
            img_indices_set = set()
            
        # Process image nodes that are in our sampled entities
        total_to_process = len(entity_indices)
        
        for i, entity_idx in enumerate(entity_indices):
            if i % 100 == 0:
                self.print_progress(i, total_to_process, "Image feature extraction")
                
            if entity_idx in img_indices_set:
                # This is an image node - extract features
                try:
                    # Get the actual image if available
                    if hasattr(self.kg_data, 'get_images'):
                        # Get local index in image list
                        local_idx = list(img_indices_set).index(entity_idx)
                        # Get the images
                        images = self.kg_data.get_images()
                        if local_idx < len(images):
                            # Process image
                            img = images[local_idx]
                            img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                features = self.cnn_model(img_tensor)
                                image_features[entity_idx] = features.cpu().squeeze().numpy()
                                processed_count += 1
                                continue
                except Exception as e:
                    # Fall back to random features on error
                    pass
                    
                # Random features for image nodes we couldn't process
                np.random.seed(entity_idx)  # For consistency
                image_features[entity_idx] = np.random.normal(0, 0.01, 512)
            else:
                # Non-image nodes get zeros
                image_features[entity_idx] = np.zeros(512)
                
        print(f"\nProcessed {processed_count} actual images out of {len(img_indices_set)} image nodes")
        print(f"Image feature extraction completed in {time.time()-self.phase_start_time:.2f}s")
        
        # Return features in the order of entity_indices
        return [image_features[idx] for idx in entity_indices]
    
    def extract_image_features(self, entity: Dict[str, Any]) -> np.ndarray:
        """Extract image features using ResNet from entity"""
        if 'image' in entity and entity['image']:
            img_path = os.path.join(self.root_dir, 'images', entity['image'])
            if os.path.exists(img_path):
                try:
                    # Load and transform image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    if self.cnn_model is not None:
                        with torch.no_grad():
                            features = self.cnn_model(img_tensor)
                        return features.cpu().squeeze().numpy()
                except Exception as e:
                    print(f"Error processing image: {img_path}, error: {e}")
                    return np.random.normal(0, 0.01, 512)
            else:
                return np.random.normal(0, 0.01, 512)
        return np.zeros(512)
    
    def extract_text_features(self, entity: Dict[str, Any]) -> np.ndarray:
        """Extract text features using vectorizer"""
        if 'description' in entity and entity['description']:
            text = entity['description']
            
            # Augment text if enabled
            if self.augment_text and hasattr(self, 'text_augmenter') and random.random() < 0.5:
                text = self.text_augmenter.random_swap(text)
                
            try:
                # Vectorize text
                features = self.text_vectorizer.transform([text]).toarray()
                return features.squeeze()
            except Exception as e:
                print(f"Error vectorizing text, error: {e}")
                return np.zeros(self.text_vectorizer.max_features)
                
        return np.zeros(self.text_vectorizer.max_features)
    def extract_image_features_kgbench_fast(self, entity_indices) -> List[np.ndarray]:
        """
        Extract image features using batched processing and caching.
        
        Args:
            entity_indices: List of entity indices to process
        Returns:
            List of feature arrays
        """
        print("\n=== Fast Image Feature Extraction ===")
        self.phase_start_time = time.time()
        
        # Create a cache directory if it doesn't exist
        cache_dir = "image_feature_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "image_features.pt")
        
        # Try to load from cache first
        cached_features = {}
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            try:
                cached_features = torch.load(cache_file)
                print(f"Loaded {len(cached_features)} cached features")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Find image nodes
        image_features = {}
        processed_count = 0
        batch_size = 32  # Process images in batches of 16
        
        if hasattr(self.kg_data, 'datatype_g2l'):
            image_type = 'http://kgbench.info/dt#base64Image'
            try:
                img_indices = self.kg_data.datatype_g2l(image_type)
                img_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in img_indices])
                print(f"Found {len(img_indices_set)} image nodes")
            except Exception as e:
                print(f"Error getting image indices: {e}")
                img_indices_set = set()
        else:
            img_indices_set = set()
        
        # Max images to process - limit for speed
        max_images_to_process = min(5000, len(entity_indices))
        
        # First, fill in cached values
        for idx in entity_indices:
            if idx in cached_features:
                image_features[idx] = cached_features[idx]
                processed_count += 1
        
        # Create a list of remaining entities that need processing
        entities_to_process = [idx for idx in entity_indices[:max_images_to_process] 
                              if idx in img_indices_set and idx not in cached_features]
        
        # Process remaining images in batches
        total_to_process = len(entities_to_process)
        print(f"Processing {total_to_process} images in batches of {batch_size}")
        
        for i in range(0, total_to_process, batch_size):
            # Get batch of indices
            batch_indices = entities_to_process[i:i+batch_size]
            
            # Progress report
            self.print_progress(i, total_to_process, "Image batch processing")
            
            # Collect batch of images
            batch_tensors = []
            valid_indices = []
            
            if hasattr(self.kg_data, 'get_images'):
                images = self.kg_data.get_images()
                
                for entity_idx in batch_indices:
                    try:
                        # Get local index in image list
                        local_idx = list(img_indices_set).index(entity_idx)
                        
                        if local_idx < len(images):
                            # Process image
                            img = images[local_idx]
                            img_tensor = self.image_transform(img).unsqueeze(0)
                            batch_tensors.append(img_tensor)
                            valid_indices.append(entity_idx)
                    except Exception as e:
                        # Skip problematic images
                        continue
                
                # Process batch if not empty
                if batch_tensors:
                    try:
                        # Stack tensors into a single batch
                        batch_input = torch.cat(batch_tensors, dim=0).to(self.device)
                        
                        # Process batch with model
                        with torch.no_grad():
                            batch_features = self.cnn_model(batch_input)
                        
                        # Save features for each entity
                        for j, entity_idx in enumerate(valid_indices):
                            features = batch_features[j].cpu().numpy()
                            image_features[entity_idx] = features
                            cached_features[entity_idx] = features
                            processed_count += 1
                    except Exception as e:
                        print(f"Error processing batch: {e}")
            
            # Save cache every 10 batches
            if i % (batch_size * 10) == 0 and i > 0:
                print(f"Saving {len(cached_features)} features to cache")
                torch.save(cached_features, cache_file)
        
        # Final cache save
        print(f"Saving {len(cached_features)} features to cache")
        torch.save(cached_features, cache_file)
        
        # Fill in features for unprocessed entities
        for idx in entity_indices:
            if idx not in image_features:
                if idx in img_indices_set:
                    # Random features for image nodes we couldn't process
                    np.random.seed(idx)  # For consistency
                    image_features[idx] = np.random.normal(0, 0.01, 512)
                else:
                    # Non-image nodes get zeros
                    image_features[idx] = np.zeros(512)
        
        print(f"\nProcessed {processed_count} actual images")
        print(f"Image feature extraction completed in {time.time()-self.phase_start_time:.2f}s")
        
        # Return features in the order of entity_indices
        return [image_features[idx] for idx in entity_indices]

    
    def extract_features(self) -> Dict[str, torch.Tensor]:
        """Extract features for all entities and build the graph structure"""
        # Check if features are already extracted
        features_path = "extracted_features/simple_features.pt"
        if os.path.exists(features_path):
            print(f"Loading simple features from {features_path}")
            return torch.load(features_path)
            
        # Initialize feature collections
        image_features = []
        text_features = []
        labels = []
        edge_index = []
        edge_attributes = []
        
        if self.using_kgbench:
            print(f"Extracting features for KGBench dataset with {self.kg_data.num_entities} entities")
            overall_start_time = self.start_time = time.time()
            
            # Step 1: Map entities to their modalities
            self._map_entity_modalities()
            
            # Step 2: Extract all text features using the enhanced approach
            print("\nStep 1: Extracting text features")
            all_text_features = self.extract_text_features_kgbench()
            text_time = time.time() - self.start_time
            print(f"\nText feature extraction completed in {text_time:.2f}s")
            
            # Step 3: Sample entities using the adaptive sampler
            print("\nStep 2: Sampling entities using adaptive strategy")
            self.phase_start_time = time.time()
            entities = list(range(self.kg_data.num_entities))
            
            # Use the sampler to select which entities to process in detail
            sampled_entities, sampled_indices = self.sampler.sample_entities(
                entities, 
                self.modality_map,
                self.kg_data.triples
            )
            
            print(f"Sampled {len(sampled_entities)} entities out of {self.kg_data.num_entities} ({len(sampled_entities)/self.kg_data.num_entities*100:.2f}%)")
            sampling_time = time.time() - self.phase_start_time
            
            # Step 4: Extract image features for the sampled entities
            print("\nStep 3: Extracting image features for sampled entities")
            sampled_image_features = self.extract_image_features_kgbench_fast(sampled_indices)
            image_time = time.time() - self.phase_start_time - sampling_time
            
            # Step 5: Process all entities (adding features and labels)
            print("\nStep 4: Building complete feature set")
            self.phase_start_time = time.time()

            # Initialize features for all entities
            all_image_features = [np.zeros(512) for _ in range(self.kg_data.num_entities)]

            # Fill in sampled image features with progress tracking
            print("Phase 1: Filling in sampled features...")
            for i, idx in enumerate(sampled_indices):
                all_image_features[idx] = sampled_image_features[i]
                # Progress reporting for phase 1
                if i % max(1, len(sampled_indices) // 20) == 0 or i == len(sampled_indices) - 1:
                    percentage = (i + 1) / len(sampled_indices) * 100
                    elapsed = time.time() - self.phase_start_time
                    eta = elapsed / (i + 1) * (len(sampled_indices) - (i + 1)) if i > 0 else 0
                    print(f"\rPhase 1: {percentage:.2f}% ({i + 1}/{len(sampled_indices)}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="", flush=True)

            print("\nPhase 2: Processing entity labels...")
            # Step 1: Create label dictionaries (at the beginning of Phase 2)
            print("Creating label dictionaries...")
            training_labels = {}
            withheld_labels = {}

            # Create maps once - O(m+v) operation
            for i in range(len(self.kg_data.training)):
                entity_id = self.kg_data.training[i, 0].item()
                label = self.kg_data.training[i, 1].item()
                training_labels[entity_id] = label

            for i in range(len(self.kg_data.withheld)):
                entity_id = self.kg_data.withheld[i, 0].item()
                label = self.kg_data.withheld[i, 1].item()
                withheld_labels[entity_id] = label

            # Step 2: Preallocate arrays
            print("Preallocating arrays...")
            num_entities = self.kg_data.num_entities
            final_image_features = [None] * num_entities 
            final_text_features = [None] * num_entities
            final_labels = [-1] * num_entities  # Default to -1

            # Step 3: Process in batches
            batch_size = 10000
            phase2_start = time.time()

            for i in range(0, num_entities, batch_size):
                end_idx = min(i + batch_size, num_entities)
                
                # Process batch
                for idx in range(i, end_idx):
                    # Fast dictionary lookup instead of nested loops
                    if idx in training_labels:
                        final_labels[idx] = training_labels[idx]
                    elif idx in withheld_labels:
                        final_labels[idx] = withheld_labels[idx]
                    
                    # Set features
                    final_image_features[idx] = all_image_features[idx]
                    final_text_features[idx] = all_text_features[idx]
                
                # Report progress only after each batch
                percentage = end_idx / num_entities * 100
                elapsed = time.time() - phase2_start
                eta = elapsed / end_idx * (num_entities - end_idx) if end_idx > 0 else 0
                print(f"\rPhase 2: {percentage:.2f}% ({end_idx}/{num_entities}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="", flush=True)

            # Convert to final lists
            image_features = final_image_features
            text_features = final_text_features
            labels = final_labels
            entity_time = time.time() - phase2_start

            # Step 6: Extract graph structure from triples with sampling
            print("\nStep 5: Building graph structure")
            self.phase_start_time = time.time()

            # Sample edges, but keep at least 50%
            edge_sample_rate = max(0.5, self.sampler.sampling_rate)
            if edge_sample_rate < 1.0:
                # Sample a subset of triples
                num_triples = len(self.kg_data.triples)
                num_to_sample = int(num_triples * edge_sample_rate)
                print(f"Sampling {num_to_sample} triples out of {num_triples} ({edge_sample_rate:.1%})")
                triple_indices = random.sample(range(num_triples), num_to_sample)
                triples_to_process = [self.kg_data.triples[i] for i in triple_indices]
            else:
                triples_to_process = self.kg_data.triples

            # Process triples with enhanced progress tracking
            for i, (s, p, o) in enumerate(triples_to_process):
                s_val = s.item() if hasattr(s, 'item') else s
                o_val = o.item() if hasattr(o, 'item') else o
                p_val = p.item() if hasattr(p, 'item') else p
                
                edge_index.append([s_val, o_val])
                if self.edge_attr:
                    edge_attributes.append(p_val)
                
                # Enhanced progress reporting
                if i % max(1, len(triples_to_process) // 100) == 0 or i == len(triples_to_process) - 1:
                    percentage = (i + 1) / len(triples_to_process) * 100
                    elapsed = time.time() - self.phase_start_time
                    eta = elapsed / (i + 1) * (len(triples_to_process) - (i + 1)) if i > 0 else 0
                    print(f"\rGraph building: {percentage:.2f}% ({i + 1}/{len(triples_to_process)}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="", flush=True)

            total_time = time.time() - self.phase_start_time
            print(f"\nGraph building completed in {total_time:.2f}s")
            edge_time = total_time


        else:
            # Original code for non-KGBench datasets
            print(f"Extracting features for {len(self.entities)} entities")
            
            # Extract node features
            for i, entity in enumerate(tqdm(self.entities, desc="Extracting features")):
                img_feat = self.extract_image_features(entity)
                text_feat = self.extract_text_features(entity)
                
                image_features.append(img_feat)
                text_features.append(text_feat)
                labels.append(entity.get('label', -1))
                
                # Show progress
                self.print_progress(i + 1, len(self.entities), "Entity processing")
                
            # Build graph edges
            for i, (_, row) in enumerate(tqdm(self.triples.iterrows(), desc="Building graph", total=len(self.triples))):
                if str(row['subject']) in self.e2i and str(row['object']) in self.e2i:
                    edge_index.append([self.e2i[str(row['subject'])], self.e2i[str(row['object'])]])
                    
                    if self.edge_attr:
                        edge_attributes.append(self.predicate_to_idx[row['predicate']])
                        
                # Show progress
                self.print_progress(i + 1, len(self.triples), "Graph building")
                
        print("\nConverting features to tensors")
        
        # Convert to tensors
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        if self.edge_attr and edge_attributes:
            edge_attributes_tensor = torch.tensor(edge_attributes, dtype=torch.float32)
            # Normalize edge attributes
            edge_attributes_tensor = (edge_attributes_tensor - edge_attributes_tensor.mean()) / \
                                     (torch.norm(edge_attributes_tensor) + 1e-8)
        else:
            edge_attributes_tensor = None

        normalized_image_features = []
        for feat in image_features:
            # Convert to numpy if it's a tensor
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            
            # Flatten any multi-dimensional arrays to 1D
            if feat.ndim > 1:
                feat = feat.reshape(512)
            
            normalized_image_features.append(feat)
        # Convert features to tensors
        image_features_tensor = torch.stack([torch.from_numpy(feat).float() for feat in normalized_image_features])
        text_features_tensor = torch.stack([torch.from_numpy(feat).float() for feat in text_features])
        
        # Save feature statistics
        self._save_feature_statistics(image_features, text_features)
        
        # Create result dictionary
        result = {
            'image_features': image_features_tensor,
            'text_features': text_features_tensor,
            'edge_index': edge_index_tensor,
            'edge_attributes': edge_attributes_tensor,
            'labels': torch.tensor(labels, dtype=torch.long),
            'indices': torch.arange(len(image_features))
        }
        
        # Save features
        torch.save(result, features_path)
        print(f"Saved features to {features_path}")
        
        if self.using_kgbench:
            total_time = time.time() - overall_start_time
            print(f"\nTotal processing time: {total_time:.2f}s")
            print(f"Text extraction: {text_time:.2f}s ({text_time/total_time*100:.1f}%)")
            print(f"Sampling: {sampling_time:.2f}s ({sampling_time/total_time*100:.1f}%)")
            print(f"Image extraction: {image_time:.2f}s ({image_time/total_time*100:.1f}%)")
            print(f"Feature building: {entity_time:.2f}s ({entity_time/total_time*100:.1f}%)")
            print(f"Graph building: {edge_time:.2f}s ({edge_time/total_time*100:.1f}%)")
            
        return result
        
    def _save_feature_statistics(self, image_features: List[np.ndarray], text_features: List[np.ndarray]) -> None:
        """Save statistics about extracted features"""
        print("Calculating feature statistics")
        
        # Calculate statistics
        img_norms = [np.linalg.norm(feat) for feat in image_features]
        txt_norms = [np.linalg.norm(feat) for feat in text_features]
        
        # Plot feature norm distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(img_norms, bins=50)
        plt.title('Simple Image Feature Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(txt_norms, bins=50)
        plt.title('Simple Text Feature Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('feature_extraction_results/simple_feature_norms.png')
        plt.close()
        
        # Save statistics to file
        with open('feature_extraction_results/simple_feature_stats.txt', 'w') as f:
            f.write(f"Total entities: {len(image_features)}\n")
            f.write(f"Image feature dimension: {image_features[0].shape[0]}\n")
            f.write(f"Text feature dimension: {text_features[0].shape[0]}\n")
            f.write(f"Avg image feature norm: {np.mean(img_norms):.4f}\n")
            f.write(f"Avg text feature norm: {np.mean(txt_norms):.4f}\n")
            f.write(f"Entities with zero image features: {sum(1 for norm in img_norms if norm < 1e-6)}\n")
            f.write(f"Entities with zero text features: {sum(1 for norm in txt_norms if norm < 1e-6)}\n")
            

if __name__ == "__main__":
    # Example usage:
    extractor = SimpleFeatureExtractor("dmg777k_dataset", sample_rate=0.05)
    features = extractor.extract_features()
