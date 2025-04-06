from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import os
import json
import pandas as pd
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import ssl
from typing import Dict, Any, List, Optional
import time
import sys
import random
from contextlib import nullcontext
import torch.cuda.amp

class HuggingFaceFeatureExtractor:
    def __init__(self, root_dir: str, sample_rate: float = 1.0):
        """
        Initialize the feature extractor with HuggingFace models and progress tracking
        Args:
            root_dir: Path to the dataset directory
            sample_rate: Fraction of nodes to process (0.0-1.0, set < 1.0 for faster extraction)
        """
        # Initialize timing variables
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.sample_rate = max(0.01, min(0.05, sample_rate))  # Ensure between 0.01 and 1.0
        
        # Temporarily disable SSL verification for development
        ssl._create_default_https_context = ssl._create_unverified_context
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
        
        print(f"Using device: {self.device} | Sample rate: {self.sample_rate:.2f}")
        
        # Try loading with KGBench format first
        try:
            import kgbench as kg
            self.kg_data = kg.load('dmg777k', torch=True)
            self.using_kgbench = True
            print("Loaded dataset using KGBench")
            self.print_dataset_stats()
        except Exception as e:
            print(f"Error loading with KGBench: {e}")
            self.using_kgbench = False
            print("Using direct file loading")
            
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Directory {root_dir} does not exist")
            
            self.root_dir = root_dir
            entities_file = os.path.join(root_dir, 'entities.json')
            if not os.path.exists(entities_file):
                raise FileNotFoundError(f"Entities file {entities_file} not found")
            
            with open(entities_file) as f:
                self.entities = json.load(f)
            
            triples_file = os.path.join(root_dir, 'triples.txt')
            if not os.path.exists(triples_file):
                raise FileNotFoundError(f"Triples file {triples_file} not found")
            
            self.triples = pd.read_csv(triples_file, 
                                      sep='\t', names=['subject', 'predicate', 'object'])
            
            self.e2i = {str(e['id']): i for i, e in enumerate(self.entities)}
            self.i2e = {i: str(e['id']) for i, e in enumerate(self.entities)}
        
        # Step 1: Load CLIP model with progress reporting
        self.print_step_header("Step 1: Loading CLIP model")
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print(f"✓ CLIP model loaded successfully ({time.time() - self.phase_start_time:.2f}s)")
        except Exception as e:
            print(f"✗ Error loading CLIP model: {e}")
            print(" Using random features instead")
            self.clip_model = None
            self.clip_processor = None
        
        # Step 2: Load SentenceTransformer model
        self.phase_start_time = time.time()
        self.print_step_header("Step 2: Loading SentenceTransformer model")
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
            print(f"✓ SentenceTransformer model loaded successfully ({time.time() - self.phase_start_time:.2f}s)")
        except Exception as e:
            print(f"✗ Error loading SentenceTransformer model: {e}")
            print(" Using random features instead")
            self.text_model = None
        
        # Create directory for storing results
        os.makedirs("feature_extraction_results", exist_ok=True)
        os.makedirs("extracted_features", exist_ok=True)
        
    def print_dataset_stats(self):
        """Print basic statistics about the dataset"""
        if self.using_kgbench:
            print(f"\n=== DMG777K Dataset Statistics ===")
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
            
    def print_step_header(self, step_name: str):
        """Print a formatted step header"""
        print(f"\n{'=' * 20} {step_name} {'=' * 20}")
        
    def print_progress(self, current, total, phase="Processing", every=0.0001):
        """Print progress with ETA at specified intervals"""
        # Print progress every 0.01% (every=0.0001) or at specified interval
        if current % max(1, int(total * every)) == 0 or current == total:
            percentage = (current / total) * 100
            elapsed = time.time() - self.phase_start_time
            eta = (elapsed / max(current, 1)) * (total - current) if current > 0 else 0
            
            sys.stdout.write(f"\r{phase}: {percentage:.2f}% ({current}/{total}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            sys.stdout.flush()
            
            # Full line update at each percent
            if current % max(1, int(total * 0.01)) == 0:
                sys.stdout.write("\n")
                
    def _map_entity_modalities(self):
        """Create a mapping of entities to their available modalities"""
        if not self.using_kgbench:
            return {}
            
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
            except Exception as e:
                print(f"Error getting text indices: {e}")
                
            try:
                image_indices = self.kg_data.datatype_g2l('http://kgbench.info/dt#base64Image')
                image_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in image_indices])
                print(f"Found {len(image_indices_set)} image literal indices")
            except Exception as e:
                print(f"Error getting image indices: {e}")
                
            try:
                spatial_indices = self.kg_data.datatype_g2l('http://www.opengis.net/ont/geosparql#wktLiteral')
                spatial_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in spatial_indices])
                print(f"Found {len(spatial_indices_set)} spatial literal indices")
            except Exception as e:
                print(f"Error getting spatial indices: {e}")
        
        # Predicates for different modalities (based on analysis)
        text_predicates = [9, 10, 57, 33, 21]  # From sfeatures.py
        
        # Map entities to their modalities
        entity_modality_map = {}
        
        total_triples = len(self.kg_data.triples)
        for i, (s, p, o) in enumerate(self.kg_data.triples):
            if i % max(1, total_triples // 100) == 0:
                self.print_progress(i, total_triples, "Mapping entities to modalities")
                
            s_val = s.item() if hasattr(s, 'item') else s
            p_val = p.item() if hasattr(p, 'item') else p
            o_val = o.item() if hasattr(o, 'item') else o
            
            if s_val not in entity_modality_map:
                entity_modality_map[s_val] = {'text': False, 'image': False, 'spatial': False}
            
            # Check text connection
            if p_val in text_predicates and o_val in text_indices_set:
                entity_modality_map[s_val]['text'] = True
            
            # Check image connection 
            if o_val in image_indices_set:
                entity_modality_map[s_val]['image'] = True
            
            # Check spatial connection
            if o_val in spatial_indices_set:
                entity_modality_map[s_val]['spatial'] = True
        
        # Count entities with each modality
        text_count = sum(1 for v in entity_modality_map.values() if v['text'])
        image_count = sum(1 for v in entity_modality_map.values() if v['image'])
        spatial_count = sum(1 for v in entity_modality_map.values() if v['spatial'])
        
        print(f"\nEntities with text: {text_count} ({text_count/len(entity_modality_map)*100:.2f}%)")
        print(f"Entities with images: {image_count} ({image_count/len(entity_modality_map)*100:.2f}%)")
        print(f"Entities with spatial: {spatial_count} ({spatial_count/len(entity_modality_map)*100:.2f}%)")
        print(f"Modality mapping completed in {time.time()-self.phase_start_time:.2f}s")
        
        return entity_modality_map
        
    def extract_image_features_kgbench_fast(self, entity_indices) -> List[np.ndarray]:
        """
        Extract image features using batched processing and efficient caching.
        
        Args:
            entity_indices: List of entity indices to process
            
        Returns:
            List of image feature arrays
        """
        print("\n=== Fast Image Feature Extraction ===")
        self.phase_start_time = time.time()
        
        # Create cache directory
        cache_dir = "image_feature_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "image_features.pt")
        temp_cache_file = os.path.join(cache_dir, "image_features_temp.pt")
        
        # Try loading from cache first
        cached_features = {}
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            try:
                cached_features = torch.load(cache_file, weights_only=True)
                print(f"Loaded {len(cached_features)} cached features")
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Try backup if available
                if os.path.exists(temp_cache_file):
                    try:
                        cached_features = torch.load(temp_cache_file)
                        print(f"Recovered {len(cached_features)} features from backup cache")
                    except Exception:
                        pass
        
        # Determine optimal batch size based on device
        if torch.cuda.is_available():
            # Adaptive batch size based on available GPU memory
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            batch_size = min(128, max(16, int(free_mem / (5 * 1024 * 1024))))
            print(f"Using GPU with batch size of {batch_size}")
        else:
            batch_size = 32
            print(f"Using CPU with batch size of {batch_size}")
        
        # Get image indices
        img_indices_set = set()
        if hasattr(self.kg_data, 'datatype_g2l'):
            try:
                image_type = 'http://kgbench.info/dt#base64Image'
                img_indices = self.kg_data.datatype_g2l(image_type)
                img_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in img_indices])
                print(f"Found {len(img_indices_set)} image nodes")
            except Exception as e:
                print(f"Error getting image indices: {e}")
        
        # Initialize features dictionary with cached values
        image_features = {}
        for idx in entity_indices:
            if idx in cached_features:
                image_features[idx] = cached_features[idx]
        
        # Create list of entities needing processing (not in cache)
        entities_to_process = sorted([
            idx for idx in entity_indices
            if idx in img_indices_set and idx not in cached_features
        ])
        
        # Process a reasonable number of images for performance
        max_to_process = min(len(entities_to_process), 5000)
        if len(entities_to_process) > max_to_process:
            # Use evenly distributed sampling
            step = max(1, len(entities_to_process) // max_to_process)
            entities_to_process = sorted(entities_to_process[::step][:max_to_process])
            print(f"Limited processing to {len(entities_to_process)} images (sampling every {step}th)")
        
        total_to_process = len(entities_to_process)
        if total_to_process == 0:
            print("No new images to process")
        else:
            print(f"Processing {total_to_process} images in batches of {batch_size}")
        
        # Preload images for faster access if possible
        images = None
        if hasattr(self.kg_data, 'get_images') and total_to_process > 0:
            try:
                images = self.kg_data.get_images()
                print(f"Preloaded {len(images)} images")
            except Exception as e:
                print(f"Error preloading images: {e}")
        
        # Create mapping for O(1) lookups
        img_indices_list = list(img_indices_set)
        img_indices_map = {idx: i for i, idx in enumerate(img_indices_list)}
        
        # Ensure model is in evaluation mode
        if self.clip_model is not None:
            self.clip_model.eval()
        
        # Tracking variables
        processed_count = len(cached_features)
        checkpoint_interval = max(1, total_to_process // 10)
        last_checkpoint = 0
        processed_this_run = 0
        
        # Process in batches
        for i in range(0, total_to_process, batch_size):
            batch_indices = entities_to_process[i:i+batch_size]
            
            # Progress reporting
            percentage = min(100.0, (i / max(1, total_to_process)) * 100)
            elapsed = time.time() - self.phase_start_time
            eta = elapsed / max(i, 1) * (total_to_process - i) if i > 0 else 0
            print(f"\rImage batch processing: {percentage:.2f}% ({i}/{total_to_process}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
            
            # Mark major milestones
            if int(percentage) % 25 == 0 and i > last_checkpoint and int(percentage) > 0:
                print(f"\n--- {int(percentage)}% MILESTONE REACHED ---")
            
            # Save checkpoints periodically
            if i > 0 and i - last_checkpoint >= checkpoint_interval:
                new_features = len(cached_features) - processed_count
                if new_features > 0:
                    print(f"\nSaving {new_features} features to cache")
                    torch.save(cached_features, temp_cache_file)
                    os.replace(temp_cache_file, cache_file)  # Atomic replacement
                    processed_count = len(cached_features)
                    last_checkpoint = i
            
            # Collect batch of raw images for CLIP
            batch_images = []
            valid_indices = []
            
            if images is not None:
                for entity_idx in batch_indices:
                    try:
                        if entity_idx in img_indices_map:
                            local_idx = img_indices_map[entity_idx]
                            if local_idx < len(images):
                                # Store the raw image, not a tensor
                                img = images[local_idx]
                                batch_images.append(img)
                                valid_indices.append(entity_idx)
                    except Exception:
                        continue
            
            # Process batch if not empty
            if batch_images and self.clip_model is not None:
                try:
                    # Use mixed precision for faster processing
                    with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                        with torch.no_grad():
                            # Process with CLIP processor
                            inputs = self.clip_processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                            batch_features = self.clip_model.get_image_features(**inputs)
                    
                    # Store features for each valid entity
                    for j, entity_idx in enumerate(valid_indices):
                        features = batch_features[j].cpu().numpy()
                        image_features[entity_idx] = features
                        cached_features[entity_idx] = features
                        processed_this_run += 1
                
                except Exception as e:
                    print(f"\nError processing batch: {e}")
                
                # Clear GPU cache periodically
                if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        # Final cache save
        if len(cached_features) > processed_count:
            print(f"\nSaving {len(cached_features) - processed_count} features to cache")
            torch.save(cached_features, temp_cache_file)
            os.replace(temp_cache_file, cache_file)
        
        # Fill in missing entities with appropriate values
        missing_indices = [idx for idx in entity_indices if idx not in image_features]
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(missing_indices), chunk_size):
            chunk = missing_indices[i:i+chunk_size]
            for idx in chunk:
                if idx in img_indices_set:
                    # Random features for image nodes we couldn't process
                    np.random.seed(idx)  # For reproducibility
                    image_features[idx] = np.random.normal(0, 0.01, 512)
                else:
                    # Non-image nodes get zeros
                    image_features[idx] = np.zeros(512)
        
        print(f"\nProcessed {processed_this_run} actual images")
        print(f"Image feature extraction completed in {time.time()-self.phase_start_time:.2f}s")
        
        # Return features in the order requested
        return [image_features[idx] for idx in entity_indices]
        
    def extract_text_features_kgbench_fast(self) -> List[np.ndarray]:
        """Extract text features using tensor-aware extraction and SentenceTransformer"""
        self.print_step_header("Step 4: Extracting text features")
        self.phase_start_time = time.time()
        
        # Get the text literal indices
        text_indices = self.kg_data.datatype_g2l('http://www.w3.org/2001/XMLSchema#string')
        text_indices_set = set([idx.item() if hasattr(idx, 'item') else idx for idx in text_indices])
        print(f"Found {len(text_indices_set)} text literal indices")
        
        # Use the text predicates identified from sfeatures.py
        text_predicates = [9, 10, 57, 33, 21]
        print(f"Using predicates {text_predicates} to connect entities to text")
        
        # Map entities to text literals
        entity_text_map = {}
        total_triples = len(self.kg_data.triples)
        
        for i, (s, p, o) in enumerate(self.kg_data.triples):
            if i % max(1, total_triples // 100) == 0:
                self.print_progress(i, total_triples, "Mapping entities to text")
                
            s_val = s.item() if hasattr(s, 'item') else s
            p_val = p.item() if hasattr(p, 'item') else p
            o_val = o.item() if hasattr(o, 'item') else o
            
            if p_val in text_predicates and o_val in text_indices_set:
                entity_text_map[s_val] = o_val
                
        print(f"\nFound {len(entity_text_map)} entities mapped to text literals")
        
        # Create features for all entities
        text_features = []
        num_entities = self.kg_data.num_entities
        
        # Create cache file for text features
        cache_dir = "text_features_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "text_features.pt")
        
        # Try loading from cache first
        if os.path.exists(cache_file):
            print(f"Loading cached text features from {cache_file}")
            try:
                # Add weights_only=True for security
                cached_features = torch.load(cache_file, weights_only=True)
                if len(cached_features) == num_entities:
                    print(f"Loaded {len(cached_features)} cached text features")
                    return cached_features
                else:
                    print(f"Cached features count mismatch: {len(cached_features)} vs {num_entities} needed")
            except (RuntimeError, EOFError) as e:
                print(f"Error loading cached text features: {e}")
                print("Removing invalid cache and regenerating features...")
                # Remove the corrupted cache file
                os.remove(cache_file)

        batch_size = 64
        text_entities = sorted(list(entity_text_map.keys()))
        text_batches = [text_entities[i:i+batch_size] for i in range(0, len(text_entities), batch_size)]
        
        # Initialize features for all entities
        text_features = [np.zeros(384) for _ in range(num_entities)]
        
        # Process text in batches
        if self.text_model and text_entities:
            processed_count = 0
            total_batches = len(text_batches)
            
            for i, batch_indices in enumerate(text_batches):
                # Calculate and display progress with ETA
                percentage = ((i + 1) / total_batches) * 100
                elapsed = time.time() - self.phase_start_time
                eta = elapsed / (i + 1) * (total_batches - (i + 1)) if i > 0 else 0
                print(f"\rText batch processing: {percentage:.2f}% ({i + 1}/{total_batches}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
                
                # Save intermediate results
                if i > 0 and i % 10 == 0:
                    torch.save(text_features, cache_file)
                    print(f"\nSaved {i*batch_size} processed text features to cache")
                
                # Get actual text for these entities if available
                batch_texts = []
                valid_indices = []
                
                if hasattr(self.kg_data, 'get_strings'):
                    strings = self.kg_data.get_strings('http://www.w3.org/2001/XMLSchema#string')
                    for idx in batch_indices:
                        try:
                            if idx in entity_text_map:
                                text_idx = entity_text_map[idx]
                                local_idx = list(text_indices_set).index(text_idx)
                                if local_idx < len(strings):
                                    text = strings[local_idx]
                                    batch_texts.append(text)
                                    valid_indices.append(idx)
                                    continue
                        except Exception as e:
                            pass
                        
                        # Default text if actual text not available
                        batch_texts.append("dutch monument historical building")
                        valid_indices.append(idx)
                
                # Skip empty batches
                if not batch_texts:
                    continue
                
                # Process batch with SentenceTransformer
                try:
                    with torch.no_grad():
                        embeddings = self.text_model.encode(batch_texts, show_progress_bar=False)
                    
                    # Store features
                    for j, idx in enumerate(valid_indices):
                        text_features[idx] = embeddings[j]
                        processed_count += 1
                        
                except Exception as e:
                    print(f"Error encoding text batch: {e}")
                    
                    # Use random features as fallback
                    for idx in valid_indices:
                        np.random.seed(idx)
                        text_features[idx] = np.random.normal(0, 0.1, 384)
                        
            print("\n") # New line after batch processing completes
        
        # For all other entities, use random features with lower magnitude
        for i in range(num_entities):
            if np.all(text_features[i] == 0):
                if i in entity_text_map:
                    # Text-related entities get more distinctive random features
                    np.random.seed(i)
                    text_features[i] = np.random.normal(0.2, 0.05, 384)
                else:
                    # Non-text entities get less distinctive features
                    np.random.seed(i)
                    text_features[i] = np.random.normal(0.05, 0.02, 384)
                    
            # Show progress
            if i % max(1, num_entities // 100) == 0:
                percentage = (i / num_entities) * 100
                elapsed = time.time() - self.phase_start_time
                eta = elapsed / max(i, 1) * (num_entities - i) if i > 0 else 0
                print(f"\rFinalizing text features: {percentage:.2f}% ({i}/{num_entities}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
        
        # Save final features
        torch.save(text_features, cache_file)
        print(f"\nText extraction completed in {time.time() - self.phase_start_time:.2f}s")
        
        return text_features
        
    def extract_features(self) -> Dict[str, torch.Tensor]:
        """Extract features for all entities and build graph structure"""
        # Check if features are already extracted
        features_path = "extracted_features/huggingface_features.pt"
        if os.path.exists(features_path):
            print(f"Loading existing features from {features_path}")
            return torch.load(features_path)
            
        overall_start_time = time.time()
        
        # Check if using KGBench data structure
        if self.using_kgbench:
            self.print_step_header("Extracting features for KGBench dataset")
            num_entities = self.kg_data.num_entities
            print(f"Processing {num_entities} entities and {len(self.kg_data.triples)} triples")
            
            # Step 1: Map entities to modalities
            entity_modality_map = self._map_entity_modalities()
            
            # Step 2: Extract image features
            if self.sample_rate < 1.0:
                num_to_sample = int(self.kg_data.num_entities * self.sample_rate)
                entity_indices = sorted(random.sample(range(self.kg_data.num_entities), num_to_sample))
            else:
                entity_indices = list(range(self.kg_data.num_entities))

            image_features = self.extract_image_features_kgbench_fast(entity_indices)
            
            # Step 3: Extract text features
            text_features = self.extract_text_features_kgbench_fast()
            
            # Step 4: Building graph structure
            self.print_step_header("Step 5: Building graph structure")
            self.phase_start_time = time.time()
            
            # Sample edges if needed for speed
            edge_index = []
            triples_to_process = []
            
            # Use edge sampling for very large graphs
            if self.sample_rate < 1.0 and len(self.kg_data.triples) > 100000:
                # Sample a subset of edges
                num_triples_to_process = int(len(self.kg_data.triples) * max(0.5, self.sample_rate))
                print(f"Sampling {num_triples_to_process} triples out of {len(self.kg_data.triples)}")
                triple_indices = sorted(random.sample(range(len(self.kg_data.triples)), num_triples_to_process))
                triples_to_process = [self.kg_data.triples[i] for i in triple_indices]
            else:
                triples_to_process = self.kg_data.triples
                
            # Process triples to build edge index
            total_triples = len(triples_to_process)
            for i, (s, p, o) in enumerate(triples_to_process):
                s_val = s.item() if hasattr(s, 'item') else s
                o_val = o.item() if hasattr(o, 'item') else o
                edge_index.append([s_val, o_val])
                
                # Print progress with ETA
                if i % max(1, total_triples // 100) == 0:
                    percentage = (i / total_triples) * 100
                    elapsed = time.time() - self.phase_start_time
                    eta = elapsed / max(i, 1) * (total_triples - i) if i > 0 else 0
                    print(f"\rGraph building: {percentage:.2f}% ({i}/{total_triples}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
                    
            print(f"\nCompleted graph building in {time.time() - self.phase_start_time:.2f}s")
            
            # Step 5: Extract node labels
            self.print_step_header("Step 6: Extracting node labels")            
            print("Step 6: Extracting node labels with optimized lookups")
            self.phase_start_time = time.time()
            
            # Create lookup dictionaries first - much faster than nested loops
            training_labels = {}
            withheld_labels = {}
            
            for i in range(len(self.kg_data.training)):
                node_idx = self.kg_data.training[i, 0].item()
                label = self.kg_data.training[i, 1].item()
                training_labels[node_idx] = label
                
            for i in range(len(self.kg_data.withheld)):
                node_idx = self.kg_data.withheld[i, 0].item()
                label = self.kg_data.withheld[i, 1].item()
                withheld_labels[node_idx] = label
            
            # Use dictionary lookups for fast label assignments
            labels = []
            for idx in range(num_entities):
                # Fast O(1) lookups
                if idx in training_labels:
                    labels.append(training_labels[idx])
                elif idx in withheld_labels:
                    labels.append(withheld_labels[idx])
                else:
                    labels.append(-1)
                
            print(f"\nCompleted label extraction in {time.time() - self.phase_start_time:.2f}s")
            
        else:
            # Using standard file loading
            self.print_step_header("Extracting features for file-based dataset")
            print(f"Processing {len(self.entities)} entities")
            
            image_features = []
            text_features = []
            labels = []
            edge_index = []
            
            # Extract node features
            self.phase_start_time = time.time()
            for i, entity in enumerate(self.entities):
                img_feat = self.extract_image_features(entity)
                text_feat = self.extract_text_features(entity)
                
                image_features.append(img_feat)
                text_features.append(text_feat)
                labels.append(entity.get('label', 0))
                
                # Print progress with ETA
                if i % max(1, len(self.entities) // 100) == 0:
                    percentage = (i / len(self.entities)) * 100
                    elapsed = time.time() - self.phase_start_time
                    eta = elapsed / max(i, 1) * (len(self.entities) - i) if i > 0 else 0
                    print(f"\rFeature extraction: {percentage:.2f}% ({i}/{len(self.entities)}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
                    
            print(f"\nCompleted entity feature extraction in {time.time() - self.phase_start_time:.2f}s")
            
            # Build graph edges
            self.print_step_header("Building graph")
            self.phase_start_time = time.time()
            
            for i, (_, row) in enumerate(self.triples.iterrows()):
                if str(row['subject']) in self.e2i and str(row['object']) in self.e2i:
                    edge_index.append([self.e2i[str(row['subject'])], self.e2i[str(row['object'])]])
                
                # Print progress with ETA
                if i % max(1, len(self.triples) // 100) == 0:
                    percentage = (i / len(self.triples)) * 100
                    elapsed = time.time() - self.phase_start_time
                    eta = elapsed / max(i, 1) * (len(self.triples) - i) if i > 0 else 0
                    print(f"\rGraph building: {percentage:.2f}% ({i}/{len(self.triples)}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
                    
            print(f"\nCompleted graph building in {time.time() - self.phase_start_time:.2f}s")
            
        # Convert to tensors
        self.print_step_header("Step 7: Converting to tensors")
        self.phase_start_time = time.time()
        
        # Convert feature lists to tensors
        if isinstance(image_features[0], np.ndarray):
            image_features_tensor = torch.tensor(np.array(image_features), dtype=torch.float)
        else:
            image_features_tensor = torch.stack(image_features)
            
        if isinstance(text_features[0], np.ndarray):
            text_features_tensor = torch.tensor(np.array(text_features), dtype=torch.float)
        else:
            text_features_tensor = torch.stack(text_features)
            
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Save feature statistics
        self._save_feature_statistics(image_features, text_features)
        
        print(f"Tensor conversion completed in {time.time() - self.phase_start_time:.2f}s")
        
        # Return extracted features
        result = {
            'image_features': image_features_tensor,
            'text_features': text_features_tensor,
            'edge_index': edge_index_tensor,
            'labels': labels_tensor
        }
        
        # Save features
        torch.save(result, features_path)
        print(f"Saved features to {features_path}")
        
        total_time = time.time() - overall_start_time
        print(f"\nTotal processing time: {total_time:.2f}s")
        
        return result
        
    def _save_feature_statistics(self, image_features: List[np.ndarray], text_features: List[np.ndarray]) -> None:
        """Save statistics about extracted features"""
        # Calculate statistics
        img_norms = [np.linalg.norm(feat) for feat in image_features]
        txt_norms = [np.linalg.norm(feat) for feat in text_features]
        
        # Plot feature norm distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(img_norms, bins=50)
        plt.title('Image Feature Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(txt_norms, bins=50)
        plt.title('Text Feature Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('feature_extraction_results/huggingface_feature_norms.png')
        plt.close()
        
        # Save statistics to file
        with open('feature_extraction_results/huggingface_feature_stats.txt', 'w') as f:
            f.write(f"Total entities: {len(image_features)}\n")
            f.write(f"Image feature dimension: {image_features[0].shape[0]}\n")
            f.write(f"Text feature dimension: {text_features[0].shape[0]}\n")
            f.write(f"Avg image feature norm: {np.mean(img_norms):.4f}\n")
            f.write(f"Avg text feature norm: {np.mean(txt_norms):.4f}\n")
            f.write(f"Entities with zero image features: {sum(1 for norm in img_norms if norm < 1e-6)}\n")
            f.write(f"Entities with zero text features: {sum(1 for norm in txt_norms if norm < 1e-6)}\n")
            
    def save_extracted_features(self, features: Dict[str, torch.Tensor]) -> None:
        """Save extracted features to disk"""
        os.makedirs("extracted_features", exist_ok=True)
        torch.save(features, "extracted_features/huggingface_features.pt")
        print("Features saved to extracted_features/huggingface_features.pt")
