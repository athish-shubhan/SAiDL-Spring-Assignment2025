import os
import time
from datetime import datetime
import json

# Best GNN models for multimodal graphs (based on research)
best_gnns = ["gcn", "sage"]  

# All feature extractors
feature_extractors = ["simple", "huggingface", "intermediate"]

# Create timestamped results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"focused_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Extract all features first (one-time operation)
default_sample_rate = 0.05
sampler_config = {
    'modality_weights': {'text': 0.4, 'image': 0.4, 'spatial': 0.2},
    'learn_rate': True
}

# Extract all features first (one-time operation)
for feature_type in feature_extractors:
    print(f"\n{'='*50}")
    print(f"EXTRACTING {feature_type.upper()} FEATURES")
    print(f"{'='*50}")
    
    features_path = f"extracted_features/{feature_type}_features.pt"
    if os.path.exists(features_path):
        print(f"Using existing {feature_type} features from {features_path}")
        continue
    
    # Save sampler config to file for reference
    with open(f"extracted_features/{feature_type}_sampler_config.json", 'w') as f:
        json.dump({
            'sample_rate': default_sample_rate,
            'sampler_config': sampler_config,
            'extraction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
        
    if feature_type == "simple":
        from sfeatures import SimpleFeatureExtractor
        extractor = SimpleFeatureExtractor('dmg777k_dataset', sample_rate=default_sample_rate)
        features = extractor.extract_features()
    elif feature_type == "huggingface":
        from huggingface import HuggingFaceFeatureExtractor
        extractor = HuggingFaceFeatureExtractor('dmg777k_dataset', sample_rate=default_sample_rate)
        features = extractor.extract_features()
        extractor.save_extracted_features(features)
    elif feature_type == "intermediate":
        from sfeatures import SimpleFeatureExtractor
        from customencoder import IntermediateFeatureExtractor
        simple_extractor = SimpleFeatureExtractor('dmg777k_dataset', sample_rate=default_sample_rate)
        intermediate_extractor = IntermediateFeatureExtractor(simple_extractor)
        features = intermediate_extractor.extract_features()

for gnn_type in best_gnns:
    for feature_type in feature_extractors:
        combo_dir = os.path.join(results_dir, f"{gnn_type}_{feature_type}")
        os.makedirs(combo_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"TRAINING {gnn_type.upper()} WITH {feature_type.upper()} FEATURES")
        print(f"{'='*50}")
        
        # Run training with current combination
        start_time = time.time()
        cmd = f"python train.py --gnn_type {gnn_type} --feature_type {feature_type} --output_dir {combo_dir}"
        exit_code = os.system(cmd)
        
        # Record training information
        training_time = time.time() - start_time
        with open(os.path.join(combo_dir, "training_info.txt"), "w") as f:
            f.write(f"GNN type: {gnn_type}\n")
            f.write(f"Feature type: {feature_type}\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
            f.write(f"Exit code: {exit_code}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\nTraining complete! Results saved to {results_dir}")
print("\n==== CREATING SUMMARY REPORT ====")
with open(os.path.join(results_dir, "execution_summary.txt"), "w") as f:
    f.write(f"Multimodal Graph Learning Pipeline Summary\n")
    f.write(f"=======================================\n\n")
    f.write(f"Execution started: {timestamp}\n")
    f.write(f"Execution completed: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")
    f.write(f"Tested GNN architectures: {', '.join(best_gnns)}\n")
    f.write(f"Tested feature extractors: {', '.join(feature_extractors)}\n\n")
    f.write(f"Total combinations tested: {len(best_gnns) * len(feature_extractors)}\n")
    f.write(f"Results directory: {results_dir}\n")

print(f"\nPipeline execution complete! Results saved to {results_dir}")
print(f"Check {results_dir}/execution_summary.txt for execution details")
