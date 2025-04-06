import os
import json
import pandas as pd
import numpy as np
import kgbench as kg
from tqdm import tqdm

def convert_kgbench_to_explorer_format(dataset_name, output_dir):
    print(f"Loading {dataset_name} dataset from kgbench...")
    data = kg.load(dataset_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)  # Still create the directory for consistency
    
    print("Converting entities...")
    entities = []
    for i in tqdm(range(data.num_entities)):
        entity = {
            'id': data.i2e[i] if hasattr(data, 'i2e') else str(i),
            'type': 'entity',
        }
        
        if hasattr(data, 'training') and i in data.training:
            idx = np.where(data.training == i)[0][0]
            if hasattr(data, 'training_labels'):
                entity['label'] = int(data.training_labels[idx])
        
        if hasattr(data, 'e2i') and hasattr(data, 'literals'):
            entity_id = data.i2e[i] if hasattr(data, 'i2e') else str(i)
            for lit_id, lit_value in data.literals.items():
                if lit_id.startswith(entity_id):
                    attr_name = lit_id.split('/')[-1] if '/' in lit_id else 'description'
                    entity[attr_name] = lit_value
                    
                    # Simply store the attribute, no image processing
                    # The image attribute will remain as the base64 string
        
        entities.append(entity)
    
    print("Saving entities.json...")
    with open(os.path.join(output_dir, 'entities.json'), 'w') as f:
        json.dump(entities, f)
    
    print("Converting triples...")
    triples = []
    for s, p, o in tqdm(data.triples):
        subject = data.i2e[s] if hasattr(data, 'i2e') else str(s)
        predicate = data.i2r[p]
        object_id = data.i2e[o] if hasattr(data, 'i2e') else str(o)
        triples.append([subject, predicate, object_id])
    
    triples_df = pd.DataFrame(triples)
    print("Saving triples.txt...")
    triples_df.to_csv(os.path.join(output_dir, 'triples.txt'), sep='\t', index=False, header=False)
    
    print(f"Conversion complete. Files saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    convert_kgbench_to_explorer_format('dmg777k', 'dmg77k_dataset')
