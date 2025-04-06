import os
import json
import base64
import io
from PIL import Image
from tqdm import tqdm
import kgbench as kg

def analyze_entities_structure():
    """Analyze the structure of entities.json to understand its format"""
    entities_path = 'dmg777k_dataset/entities.json'
    with open(entities_path) as f:
        entities = json.load(f)
    
    # Check the first 5 entities
    print("First 5 entities structure:")
    for i, entity in enumerate(entities[:5]):
        print(f"Entity {i}:")
        for key, value in entity.items():
            value_type = type(value).__name__
            value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else value
            print(f"  {key} ({value_type}): {value_preview}")
    
    # Count different ID types
    id_types = {}
    for entity in entities:
        if 'id' in entity:
            id_type = type(entity['id']).__name__
            id_types[id_type] = id_types.get(id_type, 0) + 1
    
    print("\nID Types:")
    for id_type, count in id_types.items():
        print(f"{id_type}: {count}")
    
    return entities

def extract_images_safely(dataset_name='dmg777k', output_dir='dmg777k_dataset/images'):
    """Extract images with safe handling of entity ID types"""
    print(f"Loading {dataset_name} dataset from kgbench...")
    data = kg.load(dataset_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load entities.json
    entities_path = 'dmg777k_dataset/entities.json'
    with open(entities_path) as f:
        entities = json.load(f)
    
    # First analyze the entities structure
    print("Analyzing entities structure...")
    analyze_entities_structure()
    
    # Image datatype in kgbench
    image_datatype = "http://kgbench.info/dt#base64Image"
    
    print("Extracting images from kgbench dataset...")
    image_count = 0
    
    # Use get_images() method if available
    if hasattr(data, 'get_images') and callable(data.get_images):
        try:
            print("Using get_images() method...")
            images = data.get_images()
            print(f"Found {len(images)} images with get_images()")
            
            # Match these images to entities using data.datatype_l2g
            if hasattr(data, 'datatype_l2g') and callable(data.datatype_l2g):
                image_indices = data.datatype_l2g(image_datatype)
                print(f"Found {len(image_indices)} image indices")
                
                for local_idx, image in enumerate(images):
                    if local_idx < len(image_indices):
                        global_idx = image_indices[local_idx]
                        
                        # Find entity this image belongs to
                        for s, p, o in data.triples:
                            if o == global_idx:
                                subject_id = data.i2e[s] if hasattr(data, 'i2e') else str(s)
                                
                                # Save image
                                img_filename = f"entity_{s}_image_{local_idx}.png"
                                img_path = os.path.join(output_dir, img_filename)
                                image.save(img_path)
                                
                                # Update entity in entities.json - without using IDs as keys
                                for entity in entities:
                                    # Try to match entity safely
                                    entity_id = entity.get('id', None)
                                    if entity_id == subject_id or str(entity_id) == str(subject_id):
                                        entity['image'] = img_filename
                                        image_count += 1
                                        break
                                
                                break
        except Exception as e:
            print(f"Error using get_images(): {e}")
    
    # If we didn't get any images, try the literals approach
    if image_count == 0 and hasattr(data, 'literals') and data.literals:
        print("Trying extraction from literals...")
        
        for lit_id, lit_value in tqdm(data.literals.items()):
            # Check for image literals
            is_image = False
            base64_str = None
            
            if isinstance(lit_value, dict) and 'datatype' in lit_value:
                is_image = lit_value['datatype'] == image_datatype
                base64_str = lit_value.get('value', '')
            elif isinstance(lit_value, str) and (
                lit_value.startswith('data:image') or 
                ';base64,' in lit_value
            ):
                is_image = True
                base64_str = lit_value
            
            if is_image and base64_str:
                try:
                    # Extract entity ID from literal ID
                    parts = lit_id.split('#') if '#' in lit_id else lit_id.split('/')
                    entity_id = parts[0]
                    
                    # Clean base64 string
                    if ',' in base64_str:
                        base64_data = base64_str.split(',', 1)[1]
                    else:
                        base64_data = base64_str
                    
                    # Decode image
                    image_data = base64.b64decode(base64_data)
                    img = Image.open(io.BytesIO(image_data))
                    
                    # Save image
                    img_filename = f"lit_{image_count}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    img.save(img_path)
                    
                    # Update entity - without using IDs as keys
                    for entity in entities:
                        # Try to match entity safely
                        e_id = entity.get('id', None)
                        if (e_id == entity_id or str(e_id) == str(entity_id) or 
                            (isinstance(e_id, list) and str(entity_id) in map(str, e_id))):
                            entity['image'] = img_filename
                            image_count += 1
                            break
                    
                except Exception as e:
                    print(f"Error processing image from literal {lit_id}: {e}")
    
    # Save updated entities.json
    with open(entities_path, 'w') as f:
        json.dump(entities, f)
    
    print(f"Extracted {image_count} images to {output_dir}")
    return image_count

if __name__ == "__main__":
    # First analyze the structure
    print("Analyzing entities structure...")
    entities = analyze_entities_structure()
    
    # Extract images with safe handling of entity IDs
    print("\nExtracting images safely...")
    count = extract_images_safely()
    
    if count == 0:
        print("\nNo images were extracted. You have a few options:")
        print("1. Continue without images - the GNN might work with text-only features")
        print("2. Create synthetic images (e.g., album covers with text/colors) as placeholders")
        print("3. The dataset might not include images in this version")
