import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from collections import Counter
import kgbench as kg
import argparse
from PIL import Image
import io
import base64

class DMG777KExplorer:
    def __init__(self, dataset_name="dmg777k", output_dir='eda_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading {dataset_name} dataset from kgbench...")
        self.data = kg.load(dataset_name)
        
        self._print_dataset_info()
        
        self._process_dataset()
    
    def _print_dataset_info(self):
        print("\nDataset Information:")
        print(f"  Number of entities: {self.data.num_entities}")
        print(f"  Number of relations: {self.data.num_relations}")

        class_labels={}
        
        if isinstance(self.data.triples, np.ndarray):
            print(f"  Number of triples: {len(self.data.triples)}")
        elif isinstance(self.data.triples, tuple):
            print(f"  Number of triples: {len(self.data.triples[0])}")
        
        print(f"  Training instances: {len(self.data.training)}")
        if hasattr(self.data, 'withheld'):
            for i, label in enumerate(self.data.withheld):
                node_idx = label[0]
                class_label = label[1]
                class_labels[node_idx] = class_label
        if hasattr(self.data, 'testing'):
            print(f"  Testing instances: {len(self.data.testing)}")
        
        print(f"  Number of classes: {self.data.num_classes}")
        
        print("  Available datatypes:")
        for dt in self.data.datatypes():
            print(f"    - {dt}")
    
    def _extract_entity_types(self):
        self.entity_types = {}
        
        if isinstance(self.data.triples, np.ndarray):
            for s, p, o in self.data.triples:
                pred_name = self.i2r[p].lower()
                if "type" in pred_name or "category" in pred_name or "class" in pred_name:
                    entity_type = str(self.i2e[o])
                    if entity_type not in self.entity_types:
                        self.entity_types[entity_type] = 0
                    self.entity_types[entity_type] += 1
        
        if not self.entity_types:
            print("No entity types found via type predicates, determining from relation patterns...")
            
            entity_relations = {}
            
            if isinstance(self.data.triples, np.ndarray):
                for s, p, o in self.data.triples:
                    if s not in entity_relations:
                        entity_relations[s] = {"out": set(), "in": set()}
                    entity_relations[s]["out"].add(p)
                    
                    if o not in entity_relations:
                        entity_relations[o] = {"out": set(), "in": set()}
                    entity_relations[o]["in"].add(p)
            
            pattern_groups = {}
            for entity, relations in entity_relations.items():
                out_pattern = tuple(sorted(relations["out"]))
                in_pattern = tuple(sorted(relations["in"]))
                pattern = (out_pattern, in_pattern)
                
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(entity)
            
            top_patterns = sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True)[:20]
            for i, (pattern, entities) in enumerate(top_patterns):
                type_name = f"Pattern_{i+1}"
                self.entity_types[type_name] = len(entities)
                
            print(f"Created {len(self.entity_types)} entity types from relation patterns")
    
    def _process_dataset(self):
        self.i2e = self.data.i2e  
        self.e2i = {self.data.i2e[i]: i for i in range(len(self.i2e))}  
        
        self.i2r = self.data.i2r  
        self.r2i = {self.i2r[i]: i for i in range(len(self.i2r))}          
        self._extract_entity_types()
        
        self.literal_datatypes = {}
        
        datatypes = self.data.datatypes()
        
        for dt in datatypes:
            try:
                literals = self.data.datatype_g2l(dt)
                self.literal_datatypes[dt] = len(literals)
                print(f"Found {len(literals)} entities with {dt} data")
            except Exception as e:
                print(f"Error accessing datatype {dt}: {e}")
        
        self.modality_groups = {
            'Numerical': sum(self.literal_datatypes.get(dt, 0) for dt in 
                        ['http://www.w3.org/2001/XMLSchema#nonNegativeInteger', 
                         'http://www.w3.org/2001/XMLSchema#positiveInteger',
                         'http://www.w3.org/2001/XMLSchema#boolean']),
            'Temporal': sum(self.literal_datatypes.get(dt, 0) for dt in 
                        ['http://www.w3.org/2001/XMLSchema#gYear']),
            'Textual': sum(self.literal_datatypes.get(dt, 0) for dt in 
                      ['iri', 'none', '@es', '@fy', '@nl', '@nl-nl', '@pt', '@ru',
                       'http://www.w3.org/2001/XMLSchema#anyURI']),
            'Visual': sum(self.literal_datatypes.get(dt, 0) for dt in 
                     ['http://kgbench.info/dt#base64Image']),
            'Spatial': sum(self.literal_datatypes.get(dt, 0) for dt in 
                      ['http://www.opengis.net/ont/geosparql#wktLiteral'])
        }
        
        self.triples_df = []
        
        if isinstance(self.data.triples, np.ndarray):
            for s, p, o in tqdm(self.data.triples, desc="Processing triples"):
                subj = str(self.i2e[s])
                pred = str(self.i2r[p])
                obj = str(self.i2e[o])
                self.triples_df.append([subj, pred, obj, s, p, o])
        
        elif isinstance(self.data.triples, tuple) and len(self.data.triples) == 3:
            for s, p, o in tqdm(zip(*self.data.triples), desc="Processing triples"):
                subj = str(self.i2e[s])
                pred = str(self.i2r[p])
                obj = str(self.i2e[o])
                self.triples_df.append([subj, pred, obj, s, p, o])
        
        self.triples_df = pd.DataFrame(self.triples_df, 
                                      columns=['subject', 'predicate', 'object', 
                                               'subject_idx', 'predicate_idx', 'object_idx'])
        
        self._build_graph()
    
    def _build_graph(self):
        self.G = nx.DiGraph()
        
        for i in tqdm(range(self.data.num_entities), desc="Building graph nodes"):
            label = -1
            
            if i in self.data.training:
                idx = np.where(self.data.training == i)[0][0]
                if hasattr(self.data, 'training_labels'):
                    label = int(self.data.training_labels[idx])
            
            if label == -1 and hasattr(self.data, 'withheld'):
                if i in self.data.withheld:
                    idx = np.where(self.data.withheld == i)[0][0]
                    if hasattr(self.data, 'withheld_labels'):
                        label = int(self.data.withheld_labels[idx])
            
            has_numerical = False
            has_temporal = False
            has_textual = False
            has_visual = False
            has_spatial = False
            
            for dt in self.literal_datatypes:
                try:
                    literals = self.data.datatype_g2l(dt)
                    if i in literals:
                        if dt in ['http://www.w3.org/2001/XMLSchema#nonNegativeInteger', 
                                 'http://www.w3.org/2001/XMLSchema#positiveInteger',
                                 'http://www.w3.org/2001/XMLSchema#boolean']:
                            has_numerical = True
                        elif dt in ['http://www.w3.org/2001/XMLSchema#gYear']:
                            has_temporal = True
                        elif dt in ['iri', 'none', '@es', '@fy', '@nl', '@nl-nl', '@pt', '@ru',
                                  'http://www.w3.org/2001/XMLSchema#anyURI']:
                            has_textual = True
                        elif dt in ['http://kgbench.info/dt#base64Image']:
                            has_visual = True
                        elif dt in ['http://www.opengis.net/ont/geosparql#wktLiteral']:
                            has_spatial = True
                except:
                    pass
            
            self.G.add_node(i, 
                           label=label,
                           has_numerical=has_numerical,
                           has_temporal=has_temporal,
                           has_textual=has_textual,
                           has_visual=has_visual,
                           has_spatial=has_spatial)
        
        if isinstance(self.data.triples, np.ndarray):
            for s, p, o in tqdm(self.data.triples, desc="Building graph edges"):
                self.G.add_edge(s, o, relation=p, relation_name=self.i2r[p])
        elif isinstance(self.data.triples, tuple) and len(self.data.triples) == 3:
            for s, p, o in tqdm(zip(*self.data.triples), desc="Building graph edges"):
                self.G.add_edge(s, o, relation=p, relation_name=self.i2r[p])
        
        print(f"\nGraph created with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    
    def analyze_entity_types(self):
        print("Analyzing entity types...")
        
        if self.entity_types:
            sorted_types = sorted(self.entity_types.items(), key=lambda x: x[1], reverse=True)
            top_types = sorted_types[:15]  
            
            type_labels = []
            for type_uri, count in top_types:
                if '#' in type_uri:
                    label = type_uri.split('#')[-1]
                elif '/' in type_uri:
                    label = type_uri.split('/')[-1]
                else:
                    label = type_uri.split(':')[-1] if ':' in type_uri else type_uri
                    
                if len(label) > 20:
                    label = label[:20] + "..."
                    
                type_labels.append(label)
            
            type_counts = [count for _, count in top_types]
            
            plt.figure(figsize=(15, 8))
            bars = plt.bar(type_labels, type_counts)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=10)
            
            plt.title('Top 15 Entity Types in DMG777K', fontsize=16)
            plt.xlabel('Entity Type', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'entity_type_distribution.png'))
            plt.close()
        else:
            degree_dist = [d for _, d in self.G.degree()]
            
            plt.figure(figsize=(15, 8))
            plt.hist(degree_dist, bins=30, log=True)
            plt.title('Node Degree Distribution (Log Scale) - Proxy for Entity Types', fontsize=16)
            plt.xlabel('Node Degree', fontsize=14)
            plt.ylabel('Count (Log Scale)', fontsize=14)
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'entity_type_distribution.png'))
            plt.close()
        
        return self.entity_types
    
    def analyze_edge_types(self):
        print("Analyzing edge types...")
        
        pred_counts = self.triples_df['predicate'].value_counts()
        
        top_preds = pred_counts.head(20)
        
        plt.figure(figsize=(15, 10))
        bars = plt.barh(top_preds.index, top_preds.values)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, i, f'{int(width):,}', va='center', fontsize=10)
        
        plt.title('Top 20 Edge Types (Predicates) in DMG777K', fontsize=16)
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Predicate', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'edge_type_distribution.png'))
        plt.close()
        
        in_degrees = [d for _, d in self.G.in_degree()]
        out_degrees = [d for _, d in self.G.out_degree()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(in_degrees, bins=50, log=True)
        ax1.set_title('In-Degree Distribution (log scale)', fontsize=14)
        ax1.set_xlabel('In-Degree', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(linestyle='--', alpha=0.7)
        
        ax2.hist(out_degrees, bins=50, log=True)
        ax2.set_title('Out-Degree Distribution (log scale)', fontsize=14)
        ax2.set_xlabel('Out-Degree', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'degree_distributions.png'))
        plt.close()
        
        return {
            "total_edge_types": len(pred_counts),
            "top_20_predicates": pred_counts.head(20).to_dict()
        }
    
    def analyze_modality_availability(self):
        print("Analyzing multimodal features...")
        
        plt.figure(figsize=(12, 6))
        
        if self.modality_groups:
            non_zero = {k: v for k, v in self.modality_groups.items() if v > 0}
            labels = list(non_zero.keys())
            values = list(non_zero.values())
            
            bars = plt.bar(labels, values)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=10)
                
                percentage = height / self.data.num_entities * 100
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{percentage:.1f}%', ha='center', va='center', color='white', fontsize=12)
            
            plt.title('Multimodal Feature Availability in DMG777K', fontsize=16)
            plt.ylabel('Count', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No multimodal data available", 
                   ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            plt.title('Multimodal Feature Availability', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'modality_availability.png'))
        plt.close()
        
        return {
            "modality_counts": self.modality_groups,
            "datatypes": {dt: count for dt, count in self.literal_datatypes.items()}
        }
    
    def analyze_triple_patterns(self):
        print("Analyzing triple patterns...")
        
        sp_combinations = self.triples_df.groupby(['subject_idx', 'predicate_idx']).size()
        
        sp_counts = Counter(sp_combinations.values)
        
        plt.figure(figsize=(15, 6))
        
        labels = [str(k) for k in sorted(sp_counts.keys())[:20]]  # Top 20 for clarity
        values = [sp_counts[int(k)] for k in labels]
        
        bars = plt.bar(labels, values)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=10)
        
        plt.title('Subject-Predicate Pattern Frequency in DMG777K', fontsize=16)
        plt.xlabel('Number of Occurrences', fontsize=14)
        plt.ylabel('Count of Patterns', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'triple_patterns.png'))
        plt.close()
        
        unique_subjects = self.triples_df['subject_idx'].nunique()
        avg_triples_per_entity = len(self.triples_df) / unique_subjects if unique_subjects > 0 else 0
        
        triples_per_subject = self.triples_df['subject_idx'].value_counts()
        
        plt.figure(figsize=(12, 6))
        plt.hist(triples_per_subject.values, bins=50)
        plt.axvline(x=avg_triples_per_entity, color='r', linestyle='--')
        plt.text(avg_triples_per_entity, plt.ylim()[1]*0.9, 
               f"Avg: {avg_triples_per_entity:.2f}", 
               color='r', ha='right')
        
        plt.title('Triples per Entity Distribution in DMG777K', fontsize=16)
        plt.xlabel('Number of Triples', fontsize=14)
        plt.ylabel('Count of Entities', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'triples_per_entity.png'))
        plt.close()
        
        return {
            "total_triples": len(self.triples_df),
            "unique_subjects": unique_subjects,
            "unique_objects": self.triples_df['object_idx'].nunique(),
            "avg_triples_per_entity": float(avg_triples_per_entity),
            "unique_sp_patterns": len(sp_combinations)
        }
    
    def validate_entity_mappings(self):
        print("Validating entity mappings...")
        
        id_lengths = [len(str(self.i2e[i])) for i in range(self.data.num_entities)]
        
        plt.figure(figsize=(15, 6))
        
        bins = min(50, len(set(id_lengths)))
        plt.hist(id_lengths, bins=bins)
        
        counter = Counter(id_lengths)
        for length, count in counter.most_common(5):
            plt.text(length, count, f"{count:,}", 
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Entity ID Length Distribution in DMG777K', fontsize=16)
        plt.xlabel('ID Length', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'entity_id_lengths.png'))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.title('Entity Mapping Examples (i2e/e2i)', fontsize=16)
        
        cell_text = []
        for i in range(min(5, self.data.num_entities)):
            entity_id = self.i2e[i]
            e2i_lookup = self.e2i.get(entity_id, "Not found")
            roundtrip_check = e2i_lookup == i
            
            cell_text.append([
                str(i),
                str(entity_id)[:30] + "..." if len(str(entity_id)) > 30 else str(entity_id),
                str(e2i_lookup),
                "✓" if roundtrip_check else "✗"
            ])
        
        table = plt.table(cellText=cell_text, 
                        colLabels=["Index (i)", "Entity ID (e)", "e2i[e]", "Roundtrip Check"],
                        loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'entity_mapping_examples.png'))
        plt.close()
        
        mapping_errors = 0
        for i in range(min(1000, self.data.num_entities)):
            entity_id = self.i2e[i]
            if entity_id in self.e2i and self.e2i[entity_id] != i:
                mapping_errors += 1
        
        return {
            "id_length_stats": {
                "min": min(id_lengths),
                "max": max(id_lengths),
                "mean": float(np.mean(id_lengths)),
                "unique_lengths": len(set(id_lengths))
            },
            "mapping_errors": mapping_errors
        }
    
    def analyze_gnn_preparation(self):
        print("Analyzing GNN preparation metrics...")
        
        components = list(nx.weakly_connected_components(self.G))
        component_sizes = [len(c) for c in components]
        
        plt.figure(figsize=(12, 6))
        
        if len(component_sizes) > 1:
            plt.hist(component_sizes, bins=30, log=True)
            plt.title('Connected Component Size Distribution in DMG777K', fontsize=16)
            plt.xlabel('Component Size', fontsize=14)
            plt.ylabel('Count (log scale)', fontsize=14)
            
            plt.axvline(x=max(component_sizes), color='r', linestyle='--')
            plt.text(max(component_sizes), plt.ylim()[1]*0.9, 
                   f"Giant component: {max(component_sizes):,} nodes", 
                   color='r', ha='right')
        else:
            plt.figure(figsize=(12, 6))
            plt.bar([1], [component_sizes[0]])
            plt.text(1, component_sizes[0]/2, 
                   f"Giant component: {component_sizes[0]:,} nodes", 
                   ha='center', va='center', fontsize=14, color='white')
            plt.title('Connected Component Analysis for DMG777K', fontsize=16)
            plt.xlabel('Component Index', fontsize=14)
            plt.ylabel('Size', fontsize=14)
            plt.xlim(0.5, 1.5)          
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'component_sizes.png'))
        plt.close()
        
        class_labels = {}
        
        for i, label in enumerate(self.data.training):
            node_idx = label[0]  
            class_label = label[1] 
            class_labels[node_idx] = class_label
        
        if hasattr(self.data, 'withheld') and hasattr(self.data, 'withheld_labels'):
            for i, label in zip(self.data.withheld, self.data.withheld_labels):
                class_labels[i] = int(label)
        
        class_distribution = Counter(class_labels.values())
        
        plt.figure(figsize=(10, 6))
        
        if class_distribution:
            labels = sorted(class_distribution.keys())
            values = [class_distribution[label] for label in labels]
            
            bars = plt.bar([str(label) for label in labels], values)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                        f"{values[i]:,}", ha='center', va='bottom', fontsize=12)
            
            plt.title('Class Distribution in DMG777K', fontsize=16)
            plt.xlabel('Class Label', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No class data available", 
                   ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            plt.title('Class Distribution in DMG777K', fontsize=16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'))
        plt.close()
        
        homophily = None
        labeled_edges = []
        
        for u, v in self.G.edges():
            if u in class_labels and v in class_labels:
                labeled_edges.append((class_labels[u], class_labels[v]))
        
        if labeled_edges:
            same_label = sum(u == v for u, v in labeled_edges)
            homophily = float(same_label / len(labeled_edges))
        
        plt.figure(figsize=(15, 10))
        plt.axis('off')
        
        plt.text(0.5, 0.98, "GNN Preparation Guidelines for DMG777K", 
               ha='center', va='top', fontsize=16, weight='bold')
        
        guidelines = [
            "**Graph Structure Considerations:**",
            f"- The graph has {self.G.number_of_nodes():,} nodes and {self.G.number_of_edges():,} edges",
            f"- There are {len(components)} connected components",
            f"- The largest component contains {max(component_sizes):,} nodes ({(max(component_sizes)/self.G.number_of_nodes())*100:.1f}% of the graph)",
            f"- Average in-degree: {np.mean([d for _, d in self.G.in_degree()]):.2f}",
            f"- Average out-degree: {np.mean([d for _, d in self.G.out_degree()]):.2f}",
            
            "\n**Multimodal Feature Considerations:**",
            "- Available modalities: " + ", ".join([k for k, v in self.modality_groups.items() if v > 0]),
            "- For visual data: Use CNN-based encoders (ResNet, EfficientNet)",
            "- For textual data: Use transformer-based encoders (BERT, RoBERTa)",
            "- For spatial data: Use geometric encodings or positional embeddings",
            
            "\n**Training Considerations:**",
            f"- Class distribution: {len(class_distribution)} classes with {min(class_distribution.values()) if class_distribution else 0} to {max(class_distribution.values()) if class_distribution else 0} instances per class",
            f"- Homophily (same-class connections): {homophily:.2f}" if homophily is not None else "- Homophily: Unknown (insufficient labeled edges)",
            "- Consider message-passing GNN architectures like R-GCN or R-GAT for heterogeneous relations",
            "- Use CLIP-like joint embedding for multimodal feature fusion"
        ]
        
        guidelines_text = "\n".join(guidelines)
        plt.text(0.05, 0.95, guidelines_text, 
               ha='left', va='top', fontsize=12, 
               transform=plt.gca().transAxes,
               wrap=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gnn_preparation_guidelines.png'))
        plt.close()
        
        return {
            "num_components": len(components),
            "largest_component_size": max(component_sizes) if component_sizes else None,
            "largest_component_percentage": (max(component_sizes) / self.G.number_of_nodes()) * 100 if component_sizes else None,
            "class_distribution": {str(k): v for k, v in class_distribution.items()},
            "homophily": homophily,
            "avg_degree": float(np.mean([d for _, d in self.G.degree()]))
        }
    
    def generate_eda_report(self):
        print("Starting comprehensive EDA for SAiDL tasks 1a-f...")
        
        report = {}
        
        try:
            report['task_1a_entity_types'] = self.analyze_entity_types()
            print("Completed task 1a: Entity type analysis")
        except Exception as e:
            print(f"Error in entity type analysis: {e}")
            report['task_1a_entity_types'] = str(e)
        
        try:
            report['task_1b_edge_types'] = self.analyze_edge_types()
            print("Completed task 1b: Edge type analysis")
        except Exception as e:
            print(f"Error in edge type analysis: {e}")
            report['task_1b_edge_types'] = str(e)
        
        try:
            report['task_1c_modalities'] = self.analyze_modality_availability()
            print("Completed task 1c: Modality analysis")
        except Exception as e:
            print(f"Error in modality analysis: {e}")
            report['task_1c_modalities'] = str(e)
        
        try:
            report['task_1d_triple_patterns'] = self.analyze_triple_patterns()
            print("Completed task 1d: Triple pattern analysis")
        except Exception as e:
            print(f"Error in triple pattern analysis: {e}")
            report['task_1d_triple_patterns'] = str(e)
        
        try:
            report['task_1e_entity_mappings'] = self.validate_entity_mappings()
            print("Completed task 1e: Entity mapping validation")
        except Exception as e:
            print(f"Error in entity mapping validation: {e}")
            report['task_1e_entity_mappings'] = str(e)
        
        try:
            report['task_1f_gnn_preparation'] = self.analyze_gnn_preparation()
            print("Completed task 1f: GNN preparation analysis")
        except Exception as e:
            print(f"Error in GNN preparation analysis: {e}")
            report['task_1f_gnn_preparation'] = str(e)
        
        report['dataset_summary'] = {
            'name': 'dmg777k',
            'num_entities': self.data.num_entities,
            'num_relations': self.data.num_relations,
            'num_triples': len(self.triples_df),
            'num_classes': self.data.num_classes,
            'training_instances': len(self.data.training),
            'validation_instances': len(self.data.withheld) if hasattr(self.data, 'withheld') else 0
        }
        
        with open(os.path.join(self.output_dir, 'eda_report.json'), 'w') as f:
            json_report = json.dumps(report, indent=2, default=str)
            f.write(json_report)
        
        print(f"EDA complete. Results saved to {self.output_dir}")
        return report

def main():
    parser = argparse.ArgumentParser(description='Run EDA on kgbench datasets')
    parser.add_argument('--dataset_name', type=str, default='dmg777k', help='Dataset name (default: dmg777k)')
    parser.add_argument('--output_dir', type=str, default='eda_results', help='Output directory for results')
    args = parser.parse_args()
    
    try:
        explorer = DMG777KExplorer(dataset_name=args.dataset_name, output_dir=args.output_dir)
        explorer.generate_eda_report()
    except Exception as e:
        print(f"EDA Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
