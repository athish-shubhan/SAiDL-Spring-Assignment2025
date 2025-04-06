# In ablation script (not shown in search results)
ablation_configs = [
    {"name": "Full model", "disable_components": []},
    {"name": "Without cross-modal attention", "disable_components": ["cross_attention"]},
    {"name": "Image features only", "disable_components": ["text_features"]},
    {"name": "Text features only", "disable_components": ["image_features"]},
    {"name": "Without adaptive sampling", "disable_components": ["adaptive_sampling"]}
]

for config in ablation_configs:
    results = run_evaluation_with_config(config)
    ablation_results[config["name"]] = results

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - GraphSAGE + HuggingFace')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
