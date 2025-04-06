import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def generate_summary_tables_and_plots(sym_results, asym_results=None):

    tables_dir = "tables"
    plots_dir = "plots"
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    sym_df = pd.DataFrame(sym_results).T
    sym_df.index = [f"{float(rate):.1f}" for rate in sym_df.index]
    sym_df.index.name = "Noise Rate"
    sym_df.to_csv(f"{tables_dir}/symmetric_noise_summary.csv")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sym_df.index, sym_df["CE"], 'o-', label="CE", linewidth=2)
    plt.plot(sym_df.index, sym_df["NCE"], 's-', label="NCE", linewidth=2)
    plt.plot(sym_df.index, sym_df["FL"], '^-', label="FL", linewidth=2)
    plt.plot(sym_df.index, sym_df["NFL"], 'D-', label="NFL", linewidth=2)
    plt.title("Normalized vs Vanilla Losses (Symmetric Noise)", fontsize=14)
    plt.xlabel("Noise Rate (η)", fontsize=12)
    plt.ylabel("Best Test Accuracy (%)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/normalized_vs_vanilla_symmetric.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(sym_df.index, sym_df["NCE"], 'o-', label="NCE", linewidth=2)
    plt.plot(sym_df.index, sym_df["NFL"], 's-', label="NFL", linewidth=2)
    plt.plot(sym_df.index, sym_df["APL-NCE+MAE"], '^-', label="APL-NCE+MAE", linewidth=2)
    plt.plot(sym_df.index, sym_df["APL-NCE+RCE"], 'D-', label="APL-NCE+RCE", linewidth=2)
    plt.plot(sym_df.index, sym_df["APL-NFL+MAE"], 'X-', label="APL-NFL+MAE", linewidth=2)
    plt.plot(sym_df.index, sym_df["APL-NFL+RCE"], 'P-', label="APL-NFL+RCE", linewidth=2)
    plt.title("APL Framework Performance (Symmetric Noise)", fontsize=14)
    plt.xlabel("Noise Rate (η)", fontsize=12)
    plt.ylabel("Best Test Accuracy (%)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/apl_framework_symmetric.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for method in sym_df.columns:
        plt.plot(sym_df.index, sym_df[method], 'o-', label=method, linewidth=2)
    plt.title("Impact of Symmetric Noise Rate on All Methods", fontsize=14)
    plt.xlabel("Noise Rate (η)", fontsize=12)
    plt.ylabel("Best Test Accuracy (%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/all_methods_symmetric.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if asym_results:
        asym_df = pd.DataFrame(asym_results).T
        asym_df.index = [f"{float(rate):.1f}" for rate in asym_df.index]
        asym_df.index.name = "Noise Rate"
        asym_df.to_csv(f"{tables_dir}/asymmetric_noise_summary.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(asym_df.index, asym_df["CE"], 'o-', label="CE", linewidth=2)
        plt.plot(asym_df.index, asym_df["NCE"], 's-', label="NCE", linewidth=2)
        plt.plot(asym_df.index, asym_df["FL"], '^-', label="FL", linewidth=2)
        plt.plot(asym_df.index, asym_df["NFL"], 'D-', label="NFL", linewidth=2)
        plt.title("Normalized vs Vanilla Losses (Asymmetric Noise)", fontsize=14)
        plt.xlabel("Noise Rate (η)", fontsize=12)
        plt.ylabel("Best Test Accuracy (%)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plots_dir}/normalized_vs_vanilla_asymmetric.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(asym_df.index, asym_df["NCE"], 'o-', label="NCE", linewidth=2)
        plt.plot(asym_df.index, asym_df["NFL"], 's-', label="NFL", linewidth=2)
        plt.plot(asym_df.index, asym_df["APL-NCE+MAE"], '^-', label="APL-NCE+MAE", linewidth=2)
        plt.plot(asym_df.index, asym_df["APL-NCE+RCE"], 'D-', label="APL-NCE+RCE", linewidth=2)
        plt.plot(asym_df.index, asym_df["APL-NFL+MAE"], 'X-', label="APL-NFL+MAE", linewidth=2)
        plt.plot(asym_df.index, asym_df["APL-NFL+RCE"], 'P-', label="APL-NFL+RCE", linewidth=2)
        plt.title("APL Framework Performance (Asymmetric Noise)", fontsize=14)
        plt.xlabel("Noise Rate (η)", fontsize=12)
        plt.ylabel("Best Test Accuracy (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plots_dir}/apl_framework_asymmetric.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        for method in asym_df.columns:
            plt.plot(asym_df.index, asym_df[method], 'o-', label=method, linewidth=2)
        plt.title("Impact of Asymmetric Noise Rate on All Methods", fontsize=14)
        plt.xlabel("Noise Rate (η)", fontsize=12)
        plt.ylabel("Best Test Accuracy (%)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/all_methods_asymmetric.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        common_rates = [rate for rate in sym_df.index if rate in asym_df.index]
        if common_rates:
            methods = ["CE", "FL", "NCE", "NFL", "RCE", "APL-NCE+RCE", "APL-NFL+RCE"]
            
            for rate in common_rates:
                plt.figure(figsize=(12, 6))
                
                sym_values = [sym_df.loc[rate, method] for method in methods]
                asym_values = [asym_df.loc[rate, method] for method in methods]
                
                x = np.arange(len(methods))
                width = 0.35
                
                plt.bar(x - width/2, sym_values, width, label=f'Symmetric Noise η={rate}')
                plt.bar(x + width/2, asym_values, width, label=f'Asymmetric Noise η={rate}')
                
                plt.ylabel('Best Test Accuracy (%)', fontsize=12)
                plt.title(f'Symmetric vs Asymmetric Noise (η={rate})', fontsize=14)
                plt.xticks(x, methods, rotation=45, ha='right', fontsize=10)
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.grid(True, axis='y', alpha=0.3)
                plt.savefig(f"{plots_dir}/sym_vs_asym_noise_{rate}.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        if len(common_rates) > 1:
            methods = list(sym_df.columns)
            
            diff_df = pd.DataFrame(index=common_rates, columns=methods)
            for rate in common_rates:
                for method in methods:
                    diff_df.loc[rate, method] = asym_df.loc[rate, method] - sym_df.loc[rate, method]
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(diff_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            plt.title("Performance Difference: Asymmetric - Symmetric Noise", fontsize=14)
            plt.ylabel("Noise Rate (η)", fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/noise_type_difference_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

sym_results = {
0.2: {
"CE": 77.44, "FL": 75.61, "NCE": 63.53, "NFL": 66.28,
"MAE": 65.97, "RCE": 81.65, "APL-NCE+MAE": 79.66,
"APL-NCE+RCE": 81.13, "APL-NFL+MAE": 80.38, "APL-NFL+RCE": 81.89
},
0.4: {
"CE": 71.82, "FL": 72.31, "NCE": 56.73, "NFL": 60.73,
"MAE": 66.91, "RCE": 75.77, "APL-NCE+MAE": 76.15,
"APL-NCE+RCE": 77.13, "APL-NFL+MAE": 76.59, "APL-NFL+RCE": 78.15
},
0.6: {
"CE": 59.93, "FL": 60.45, "NCE": 44.59, "NFL": 53.17,
"MAE": 57.06, "RCE": 52.00, "APL-NCE+MAE": 68.81,
"APL-NCE+RCE": 69.28, "APL-NFL+MAE": 69.99, "APL-NFL+RCE": 67.94
},
0.8: {
"CE": 29.92, "FL": 37.28, "NCE": 36.97, "NFL": 37.89,
"MAE": 32.23, "RCE": 28.47, "APL-NCE+MAE": 47.94,
"APL-NCE+RCE": 29.41, "APL-NFL+MAE": 45.05, "APL-NFL+RCE": 29.76
}
}

asym_results = {
0.1: {
"CE": 82.02, "FL": 81.56, "NCE": 69.38, "NFL": 71.78,
"MAE": 75.96, "RCE": 83.52, "APL-NCE+MAE": 81.78,
"APL-NCE+RCE": 83.45, "APL-NFL+MAE": 81.43, "APL-NFL+RCE": 83.66
},
0.2: {
"CE": 80.59, "FL": 80.47, "NCE": 66.64, "NFL": 70.09,
"MAE": 58.33, "RCE": 73.54, "APL-NCE+MAE": 79.94,
"APL-NCE+RCE": 80.88, "APL-NFL+MAE": 79.78, "APL-NFL+RCE": 80.77
},
0.3: {
"CE": 79.41, "FL": 77.98, "NCE": 62.14, "NFL": 66.27,
"MAE": 57.21, "RCE": 59.10, "APL-NCE+MAE": 77.54,
"APL-NCE+RCE": 78.66, "APL-NFL+MAE": 77.80, "APL-NFL+RCE": 78.14
},
0.4: {
"CE": 75.95, "FL": 74.64, "NCE": 57.00, "NFL": 63.57,
"MAE": 53.74, "RCE": 54.43, "APL-NCE+MAE": 72.91,
"APL-NCE+RCE": 72.36, "APL-NFL+MAE": 71.98, "APL-NFL+RCE": 72.53
}
}

generate_summary_tables_and_plots(sym_results, asym_results)
