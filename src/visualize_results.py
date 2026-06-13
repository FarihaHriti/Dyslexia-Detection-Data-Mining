import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

def generate_comparative_charts():
    # Load results
    if not os.path.exists('comparison_results.json'):
        print("Error: comparison_results.json not found.")
        return
        
    with open('comparison_results.json', 'r') as f:
        results = json.load(f)

    # 1. Bar Chart: Experiment I (ETDD70) - Accuracy and F1-scores
    data_exp1 = []
    for model_name, metrics in results['Exp1'].items():
        data_exp1.append({'Model': model_name, 'Metric': 'Accuracy', 'Value': metrics['accuracy']})
        data_exp1.append({'Model': model_name, 'Metric': 'F1-Score (Control)', 'Value': metrics['report'].get('0', {}).get('f1-score', 0)})
        data_exp1.append({'Model': model_name, 'Metric': 'F1-Score (Dyslexic)', 'Value': metrics['report'].get('1', {}).get('f1-score', 0)})
    df_exp1 = pd.DataFrame(data_exp1)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax1 = sns.barplot(x='Model', y='Value', hue='Metric', data=df_exp1, palette='muted')
    plt.title('Experiment I (Intra-Dataset ETDD70): Performance Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Add labels on top of bars
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 9), 
                         textcoords='offset points',
                         fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multi_model_accuracy.png')
    plt.close()
    print("Saved 'multi_model_accuracy.png' (Experiment I Metrics)")

    # 2. Bar Chart: Experiment II (Kronoberg Sweden) - Accuracy and F1-scores
    data_exp2 = []
    for model_name, metrics in results['Exp2'].items():
        data_exp2.append({'Model': model_name, 'Metric': 'Accuracy', 'Value': metrics['accuracy']})
        data_exp2.append({'Model': model_name, 'Metric': 'F1-Score (Control)', 'Value': metrics['report'].get('0', {}).get('f1-score', 0)})
        data_exp2.append({'Model': model_name, 'Metric': 'F1-Score (Dyslexic)', 'Value': metrics['report'].get('1', {}).get('f1-score', 0)})
    df_exp2 = pd.DataFrame(data_exp2)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax2 = sns.barplot(x='Model', y='Value', hue='Metric', data=df_exp2, palette='pastel')
    plt.title('Experiment II (Cross-Dataset Kronoberg): Performance Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.1)

    # Add labels on top of bars
    for p in ax2.patches:
        if p.get_height() > 0:
            ax2.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 9), 
                         textcoords='offset points',
                         fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('multi_model_f1_score.png')
    plt.close()
    print("Saved 'multi_model_f1_score.png' (Experiment II Metrics)")

    # 3. Grid of Confusion Matrices for Experiment II (The most challenging one)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    model_names = ['SVM', 'Random Forest', 'XGBoost']
    
    for i, name in enumerate(model_names):
        cm = np.array(results['Exp2'][name]['cm'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Control', 'Dyslexic'], 
                    yticklabels=['Control', 'Dyslexic'])
        axes[i].set_title(f'{name} (Exp II)', fontsize=14)
        axes[i].set_xlabel('Predicted', fontsize=12)
        axes[i].set_ylabel('True', fontsize=12)

    plt.suptitle('Confusion Matrices for Experiment II: Cross-Dataset Generalization', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('multi_model_cm_grid.png')
    plt.close()
    print("Saved 'multi_model_cm_grid.png'")

if __name__ == "__main__":
    generate_comparative_charts()
