"""
Visualization script for the TienetReportGenerator Model.
Creates visualizations for:
1. Training/Validation Loss Curves
2. Evaluation Metrics (BLEU/ROUGE)
"""

import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

from config import Config

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Visualizer:
    
    def __init__(self, config):
        self.config = config
        
        os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)
        print(f"Visualizations will be saved to: {config.VISUALIZATIONS_DIR}")
    
    def plot_training_curves(self):
        history_path = self.config.TRAINING_HISTORY_FILE
        
        if not os.path.exists(history_path):
            print(f"--- ⚠️ Warning: Training history not found ---")
            print(f"File not found: {history_path}")
            print("Run 'mediassist_train.py' to generate this file.")
            return
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            
            if not train_losses or not val_losses:
                print(f"Warning: History file {history_path} is empty or missing data.")
                return

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(train_losses, label='Train Loss', linewidth=2)
            ax.plot(val_losses, label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training curves saved to {save_path}")
            plt.close()
        except Exception as e:
            print(f"🚨 Error plotting training curves: {e}")

    
    def plot_metrics(self):
        metrics_path = self.config.METRICS_FILE
        
        if not os.path.exists(metrics_path):
            print(f"--- ⚠️ Warning: Metrics file not found ---")
            print(f"File not found: {metrics_path}")
            print("Run 'mediassist_calculate_metrics.py' to generate this file.")
            return
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            bleu_metrics = {k: v for k, v in metrics.items() if 'BLEU' in k}
            rouge_metrics = {k: v for k, v in metrics.items() if 'ROUGE' in k}
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # BLEU scores
            if bleu_metrics:
                bleu_names = list(bleu_metrics.keys())
                bleu_values = list(bleu_metrics.values())
                axes[0].bar(bleu_names, bleu_values, color='skyblue', edgecolor='navy', linewidth=1.5)
                axes[0].set_title('BLEU Scores', fontsize=14, fontweight='bold')
                axes[0].set_ylabel('Score', fontsize=12)
                max_bleu = max(bleu_values) if bleu_values else 0.1
                axes[0].set_ylim([0, max_bleu * 1.2])
                axes[0].grid(True, alpha=0.3, axis='y')
            
            # ROUGE scores
            if rouge_metrics:
                rouge_names = list(rouge_metrics.keys())
                rouge_values = list(rouge_metrics.values())
                axes[1].bar(rouge_names, rouge_values, color='lightcoral', edgecolor='darkred', linewidth=1.5)
                axes[1].set_title('ROUGE Scores (F1)', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('Score', fontsize=12)
                max_rouge = max(rouge_values) if rouge_values else 0.1
                axes[1].set_ylim([0, max_rouge * 1.2])
                axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'metrics.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Metrics plot saved to {save_path}")
            plt.close()
        except Exception as e:
            print(f"🚨 Error plotting metrics: {e}")
    
    def create_all_visualizations(self):
        
        print("Creating visualizations...")
        
        self.plot_training_curves()
        
        self.plot_metrics()
        
        print("🎉 All visualizations created! 🎉")

def main():
    cfg = Config()
    visualizer = Visualizer(cfg)
    
    visualizer.create_all_visualizations()
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()