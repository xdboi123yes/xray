import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def load_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def find_optimal_theta(df_sweep):
    """
    Determine the optimal theta based on available metrics.
    Prefers F1 score, falls back to Youden's J statistic equivalent.
    """
    if 'f1' in df_sweep.columns:
        optimal_idx = df_sweep['f1'].idxmax()
    elif 'sensitivity' in df_sweep.columns and 'specificity' in df_sweep.columns:
        optimal_idx = (df_sweep['sensitivity'] + df_sweep['specificity']).idxmax()
    else:
        # Default fallback
        optimal_idx = len(df_sweep) // 2
    return df_sweep.loc[optimal_idx, 'theta']

def plot_threshold_sweep(df, output_path):
    optimal_theta = find_optimal_theta(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Optimization Trade-offs', fontsize=16, y=1.02)
    
    # Top-Left: Accuracy
    axes[0, 0].plot(df['theta'], df['accuracy'], marker='o', color='b')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    
    # Top-Right: % Escalated to Tier 2
    if 'percent_tier2' in df.columns:
        axes[0, 1].plot(df['theta'], df['percent_tier2'], marker='s', color='orange')
        axes[0, 1].set_title('% Escalated to Tier 2 vs Threshold')
        axes[0, 1].set_ylabel('% Images')
    
    # Bottom-Left: FPS
    if 'estimated_fps' in df.columns:
        axes[1, 0].plot(df['theta'], df['estimated_fps'], marker='^', color='g')
        axes[1, 0].set_title('System FPS vs Threshold')
        axes[1, 0].set_ylabel('FPS')
        
    # Bottom-Right: Sensitivity & Specificity
    if 'sensitivity' in df.columns and 'specificity' in df.columns:
        axes[1, 1].plot(df['theta'], df['sensitivity'], marker='d', label='Sensitivity', color='purple')
        axes[1, 1].plot(df['theta'], df['specificity'], marker='v', label='Specificity', color='brown')
        axes[1, 1].set_title('Sensitivity & Specificity vs Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
    
    # Add vertical lines and formatting for all subplots
    for ax in axes.flat:
        ax.axvline(x=optimal_theta, color='red', linestyle='--', label=f'Optimal θ={optimal_theta:.2f}')
        ax.set_xlabel('Threshold (θ)')
        ax.grid(True, linestyle=':', alpha=0.7)
        if ax != axes[1, 1]:  # Legend already added for the bottom right
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def plot_confidence_distribution(df, output_path):
    plt.figure(figsize=(10, 6))
    
    # Convert tier_used to categorical for better legend
    df['tier_used'] = df['tier_used'].astype(str)
    
    sns.histplot(data=df, x='confidence', hue='tier_used', bins=30, multiple="stack", palette="Set2")
    
    plt.title('Confidence Distribution by Tier', fontsize=14)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def plot_roc_curve(df, output_path):
    if 'ground_truth' not in df.columns or 'confidence' not in df.columns:
        print("Missing required columns for ROC curve.")
        return
        
    y_true = df['ground_truth']
    y_probs = df['confidence']
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Find Youden's J statistic optimal point
    J = tpr - fpr
    best_idx = J.argmax()
    best_threshold = thresholds[best_idx]
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr[best_idx], tpr[best_idx], marker='o', markersize=8, color='red', 
             label=f'Optimal Threshold = {best_threshold:.2f}')
             
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def main():
    target_dir = 'thesis/figures'
    os.makedirs(target_dir, exist_ok=True)
    
    print("Generating high-DPI thesis figures...")
    
    # 1. Threshold Sweep 4-Panel
    sweep_df = load_data('outputs/results/threshold_sweep.csv')
    if sweep_df is not None and not sweep_df.empty:
        plot_threshold_sweep(sweep_df, os.path.join(target_dir, 'threshold_sweep_4panel.png'))
    else:
        print("Skipped threshold_sweep_4panel.png (outputs/results/threshold_sweep.csv not found).")
        
    # 2. Tier Confidence Distribution
    transition_df = load_data('outputs/results/tier_transition_log.csv')
    if transition_df is not None and not transition_df.empty:
        plot_confidence_distribution(transition_df, os.path.join(target_dir, 'tier_confidence_distribution.png'))
    else:
        print("Skipped tier_confidence_distribution.png (outputs/results/tier_transition_log.csv not found).")
        
    # 3. ROC Curve
    val_preds_df = load_data('outputs/results/val_predictions.csv')
    if val_preds_df is not None and not val_preds_df.empty:
        plot_roc_curve(val_preds_df, os.path.join(target_dir, 'roc_curve.png'))
    else:
        print("Skipped roc_curve.png (outputs/results/val_predictions.csv not found).")
        
    print("Done.")

if __name__ == '__main__':
    main()
