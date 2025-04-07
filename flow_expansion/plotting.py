import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Plot ground truth samples and network predicted samples for the toy model
def toy_model_plot_comparison_samples_and_prediction(samples: np.ndarray, samples_pred: np.ndarray, save: bool = False, save_path: str = None):
    
    # Configure Matplotlib to use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams.update({'font.size': 14})
    
    #To avoid future warnings
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Ground Truth samples into Dataframe
    theta_1 = pd.Series(samples[:, 0], name="$\\theta_1$")
    theta_2 = pd.Series(samples[:, 1], name="$\\theta_2$")
    samples_gt = pd.concat([theta_1, theta_2], axis=1)

    # Network Predicted Samples into Dataframe
    theta_1_pred = pd.Series(samples_pred[:, 0], name="$\\theta_1$")
    theta_2_pred = pd.Series(samples_pred[:, 1], name="$\\theta_2$")
    samples_pd = pd.concat([theta_1_pred, theta_2_pred], axis=1)

    # Create a joint plot with marginals for the ground truth samples
    g = sns.jointplot(data=samples_gt, x="$\\theta_1$", y="$\\theta_2$", kind="kde", fill=True, levels=3, color='tab:blue', alpha=0.8, marginal_kws={'fill': True, 'color': 'tab:blue', 'alpha': 0.8})
    
    # Overlay predicted samples on the joint plot with dashed lines
    sns.kdeplot(data=samples_pd, x="$\\theta_1$", y="$\\theta_2$", ax=g.ax_joint, fill=False, levels=3, color='tab:orange', linestyle='-', linewidth=2, label='Flow Predicted Samples')
    
    # Marginal plots for predicted samples with dashed lines
    sns.kdeplot(samples_pred[:, 0], ax=g.ax_marg_x, color='tab:orange', linestyle='-', linewidth=2, label='Flow Predicted Samples')
    #sns.kdeplot(samples_pred[:, 1], ax=g.ax_marg_y, color='tab:orange', linestyle='-', linewidth=2, vertical=True)
    sns.kdeplot(y=samples_pred[:, 1], ax=g.ax_marg_y, color='tab:orange', linestyle='-', linewidth=2)

    # Create custom legend
    handles = [
        plt.Line2D([0], [0], color='tab:blue', lw=2, label='training samples', alpha=0.8),
        plt.Line2D([0], [0], color='tab:orange', lw=2, linestyle='-', label='flow predicted samples')
    ]
    g.ax_joint.legend(handles=handles, loc="lower right", title=None)
    
    plt.xlim(left=-3, right = 7)
    plt.ylim(bottom= -3, top = 8.5)
    
    if save:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

    return None