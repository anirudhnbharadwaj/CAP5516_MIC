import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from medpy.metric.binary import hd95
import logging
import os
import nibabel as nib

sns.set_style("darkgrid")


def compute_metrics(pred, gt):
    regions = {
        "ET": ([3], [3]),               # Enhancing Tumor: label 3
        "TC": ([2, 3], [2, 3]),         # Tumor Core: labels 2 (non-enhancing) + 3 (enhancing)
        "WT": ([1, 2, 3], [1, 2, 3])    # Whole Tumor: labels 1 (edema) + 2 (non-enhancing) + 3 (enhancing)
    }
    metrics = {}
    for region_name, (pred_labels, gt_labels) in regions.items():
        # Binary masks
        pred_mask = np.isin(pred, pred_labels).astype(np.uint8)
        gt_mask = np.isin(gt, gt_labels).astype(np.uint8)
        
        # Dice Score
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice = dice_metric(torch.tensor(pred_mask[None, None]), torch.tensor(gt_mask[None, None])).item()
        if not np.any(gt_mask):  # Handle empty GT case
            dice = 1.0 if not np.any(pred_mask) else 0.0
        
        # HD95
        hd_95 = hd95(pred_mask, gt_mask) if np.any(pred_mask) and np.any(gt_mask) else 0.0
        
        metrics[region_name] = {"Dice": dice, "HD95": hd_95}
    return metrics


def visualize_sample(data_list, save_path):
    sample = data_list[331]  # Fixed sample for consistency
    slice_idx = 77
    img = nib.load(sample["image"]).get_fdata()     # Shape: (240, 240, 155, 4)
    lbl = nib.load(sample["label"]).get_fdata()     # Shape: (240, 240, 155)
    slice_data = img[:, :, slice_idx, :]            # Shape: (240, 240, 4)
    label_slice = lbl[:, :, slice_idx]              # Shape: (240, 240)
    label_cmap = sns.color_palette("rocket", n_colors=4)
    
    logging.info(f"Image shape: {img.shape}, GT shape: {lbl.shape}, Slice idx: {slice_idx}")
    
    modalities = ["FLAIR", "T1w", "T1gd", "T2w"]
    label_names = ["Background", "Edema", "Non-enhancing Tumor", "Enhancing Tumor"] 
    
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i, modality in enumerate(modalities):
        sns.heatmap(slice_data[:, :, i], ax=axs[i], cmap="viridis", cbar=True)
        axs[i].set_title(f"{modality} Slice")
        axs[i].axis("off")
    sns.heatmap(label_slice, ax=axs[4], cmap=label_cmap, cbar=False)
    for l, c in zip(range(4), label_cmap):
        axs[4].plot([], [], color=c, label=label_names[l])
    axs[4].legend(loc="upper right")
    axs[4].set_title("Ground Truth Mask")
    axs[4].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_prediction(model, loader, model_name, save_path, config, device):
    if not loader.dataset:
        logging.error("Dataset is empty in visualize_prediction.")
        return
    label_cmap = sns.color_palette("rocket", n_colors=4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    model.eval()
    with torch.no_grad():
        sample = loader.dataset[0]
        inputs = sample["image"].unsqueeze(0).to(device)  # [1, C, H, W, D]
        labels = sample["label"].unsqueeze(0).to(device)  # [1, 1, H, W, D]
        outputs = sliding_window_inference(inputs, config["spatial_size"], 4, model)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # [H, W, D]
        gt = labels.cpu().numpy()[0, 0]  # [H, W, D]
        slice_idx = 77
        flair = inputs.cpu().numpy()[0, 0, :, :, slice_idx]  
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sns.heatmap(flair, ax=axes[0], cmap="viridis", cbar=True, square=True)
        axes[0].set_title("FLAIR Image")
        sns.heatmap(gt[:, :, slice_idx], ax=axes[1], cmap=label_cmap, cbar=True, square=True)
        axes[1].set_title("Ground Truth")
        sns.heatmap(pred[:, :, slice_idx], ax=axes[2], cmap=label_cmap, cbar=True, square=True)
        axes[2].set_title("Prediction")
        
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"{model_name} Prediction vs Ground Truth", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Prediction visualization saved to {save_path}")


def plot_metrics(metrics_list, model_name, save_path):
    if not metrics_list:
        logging.error("metrics_list is empty in plot_metrics.")
        return
    
    os.makedirs(save_path, exist_ok=True) 
    dice_data = {region: [m[region]["Dice"] for m in metrics_list] for region in ["ET", "TC", "WT"]}
    hd_data = {region: [m[region]["HD95"] for m in metrics_list] for region in ["ET", "TC", "WT"]}
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(dice_data), palette="rocket")
    plt.title(f"{model_name} - Dice Scores", fontsize=16)
    plt.ylabel("Dice Coefficient", fontsize=12)
    plt.xlabel("Region", fontsize=12)
    plt.savefig(os.path.join(save_path, f"dice_{model_name}.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(hd_data), palette="rocket")
    plt.title(f"{model_name} - Hausdorff Distance (95%)", fontsize=16)
    plt.ylabel("HD95 (voxels)", fontsize=12)
    plt.xlabel("Region", fontsize=12)
    plt.savefig(os.path.join(save_path, f"hd95_{model_name}.png"))
    plt.close()
    logging.info(f"Metrics plots saved to {save_path}/dice_{model_name}.png and {save_path}/hd95_{model_name}.png")
    
## Added this function later on [Not Implemented in the main loop]
def plot_loss_curves(loss_history, model_name, save_path):
    if not loss_history or not isinstance(loss_history, list) or not all(isinstance(l, list) for l in loss_history):
        logging.error("loss_history is empty or invalid in plot_loss_curves.")
        return
    
    os.makedirs(save_path, exist_ok=True)  
    num_folds = len(loss_history)
    epochs = range(1, len(loss_history[0]) + 1) if loss_history else []
    
    plt.figure(figsize=(12, 6))
    for fold in range(num_folds):
        plt.plot(epochs, loss_history[fold], label=f"Fold {fold + 1}", marker='o')
    plt.title(f"{model_name} - Training Loss Across Folds", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"loss_{model_name}.png"))
    plt.close()
    logging.info(f"Loss curve plot saved to {save_path}/loss_{model_name}.png")


def generate_table(metrics_list):
    table = []
    for fold, metrics in enumerate(metrics_list):
        row = [fold + 1]
        for region in ["ET", "TC", "WT"]:
            row.extend([metrics[region]["Dice"], metrics[region]["HD95"]])
        table.append(row)
    avg_row = ["Average"]
    for region in ["ET", "TC", "WT"]:
        dice_values = [m[region]["Dice"] for m in metrics_list]
        hd_values = [m[region]["HD95"] for m in metrics_list if m[region]["HD95"] != np.inf]
        avg_dice = np.mean(dice_values) if dice_values else 0.0
        avg_hd = np.mean(hd_values) if hd_values else 0.0
        avg_row.extend([avg_dice, avg_hd])
    table.append(avg_row)
    return pd.DataFrame(table, columns=["Fold", "ET_Dice", "ET_HD95", "TC_Dice", "TC_HD95", "WT_Dice", "WT_HD95"])