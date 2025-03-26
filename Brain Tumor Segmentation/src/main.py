import os
import json
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from tqdm import tqdm
from monai.data import Dataset, list_data_collate, pad_list_data_collate
from monai.losses import DiceCELoss, DiceLoss
from monai.utils import set_determinism
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from models import UNet3D
from data import prepare_data, get_transforms
from utils import compute_metrics, visualize_sample, visualize_prediction, plot_metrics, plot_loss_curves, generate_table


def train_and_evaluate(model, model_name, train_loader, val_loader, config, device, checkpoint_dir="checkpoints", writer=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pth")
    
    # Loss and optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = CyclicLR(optimizer, 
                         base_lr=config["learning_rate"], 
                         max_lr=config["max_lr"], 
                         step_size_up=config["step_size_up"], 
                         mode='triangular')
    
    best_metric = -1  # Track best WT Dice as primary metric
    best_weights_path = os.path.join(config["output_path"], f"best_{model_name}.pth")
    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        logging.info(f"Loaded checkpoint from epoch {start_epoch - 1}, best WT Dice: {best_metric:.4f}")
    else:
        logging.info(f"No checkpoint found for {model_name}, starting from scratch.")

    # Training loop
    for epoch in tqdm(range(start_epoch, config["max_epochs"]), desc=f"{model_name} Training Epochs", colour="green"):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"{model_name} - Epoch {epoch + 1}/{config['max_epochs']}, Training Loss: {avg_loss:.4f}")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar(f"{model_name}/Training_Loss", avg_loss, epoch)
            writer.add_scalar(f"{model_name}/Learning_Rate", scheduler.get_last_lr()[0], epoch)  
            
        scheduler.step()

        # Validation every `val_interval`
        if (epoch + 1) % config["val_interval"] == 0:
            model.eval()
            val_metrics = {"ET": [], "TC": [], "WT": []}
            with torch.no_grad():
                for val_batch in val_loader:
                    inputs = val_batch["image"].to(device)
                    labels = val_batch["label"].to(device)
                    outputs = model(inputs)
                    pred = torch.argmax(outputs, dim=1).cpu().numpy()  
                    gt = labels.cpu().numpy()[:, 0]  
                    for i in range(pred.shape[0]):
                        batch_metrics = compute_metrics(pred[i], gt[i])
                        for region in val_metrics:
                            val_metrics[region].append(batch_metrics[region]["Dice"])
            
            # Average Dice per region
            avg_val_dice = {region: np.mean(scores) for region, scores in val_metrics.items()}
            logging.info(f"{model_name} - Epoch {epoch + 1}, Validation Dice: ET={avg_val_dice['ET']:.4f}, TC={avg_val_dice['TC']:.4f}, WT={avg_val_dice['WT']:.4f}")
            
            # Log to TensorBoard
            if writer:
                for region, dice in avg_val_dice.items():
                    writer.add_scalar(f"{model_name}/Validation_Dice_{region}", dice, epoch)
                writer.add_scalar(f"{model_name}/Validation_Learning_Rate", scheduler.get_last_lr()[0], epoch)

            # Use WT Dice as primary metric for best model
            val_dice = avg_val_dice["WT"]
            if val_dice > best_metric:
                best_metric = val_dice
                torch.save(model.state_dict(), best_weights_path)
                logging.info(f"New best model saved with WT Dice: {best_metric:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "loss": avg_loss
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    # Load best weights for final evaluation
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path))
        logging.info(f"Loaded best weights for {model_name} with WT Dice: {best_metric:.4f}")
    else:
        logging.warning(f"No best weights found for {model_name}, using final state.")

    # Final evaluation
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc=f"{model_name} Evaluation", colour="blue"):
            inputs = val_batch["image"].to(device)
            labels = val_batch["label"].to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # [D, H, W]
            gt = labels.cpu().numpy()[0, 0]  # [D, H, W]
            metrics = compute_metrics(pred, gt)
            metrics_list.append(metrics)
    
    logging.info(f"Best Validation WT Dice for {model_name}: {best_metric:.4f}")
    return metrics_list, best_metric


def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Setup directories
    os.makedirs(config["output_path"], exist_ok=True)
    checkpoint_dir = os.path.join(config["output_path"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = os.path.join(config["output_path"], "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Configure logging and TensorBoard
    logging.basicConfig(level=logging.INFO, filename=os.path.join(config["output_path"], "training.log"),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Device Agnostic Code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Prepare data
    data_list = prepare_data(config["data_path"])
    logging.info(f"Total training samples: {len(data_list)}")
    
    # Visualize sample
    visualize_sample(data_list, os.path.join(config["output_path"], "sample_visualization.png"))
    logging.info("Sample visualization saved to sample_visualization.png")
    
    # Cross-validation
    kfold = KFold(n_splits=config["k_folds"], shuffle=True, random_state=42)
    results = []
    best_metrics = {}
    
    for fold, (train_ids, val_ids) in enumerate(tqdm(kfold.split(data_list), desc="Cross-Validation Folds", total=config["k_folds"], colour="red")):
        logging.info(f"Fold {fold + 1}/{config['k_folds']}")
        train_subset = Subset(Dataset(data=data_list, transform=get_transforms(config, train=True)), train_ids)
        val_subset = Subset(Dataset(data=data_list, transform=get_transforms(config, train=False)), val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=pad_list_data_collate)
        val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=pad_list_data_collate)
        
        model = UNet3D(channels=tuple(config["channels"])).to(device)
        fold_results, fold_best_metric = train_and_evaluate(model, f"UNet3D_fold{fold}", train_loader, val_loader, config, device, checkpoint_dir, writer)
        results.extend(fold_results)
        best_metrics[f"fold_{fold}"] = fold_best_metric
        visualize_prediction(model, val_loader, f"UNet3D_fold{fold}", os.path.join(config["output_path"], f"prediction_fold{fold}.png"), config, device)
    
    # Post-training analysis
    plot_metrics(results, "UNet3D", config["output_path"])
    table = generate_table(results)
    table.to_csv(os.path.join(config["output_path"], "results_table.csv"), index=False)
    logging.info("Results table saved to results_table.csv")
    
    # Report best metrics
    avg_best_metric = sum(best_metrics.values()) / len(best_metrics)
    logging.info(f"Best WT Dice Scores per Fold: {best_metrics}")
    logging.info(f"Average Best WT Dice Score across Folds: {avg_best_metric:.4f}")
    
    writer.close()
    logging.info("Training and evaluation completed! TensorBoard logs saved in tensorboard/ directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation")
    parser.add_argument("--config", type=str, default="config/hyperparams.json", help="Path to config JSON file")
    args = parser.parse_args()
    main(args.config)