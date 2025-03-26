import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from monai.data import Dataset, list_data_collate, pad_list_data_collate
from monai.losses import DiceLoss
from monai.utils import set_determinism
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm
from models import UNet3D
from data import prepare_data, get_transforms
from utils import compute_metrics  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_and_evaluate(model, train_loader, val_loader, config, device, trial):
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_metric = -1  # Track best WT Dice
    
    for epoch in tqdm(range(config["max_epochs"]), desc=f"Trial {trial.number} Training Epochs", colour="green"):
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
        logging.info(f"Trial {trial.number}, Epoch {epoch + 1}/{config['max_epochs']}, Loss: {avg_loss:.4f}")

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
            logging.info(f"Trial {trial.number}, Epoch {epoch + 1}, Validation Dice: ET={avg_val_dice['ET']:.4f}, TC={avg_val_dice['TC']:.4f}, WT={avg_val_dice['WT']:.4f}")
            
            # Use WT Dice as the optimization metric
            val_dice = avg_val_dice["WT"]
            if val_dice > best_metric:
                best_metric = val_dice
    
    return best_metric


def objective(trial, data_list, config_base, device):
    config = config_base.copy()
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    config["batch_size"] = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32])  
    channels_str = trial.suggest_categorical("channels", [
        "16-32-64-128-256",
        "32-64-128-256-512",
        "8-16-32-64-128"
    ])
    config["channels"] = tuple(map(int, channels_str.split("-")))
    config["max_epochs"] = trial.suggest_int("max_epochs", 5, 50, step=5)
    config["val_interval"] = trial.suggest_int("val_interval", 1, 5)
    
    kfold = KFold(n_splits=config["k_folds"], shuffle=True, random_state=42)
    fold_dice_scores = []
    
    for fold, (train_ids, val_ids) in enumerate(tqdm(kfold.split(data_list), desc=f"Trial {trial.number} Folds", total=config["k_folds"], colour="blue")):
        logging.info(f"Trial {trial.number}, Fold {fold + 1}/{config['k_folds']}")
        train_subset = Subset(Dataset(data=data_list, transform=get_transforms(config, train=True)), train_ids)
        val_subset = Subset(Dataset(data=data_list, transform=get_transforms(config, train=False)), val_ids)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            collate_fn=pad_list_data_collate
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            collate_fn=pad_list_data_collate
        )
        
        model = UNet3D(channels=config["channels"]).to(device)
        fold_dice = train_and_evaluate(model, train_loader, val_loader, config, device, trial)
        fold_dice_scores.append(fold_dice)
        
        trial.report(fold_dice, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    avg_dice = sum(fold_dice_scores) / len(fold_dice_scores)
    return avg_dice


def run_optuna(config_path, n_trials=20):
    with open(config_path, "r") as f:
        config_base = json.load(f)
    
    os.makedirs(config_base["output_path"], exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(config_base["output_path"], "optuna.log")))
    
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    data_list = prepare_data(config_base["data_path"])
    logging.info(f"Total training samples: {len(data_list)}")
    
    try:
        study = optuna.create_study(direction="maximize", study_name="BrainTumorSegmentation", sampler=optuna.samplers.TPESampler())
    except Exception as e:
        logging.error(f"Failed to create Optuna study: {str(e)}")
        raise
    
    for trial in tqdm(range(n_trials), desc="Optuna Trials", colour="yellow"):
        study.optimize(lambda trial: objective(trial, data_list, config_base, device), n_trials=1)
    
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best average WT Dice score: {study.best_value:.4f}")
    logging.info(f"Best hyperparameters: {study.best_params}")
    
    # Prepare best config for hyperparams.json
    best_config = config_base.copy()
    best_config.update(study.best_params)
    best_config["channels"] = list(best_config["channels"])  
    with open(os.path.join(config_base["output_path"], "best_hyperparams.json"), "w") as f:
        json.dump(best_config, f, indent=4)
    logging.info("Best hyperparameters saved to best_hyperparams.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna for Brain Tumor Segmentation")
    parser.add_argument("--config", type=str, default="config/hyperparams.json", help="Path to base config JSON file")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()
    run_optuna(args.config, args.n_trials)