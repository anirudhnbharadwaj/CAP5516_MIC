import os
import json
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld,
    RandRotated, RandFlipd, ToTensord, EnsureTyped, RandGaussianNoised, RandAdjustContrastd, Lambdad, Resized
)

def prepare_data(data_path):
    with open(os.path.join(data_path, "dataset.json"), "r") as f:
        dataset_info = json.load(f)
    
    data_list = []
    for item in dataset_info["training"]:
        image_path = os.path.join(data_path, item["image"].replace("./", ""))
        label_path = os.path.join(data_path, item["label"].replace("./", ""))
        data_list.append({"image": image_path, "label": label_path})
    return data_list

# def remap_labels(seg):
#     seg_mapped = seg.clone()
#     seg_mapped[seg == 4] = 3
#     return seg_mapped

def get_transforms(config, train=True):
    keys = ["image", "label"]
    if train:
        return Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),  
            ScaleIntensityd(keys=["image"]),
            # Lambdad(keys=["label"], func=remap_labels),
            RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=config["spatial_size"],
                                   pos=1, neg=1, num_samples=config["num_samples"]),
            RandRotated(keys=keys, range_x=config["rotation_range"], prob=config["flip_prob"]),
            RandFlipd(keys=keys, spatial_axis=0, prob=config["flip_prob"]),
            RandGaussianNoised(keys=["image"], prob=config["noise_prob"], std=0.01),
            RandAdjustContrastd(keys=["image"], prob=config["contrast_prob"], gamma=config["contrast_gamma"]),
            EnsureTyped(keys=keys),
            Resized(keys=["image", "label"], spatial_size=config["spatial_size"]),
            ToTensord(keys=keys)
        ])
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),  
        ScaleIntensityd(keys=["image"]),
        # Lambdad(keys=["label"], func=remap_labels),
        EnsureTyped(keys=keys),
        Resized(keys=["image", "label"], spatial_size=config["spatial_size"]),
        ToTensord(keys=keys)
    ])
    
    
    
## Check the Unique Labels in the dataset - debugging
# import nibabel as nib
# import numpy as np
# lbl = nib.load(rf"C:\Users\aniru\Desktop\UCF\Coursework\CAP5516 - MIC\Assignments\A2\Task01\Task01_BrainTumour\labelsTr\BRATS_001.nii.gz").get_fdata()
# print(np.unique(lbl))  