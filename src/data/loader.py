"""
DataLoader utilities and factories for train/val/test sets.
Handles augmentation and class imbalance strategies.
"""
from typing import Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import ChestXrayDataset


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training augmentation pipeline.
    
    Includes:
    - RandomHorizontalFlip: Chest X-rays are horizontally symmetric
    - RandomRotation: Slight rotation robustness
    - ColorJitter: Brightness/contrast variation
    - Normalization: ImageNet stats
    
    Args:
        image_size: Target image dimension
        
    Returns:
        Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test augmentation pipeline (no random augmentations).
    
    Args:
        image_size: Target image dimension
        
    Returns:
        Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(
    data_dir: str,
    labels_df: pd.DataFrame,
    class_names: list,
    batch_size: int = 32,
    val_batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    split_col: str = "split"  # Column in df that has 'train'/'val'/'test'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from a single DataFrame.
    
    The DataFrame must have a 'split' column indicating which set each row belongs to.
    
    Args:
        data_dir: Directory containing images
        labels_df: DataFrame with image paths, labels, and split column
        class_names: List of class names in order
        batch_size: Batch size for training
        val_batch_size: Batch size for validation/test
        num_workers: Number of workers for data loading
        image_size: Target image size
        split_col: Column name indicating train/val/test split
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data
    train_df = labels_df[labels_df[split_col] == "train"].reset_index(drop=True)
    val_df = labels_df[labels_df[split_col] == "val"].reset_index(drop=True)
    test_df = labels_df[labels_df[split_col] == "test"].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        img_dir=data_dir,
        labels_df=train_df,
        transform=get_train_transforms(image_size),
        class_names=class_names
    )
    
    val_dataset = ChestXrayDataset(
        img_dir=data_dir,
        labels_df=val_df,
        transform=get_val_transforms(image_size),
        class_names=class_names
    )
    
    test_dataset = ChestXrayDataset(
        img_dir=data_dir,
        labels_df=test_df,
        transform=get_val_transforms(image_size),
        class_names=class_names
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Extract class weights from the training dataset.
    
    Args:
        train_loader: Training DataLoader
        
    Returns:
        Tensor of class weights
    """
    return train_loader.dataset.get_class_weights()
