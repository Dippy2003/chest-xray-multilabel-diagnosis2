"""
Custom PyTorch Dataset for multi-label chest X-ray classification.
"""
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class ChestXrayDataset(Dataset):
    """
    Custom Dataset for multi-label chest X-ray images.
    
    Handles multi-label format where each image can have multiple disease labels.
    Uses BCEWithLogitsLoss, so targets are expected as float tensors with values in [0, 1].
    
    Args:
        img_dir: Directory containing image files. Joined with each row's
            'image_path' (a filename, not an absolute path) to locate the file
            on disk — keeps the splits CSV portable across machines.
        labels_df: DataFrame with columns ['image_path', 'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia']
        transform: Optional image transformation pipeline
        class_names: List of class names (order matters for label encoding)
    """
    
    def __init__(
        self,
        img_dir: str,
        labels_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        class_names: List[str] = None,
    ):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        self.class_names = class_names or ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia"]
        
        # Verify all required columns exist
        required_cols = ["image_path"] + self.class_names
        missing_cols = [col for col in required_cols if col not in labels_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and labels by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image tensor, labels tensor)
        """
        row = self.labels_df.iloc[idx]

        # Load image
        img_path = Path(self.img_dir) / row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Could not load image at {img_path}: {str(e)}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Extract labels for each class
        labels = torch.tensor(
            [row[class_name] for class_name in self.class_names],
            dtype=torch.float32
        )
        
        return image, labels
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the number of positive samples per class.
        
        Returns:
            Dictionary with class names and positive sample counts
        """
        distribution = {}
        for class_name in self.class_names:
            distribution[class_name] = int(self.labels_df[class_name].sum())
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.
        Weight = 1 / (class_frequency / total_samples)
        This makes rare classes have higher weight.
        
        Returns:
            Tensor of shape (num_classes,) with normalized weights
        """
        distribution = self.get_class_distribution()
        total_samples = len(self)
        weights = []
        
        for class_name in self.class_names:
            # Frequency of positive samples for this class
            pos_count = distribution[class_name]
            # Weight inversely proportional to frequency
            # Add 1 to avoid division by zero
            weight = total_samples / (2 * (pos_count + 1))
            weights.append(weight)
        
        # Normalize weights to sum to number of classes
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum() * len(self.class_names)
        
        return weights
