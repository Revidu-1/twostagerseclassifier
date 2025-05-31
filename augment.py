import random
import numpy as np
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import ASTModel, ASTFeatureExtractor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional, Union

class SpecAugment:
    """    
    Reference: Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    def __init__(
        self,
        time_warp_param=80,       # Time warping parameter (W)
        freq_mask_param=27,       # Frequency mask parameter (F)
        time_mask_param=100,      # Time mask parameter (T)
        n_freq_masks=2,           # Number of frequency masks (mF)
        n_time_masks=2,           # Number of time masks (mT)
        apply_time_warp=True,     # Whether to apply time warping
        apply_freq_mask=True,     # Whether to apply frequency masking
        apply_time_mask=True      # Whether to apply time masking
    ):
        self.time_warp_param = time_warp_param
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask
    
    def time_warp(self, spec, W=None):
        """
        Apply time warping to a spectrogram        
       
        """
        if W is None:
            W = self.time_warp_param
            
        # Get spectrogram dimensions
        _, T = spec.shape
        
        # If spectrogram is too short, skip time warping
        if T < W * 2:
            return spec
        
        # Center point for warping
        c = random.randint(W, T - W)
        
        # Distance to warp
        w = random.randint(-W, W)
        
        # Source points for warping
        src_pts = torch.tensor([[0, 0], [0, T - 1], [c, 0]])
        
        # Destination points with warping
        dest_pts = torch.tensor([[0, 0], [0, T - 1], [c + w, 0]])
        
        # Convert to 3D tensors for affine_grid
        src_pts = src_pts.float().unsqueeze(0)
        dest_pts = dest_pts.float().unsqueeze(0)
        
        # Calculate the coefficients of the affine transformation
        theta = F.pad(F.piecewise_rational_quadratic_spline(dest_pts.view(-1), src_pts.view(-1), 1), [0, 0, 0, 3])
        theta = theta.view(1, 2, 3)
        
        # Apply warping using affine grid
        grid = F.affine_grid(theta, spec.unsqueeze(0).unsqueeze(0).size(), align_corners=True)
        warped_spec = F.grid_sample(spec.unsqueeze(0).unsqueeze(0), grid, align_corners=True)
        
        return warped_spec.squeeze()
    
    def freq_mask(self, spec, F=None, num_masks=None):
        
        if F is None:
            F = self.freq_mask_param
            
        if num_masks is None:
            num_masks = self.n_freq_masks
            
        # Getting spectrogram dimensions
        v, _ = spec.shape
        
        # Making a copy of the spectrogram
        masked_spec = spec.clone()
        
        # Applying frequency masks
        for i in range(num_masks):
            # Skip if F is too large for this spectrogram
            if v < F:
                continue
                
            # Frequency mask size
            f = random.randint(0, F)
            
            # Starting frequency
            f0 = random.randint(0, v - f)
            
            # Apply mask
            masked_spec[f0:f0 + f, :] = 0
            
        return masked_spec
    
    def time_mask(self, spec, T=None, num_masks=None):
        """
        Applying time masking to a spectrogram
        
        """
        if T is None:
            T = self.time_mask_param
            
        if num_masks is None:
            num_masks = self.n_time_masks
            
        _, tau = spec.shape
        
        # Making a copy of the spectrogram
        masked_spec = spec.clone()
        
        for i in range(num_masks):
            # Skipping if T is too large for this spectrogram
            if tau < T:
                continue
                
            t = random.randint(0, T)
            
            t0 = random.randint(0, tau - t)
            
            masked_spec[:, t0:t0 + t] = 0
            
        return masked_spec
    
    def __call__(self, spec):
        
        if self.apply_time_warp and random.random() < 0.8:
            spec = self.time_warp(spec)
            
        if self.apply_freq_mask and random.random() < 0.8:
            spec = self.freq_mask(spec)
            
        if self.apply_time_mask and random.random() < 0.8:
            spec = self.time_mask(spec)
            
        return spec


# Alternative implementation with numpy operations due to issues with pytorch that occur randomly
class SpecAugmentNumpy:
    """    
    Reference: Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    
    def __init__(
        self,
        freq_mask_param=27,       # Frequency mask parameter (F)
        time_mask_param=100,      # Time mask parameter (T)
        n_freq_masks=2,           # Number of frequency masks (mF)
        n_time_masks=2,           # Number of time masks (mT)
        apply_freq_mask=True,     # Whether to apply frequency masking
        apply_time_mask=True      # Whether to apply time masking
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask
    
    def freq_mask(self, spec, F=None, num_masks=None):
        
        if F is None:
            F = self.freq_mask_param
            
        if num_masks is None:
            num_masks = self.n_freq_masks
            
        # Getting spectrogram dimensions
        v, _ = spec.shape
        
        # Making a copy of the spectrogram
        masked_spec = spec.copy()
        
        # Applying frequency masks
        for i in range(num_masks):
            if v < F:
                continue
                
            f = np.random.randint(0, F)
            
            f0 = np.random.randint(0, v - f)
            
            masked_spec[f0:f0 + f, :] = 0
            
        return masked_spec
    
    def time_mask(self, spec, T=None, num_masks=None):
        
        if T is None:
            T = self.time_mask_param
            
        if num_masks is None:
            num_masks = self.n_time_masks
            
        _, tau = spec.shape
        
        masked_spec = spec.copy()
        
        for i in range(num_masks):
            if tau < T:
                continue
                
            t = np.random.randint(0, min(T, tau // 2))
            
            t0 = np.random.randint(0, tau - t)
            
            masked_spec[:, t0:t0 + t] = 0
            
        return masked_spec
    
    def __call__(self, spec):
        """
        Apply SpecAugment to a spectrogram
        
        Args:
            spec: Input spectrogram [freq, time] as numpy array
            
        Returns:
            Augmented spectrogram
        """
        is_tensor = isinstance(spec, torch.Tensor)
        if is_tensor:
            spec_np = spec.cpu().numpy()
        else:
            spec_np = spec
            
        if self.apply_freq_mask and np.random.random() < 0.8:
            spec_np = self.freq_mask(spec_np)
            
        if self.apply_time_mask and np.random.random() < 0.8:
            spec_np = self.time_mask(spec_np)
            
        if is_tensor:
            return torch.from_numpy(spec_np).to(spec.device)
        else:
            return spec_np