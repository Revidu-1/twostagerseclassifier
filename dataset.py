import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from transformers import  ASTFeatureExtractor
from typing import Dict
from augment import SpecAugmentNumpy
from config import Config

class ICBHIASTDataset(Dataset):
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: ASTFeatureExtractor,
        transform: bool = False,
        max_length: int = Config.MAX_LENGTH,
        sampling_rate: int = Config.SAMPLING_RATE,
        stage: str = 'all',  # 'all', 'binary', or 'abnormal'
        apply_specaugment: bool = True,  # Whether to apply SpecAugment
        specaugment_params: dict = None  # SpecAugment parameters
    ):
       
        self.df = df
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.stage = stage
        self.apply_specaugment = apply_specaugment and transform  # Only apply if transform is True
        
        # Initialising SpecAugment
        if self.apply_specaugment:
            if specaugment_params is None:
                specaugment_params = {}
            # Use the NumPy implementation as it's more stable
            self.specaugment = SpecAugmentNumpy(**specaugment_params)
        
        # Filtering dataset for abnormal stage if needed
        if stage == 'abnormal':
            self.df = self.df[self.df['label'] != 'normal'].reset_index(drop=True)
        
        self.all_class_map = Config.CLASSES
        self.binary_class_map = Config.BINARY_CLASSES
        self.abnormal_class_map = Config.ABNORMAL_CLASSES
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.df)
    
    def _extract_segment(self, waveform, sample_rate, start_time, end_time):
      
        # Converting time to samples and ensuring valid range
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)
        
        if start_sample >= end_sample:
            return torch.zeros(1, 1)
            
        segment = waveform[:, start_sample:end_sample]
        return segment
    
    def _load_audio_segment(self, file_path, start_time, end_time):
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)            
            # Converting to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extracting segment
            segment = self._extract_segment(waveform, sample_rate, start_time, end_time)
            
            if segment.numel() <= 1:
                return torch.zeros(1, self.max_length), sample_rate
            
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                segment = resampler(segment)
                sample_rate = self.sampling_rate
            
            # Ensuring proper length
            if segment.shape[1] > self.max_length:
                start = (segment.shape[1] - self.max_length) // 2
                segment = segment[:, start:start + self.max_length]
            elif segment.shape[1] < self.max_length:
                # Padding with zeros if too short
                padding = torch.zeros(1, self.max_length - segment.shape[1])
                segment = torch.cat([segment, padding], dim=1)
            
            return segment, sample_rate
            
        except Exception as e:
            print(f"Error loading audio segment {file_path} ({start_time}-{end_time}s): {e}")
            # Return zeros as fallback
            return torch.zeros(1, self.max_length), self.sampling_rate
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to audio
        
    
        """
        # Skipping augmentation randomly with 50% chance
        if not self.transform or torch.rand(1).item() < 0.2:
            return waveform
        
        # Choosing an augmentation method
        aug_type = torch.randint(0, 5, (1,)).item()
        
        if aug_type == 0:
            # Adding background noise
            noise_level = 0.005 * torch.rand(1).item()
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
            
        elif aug_type == 1:
            # Time shifting
            shift_amount = int(waveform.shape[1] * 0.2 * (torch.rand(1).item() * 2 - 1))
            if shift_amount > 0:
                waveform = torch.cat([waveform[:, -shift_amount:], waveform[:, :-shift_amount]], dim=1)
            elif shift_amount < 0:
                shift_amount = abs(shift_amount)
                waveform = torch.cat([waveform[:, shift_amount:], waveform[:, :shift_amount]], dim=1)
                
        elif aug_type == 2:
            # Pitch shifting
            stretch_factor = 0.85 + 0.3 * torch.rand(1).item()  # Between 0.85 and 1.15
            if stretch_factor != 1.0:
                orig_len = waveform.shape[1]
                stretched_len = int(orig_len / stretch_factor)
                if stretched_len > orig_len:
                    # Stretching by sampling with interpolation
                    indices = torch.linspace(0, orig_len - 1, stretched_len)
                    indices = indices.clamp(0, orig_len - 1).long()
                    waveform = waveform[:, indices]
                    # Trimming waveform to original length
                    waveform = waveform[:, :orig_len]
                else:
                    # Compressing by sampling with interpolation
                    indices = torch.linspace(0, orig_len - 1, stretched_len)
                    indices = indices.clamp(0, orig_len - 1).long()
                    waveform_stretched = waveform[:, indices]
                    # Padding waveform to original length
                    waveform = F.pad(waveform_stretched, (0, orig_len - stretched_len), "constant", 0)
        
        elif aug_type == 3:
            # Volume adjustment
            volume_factor = 0.7 + 0.6 * torch.rand(1).item()  # Between 0.7 and 1.3
            waveform = waveform * volume_factor
            
        elif aug_type == 4:
            # Time masking (randomly zero out segments)
            mask_size = int(waveform.shape[1] * 0.1 * torch.rand(1).item())
            mask_start = torch.randint(0, waveform.shape[1] - mask_size, (1,)).item()
            waveform[:, mask_start:mask_start + mask_size] = 0
        
        # Normalising waveform
        if torch.abs(waveform).max() > 0:
            waveform = waveform / torch.abs(waveform).max()
            
        return waveform
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Getting a dataset item       
       
        """
        # Getting item info
        item = self.df.iloc[idx]
        file_path = item['file_path']
        start_time = item['start_time']
        end_time = item['end_time']
        label_str = item['label']
        
        if self.stage == 'binary':
            # Binary classification (normal vs abnormal)
            label = 0 if label_str == 'normal' else 1
        elif self.stage == 'abnormal':
            # Abnormal classification (crackle, wheeze, both)
            label = self.abnormal_class_map.get(label_str, 0)
        else:
            # Default: all classes
            label = self.all_class_map.get(label_str.lower(), 0)
        
        # Loading and preprocessing audio segment
        waveform, sample_rate = self._load_audio_segment(file_path, start_time, end_time)
        
        # Applying audio augmentation if enabled
        if self.transform:
            waveform = self._augment_audio(waveform)
        
        waveform_np = waveform.squeeze().numpy()  
        
        try:
            # Processing with AST feature extractor
            inputs = self.feature_extractor(
                waveform_np, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.squeeze(0)
            
            if self.apply_specaugment and self.transform and torch.rand(1).item() < 0.8:
                # SpecAugment expects input shape [freq, time]
                # AST input_values shape is typically [time, freq]
                # Therefore, transposing for SpecAugment, then transposing back
                input_values = input_values.transpose(0, 1)  # [freq, time]
                input_values = self.specaugment(input_values)
                input_values = input_values.transpose(0, 1)  # [time, freq]
                
        except Exception as e:
            print(f"Error processing audio with feature extractor: {e}")
            input_shape = (498, 128)  
            input_values = torch.zeros(input_shape)
        
        # Returning dictionary
        return {
            'input_values': input_values,
            'label': torch.tensor(label, dtype=torch.long),
            'file_path': file_path,
            'segment': f"{start_time:.2f}-{end_time:.2f}"
        }

    def create_balanced_sampler(self):
        """
        Creates a weighted sampler for balanced training
        
        
        """
        if self.stage == 'binary':
            # Mapping labels for binary classification
            labels = [0 if l == 'normal' else 1 for l in self.df['label']]
        elif self.stage == 'abnormal':
            # Mapping labels for abnormal classification
            labels = [self.abnormal_class_map.get(l, 0) for l in self.df['label']]
        else:
            # Falling back to original labels
            labels = [self.all_class_map.get(l.lower(), 0) for l in self.df['label']]
        
        label_counts = {}
        for label in labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        # Calculating weights (inverse frequency)
        weights = []
        for label in labels:
            weight = 1.0 / label_counts[label]
            weights.append(weight)
        
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )


def ast_collate_fn(batch):
    "Function to handle variable-sized inputs"
    
    input_values = [item['input_values'] for item in batch]
    labels = [item['label'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    segments = [item['segment'] for item in batch]
    
    labels = torch.stack(labels)
    
    # Processing input_values 
    # For AST, the expected shape is [batch_size, sequence_length, feature_dim] or [batch_size, sequence_length]
    
    if len(input_values[0].shape) == 2:
        
        max_len = max([x.shape[0] for x in input_values])
        feature_dim = input_values[0].shape[1]
        
        # Padding all inputs to max length
        padded_inputs = []
        for x in input_values:
            if x.shape[0] < max_len:
                padding = torch.zeros((max_len - x.shape[0], feature_dim), dtype=x.dtype)
                padded_x = torch.cat([x, padding], dim=0)
            else:
                padded_x = x
            padded_inputs.append(padded_x)
        
        # Stacking into a batch
        batched_inputs = torch.stack(padded_inputs)
    else:
        max_len = max([x.shape[0] for x in input_values])
        
        padded_inputs = []
        for x in input_values:
            if x.shape[0] < max_len:
                padding = torch.zeros(max_len - x.shape[0], dtype=x.dtype)
                padded_x = torch.cat([x, padding], dim=0)
            else:
                padded_x = x
            padded_inputs.append(padded_x)
        
        batched_inputs = torch.stack(padded_inputs)
    
    return {
        'input_values': batched_inputs,
        'label': labels,
        'file_path': file_paths,
        'segment': segments
    }
