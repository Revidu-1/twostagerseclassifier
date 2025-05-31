class Config:
    """Configuration for ICBHI dataset classifier with AST"""
    # Audio processing
    SAMPLING_RATE = 16000
    MAX_LENGTH = 16000 * 5  # 5 seconds at 16kHz
    
    # All classes
    CLASSES = {
        'normal': 0,
        'crackle': 1,
        'wheeze': 2,
        'both': 3  # Both crackle and wheeze
    }
    
    # Binary classes (stage 1)
    BINARY_CLASSES = {
        'normal': 0,
        'abnormal': 1
    }
    
    # Abnormal classes (stage 2)
    ABNORMAL_CLASSES = {
        'crackle': 0,
        'wheeze': 1,
        'both': 2
    }
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 9
    
    # Model parameters
    PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
    FREEZE_FEATURE_EXTRACTOR = True
    DROPOUT_RATE = 0.4
    
    # SpecAugment parameters
    SPECAUGMENT = {
        'enabled': True,
        'freq_mask_param': 20,       # Frequency mask parameter (F)
        'time_mask_param': 50,       # Time mask parameter (T)
        'n_freq_masks': 2,           # Number of frequency masks (mF)
        'n_time_masks': 2,           # Number of time masks (mT)
    }
    
    # Paths
    OUTPUT_DIR = "./ast_two_stage_icbhi_model"
