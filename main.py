
import argparse
import os
import os
import torch
import pandas as pd
from transformers import  ASTFeatureExtractor
import json
import argparse

from config import Config
import pandas as pd
import torch

from config import Config
from train import train_two_stage_model


def main():
    """
    Main function for running the two-stage AST-based ICBHI classifier
    """
    parser = argparse.ArgumentParser(description="Train AST-based two-stage ICBHI classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with preprocessed ICBHI dataset")
    parser.add_argument("--output_dir", type=str, default=Config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=Config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=Config.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--use_specaugment", action="store_true", help="Enable SpecAugment")
    parser.add_argument("--freq_mask_param", type=int, default=None, help="SpecAugment frequency mask parameter")
    parser.add_argument("--time_mask_param", type=int, default=None, help="SpecAugment time mask parameter")
    parser.add_argument("--n_freq_masks", type=int, default=None, help="SpecAugment number of frequency masks")
    parser.add_argument("--n_time_masks", type=int, default=None, help="SpecAugment number of time masks")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset csvs
    train_csv = os.path.join(args.data_dir, "train.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"Error: CSV files not found in {args.data_dir}")
        print("Run preprocessing first to generate CSV files.")
        return
    
    # Loading dataframes and feature extractor
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    
    feature_extractor = ASTFeatureExtractor.from_pretrained(Config.PRETRAINED_MODEL)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure SpecAugment parameters
    specaugment_params = {}
    
    # Use command line arguments if provided, otherwise fall back to Config
    use_specaugment = args.use_specaugment if args.use_specaugment else Config.SPECAUGMENT.get('enabled', True)
    
    if use_specaugment:
        specaugment_params = {
            'freq_mask_param': args.freq_mask_param or Config.SPECAUGMENT.get('freq_mask_param', 20),
            'time_mask_param': args.time_mask_param or Config.SPECAUGMENT.get('time_mask_param', 50),
            'n_freq_masks': args.n_freq_masks or Config.SPECAUGMENT.get('n_freq_masks', 2),
            'n_time_masks': args.n_time_masks or Config.SPECAUGMENT.get('n_time_masks', 2),
        }
        
        print("\nUsing SpecAugment with parameters:")
        for param, value in specaugment_params.items():
            print(f"  {param}: {value}")
    else:
        print("\nSpecAugment is disabled")
    
    # Training two-stage model
    results = train_two_stage_model(
        train_df=train_df,
        test_df=test_df,
        feature_extractor=feature_extractor,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        patience=args.patience,
        specaugment_params=specaugment_params if use_specaugment else None
    )
    
    # Saving config
    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "pretrained_model": Config.PRETRAINED_MODEL,
        "freeze_feature_extractor": Config.FREEZE_FEATURE_EXTRACTOR,
        "binary_classes": Config.BINARY_CLASSES,
        "abnormal_classes": Config.ABNORMAL_CLASSES,
        "all_classes": Config.CLASSES,
        "specaugment": {
            "enabled": use_specaugment,
            **specaugment_params
        } if use_specaugment else {"enabled": False}
    }
    
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    print("\nTraining and evaluation completed!")
    print(f"Models and results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()