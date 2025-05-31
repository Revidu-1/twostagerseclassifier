import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from config import Config
from dataset import ICBHIASTDataset, ast_collate_fn
from model import ASTRespiratoryModel


def train_two_stage_model(
    train_df,
    test_df,
    feature_extractor,
    device,
    output_dir=Config.OUTPUT_DIR,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
    specaugment_params=None
):
    """
    Training the two-stage model:
    1. First stage: normal vs abnormal
    2. Second stage: classify abnormal sounds (crackle, wheeze, both)
    
    """
    if specaugment_params is None and hasattr(Config, 'SPECAUGMENT'):
        specaugment_params = {
            'freq_mask_param': Config.SPECAUGMENT.get('freq_mask_param', 27),
            'time_mask_param': Config.SPECAUGMENT.get('time_mask_param', 100),
            'n_freq_masks': Config.SPECAUGMENT.get('n_freq_masks', 2),
            'n_time_masks': Config.SPECAUGMENT.get('n_time_masks', 2),
        }
    elif specaugment_params is None:
        specaugment_params = {}
    
    apply_specaugment = Config.SPECAUGMENT.get('enabled', True) if hasattr(Config, 'SPECAUGMENT') else True
    
    print("\n=== Stage 1: Binary Classification (Normal vs Abnormal) ===")
    
    # Creating binary datasets
    train_dataset_binary = ICBHIASTDataset(
        train_df,
        feature_extractor=feature_extractor,
        transform=True,
        stage='binary',
        apply_specaugment=apply_specaugment,
        specaugment_params=specaugment_params
    )
    
    test_dataset_binary = ICBHIASTDataset(
        test_df,
        feature_extractor=feature_extractor,
        transform=False,
        stage='binary',
        apply_specaugment=False  # No augmentation for test set
    )
    
    # Creating data loaders
    train_sampler_binary = train_dataset_binary.create_balanced_sampler()
    
    train_loader_binary = DataLoader(
        train_dataset_binary,
        batch_size=batch_size,
        sampler=train_sampler_binary,
        num_workers=2,
        pin_memory=True,
        collate_fn=ast_collate_fn  # Using custom collate function
    )
    
    test_loader_binary = DataLoader(
        test_dataset_binary,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=ast_collate_fn  # Use custom collate function
    )
    
    # Create binary model
    binary_model = ASTRespiratoryModel(
        pretrained_model=Config.PRETRAINED_MODEL,
        num_classes=len(Config.BINARY_CLASSES),
        dropout_rate=Config.DROPOUT_RATE,
        freeze_feature_extractor=Config.FREEZE_FEATURE_EXTRACTOR
    ).to(device)
    
    # Train binary model
    binary_model, binary_metrics = train_model(
        model=binary_model,
        train_loader=train_loader_binary,
        val_loader=test_loader_binary,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_dir=os.path.join(output_dir, "stage1_binary"),
        patience=patience
    )
    
    print("Stage 2: Abnormal Classification (Crackle, Wheeze, Both) ===")
    
    # Create abnormal datasets
    train_dataset_abnormal = ICBHIASTDataset(
        train_df,
        feature_extractor=feature_extractor,
        transform=True,
        stage='abnormal',
        apply_specaugment=apply_specaugment,
        specaugment_params=specaugment_params
    )
    
    test_dataset_abnormal = ICBHIASTDataset(
        test_df,
        feature_extractor=feature_extractor,
        transform=False,
        stage='abnormal',
        apply_specaugment=False  # No augmentation for test set
    )
    
    # Creating data loaders
    train_sampler_abnormal = train_dataset_abnormal.create_balanced_sampler()
    
    train_loader_abnormal = DataLoader(
        train_dataset_abnormal,
        batch_size=batch_size,
        sampler=train_sampler_abnormal,
        num_workers=2,
        pin_memory=True,
        collate_fn=ast_collate_fn  # Using collate function definned in dataset.py
    )
    
    test_loader_abnormal = DataLoader(
        test_dataset_abnormal,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=ast_collate_fn  
    )
    
    # Creating abnormal model
    abnormal_model = ASTRespiratoryModel(
        pretrained_model=Config.PRETRAINED_MODEL,
        num_classes=len(Config.ABNORMAL_CLASSES),
        dropout_rate=Config.DROPOUT_RATE,
        freeze_feature_extractor=Config.FREEZE_FEATURE_EXTRACTOR
    ).to(device)
    
    # Training abnormal model
    abnormal_model, abnormal_metrics = train_model(
        model=abnormal_model,
        train_loader=train_loader_abnormal,
        val_loader=test_loader_abnormal,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_dir=os.path.join(output_dir, "stage2_abnormal"),
        patience=patience
    )
    
    print("\n=== Evaluating Two-Stage Model ===")
    
    # Creaing regular test dataset for full evaluation
    test_dataset_full = ICBHIASTDataset(
        test_df,
        feature_extractor=feature_extractor,
        transform=False,
        stage='all',
        apply_specaugment=False  # No augmentation for test set
    )
    
    test_loader_full = DataLoader(
        test_dataset_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=ast_collate_fn  # Using custom collate function
    )
    
    # Evaluating two-stage model
    combined_results = evaluate_two_stage_model(
        binary_model=binary_model,
        abnormal_model=abnormal_model,
        test_loader=test_loader_full,
        device=device,
        output_dir=output_dir
    )
    
    return {
        'binary_model': binary_model,
        'abnormal_model': abnormal_model,
        'binary_metrics': binary_metrics,
        'abnormal_metrics': abnormal_metrics,
        'combined_metrics': combined_results
    }

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    learning_rate,
    output_dir,
    patience
):
    """
    Train a model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs
        learning_rate: Learning rate
        output_dir: Directory to save model
        patience: Early stopping patience
        
    Returns:
        Trained model and evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2, verbose=True
    )
    
    # Training variables
    best_val_loss = float('inf')
    best_model = None
    best_metrics = None
    patience_counter = 0
    best_val_f1=0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    num_classes = len(model.class_names)
    print(f"Training model with {num_classes} classes: {model.class_names}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in progress_bar:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass 
            outputs = model(input_values)
            
            # Compute loss
            loss = model.compute_loss(outputs['logits'], labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            
            # Updating parameters
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculating average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Moving data to device
                input_values = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_values)
                
                loss = model.compute_loss(outputs['logits'], labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs['logits'], dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculating average validation loss and metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        # Updating history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Check if this is the best model - using F1 score 
        if val_f1 > best_val_f1:  
            best_val_f1 = val_f1  
            best_metrics = {
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            }
    
            
            # Savinng a deep copy of the model
            best_model = type(model)(
                pretrained_model=Config.PRETRAINED_MODEL,
                num_classes=num_classes,
                dropout_rate=Config.DROPOUT_RATE,
                freeze_feature_extractor=Config.FREEZE_FEATURE_EXTRACTOR
            ).to(device)
            best_model.load_state_dict(model.state_dict())
            
            # Saving model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': model.class_names
            }, os.path.join(output_dir, 'best_model.pt'))
            
            # Save confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=model.class_names, 
                       yticklabels=model.class_names)
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()
            
            # Reset patience counter
            patience_counter = 0
            print(f"  New best model saved!")
        else:
            # Increment patience counter
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Updating learning rate
        scheduler.step(val_loss)
    
    # Saving training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot(history['val_f1'], label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']],
            'val_f1': [float(x) for x in history['val_f1']]
        }, f, indent=4)
    
    # Returning best model and metrics
    return best_model or model, best_metrics or {
        'epoch': num_epochs,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }

def evaluate_two_stage_model(
    binary_model,
    abnormal_model,
    test_loader,
    device,
    output_dir
):
    """
    Evaluate the two-stage model on a test set
    
    Args:
        binary_model: Binary classification model (normal vs abnormal)
        abnormal_model: Abnormal classification model (crackle, wheeze, both)
        test_loader: Test data loader with full classes
        device: Device to run evaluation on
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Setting models to evaluation mode
    binary_model.eval()
    abnormal_model.eval()
    
    # Metrics
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating two-stage model"):
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Stage 1: Binary classification (normal vs abnormal)
            binary_outputs = binary_model(input_values)
            binary_preds = torch.argmax(binary_outputs['logits'], dim=1)
            
            # Initialising final predictions with all normal
            final_preds = torch.zeros_like(labels)            
            abnormal_indices = (binary_preds == 1).nonzero(as_tuple=True)[0]
            
            # If there are abnormal predictions, run stage 2
            if len(abnormal_indices) > 0:
                abnormal_inputs = input_values[abnormal_indices]
                
                abnormal_outputs = abnormal_model(abnormal_inputs)
                abnormal_preds = torch.argmax(abnormal_outputs['logits'], dim=1)
                
                # Mapping abnormal predictions to original class indices
                
                abnormal_mapped = abnormal_preds + 1                
                final_preds[abnormal_indices] = abnormal_mapped
            
            # Storing predictions and labels
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculating metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    precision = []
    recall = []
    
    for i in range(len(class_names)):
        # Precision (how many selected items are relevant)
        if cm[:, i].sum() > 0:
            precision.append(float(cm[i, i] / cm[:, i].sum()))
        else:
            precision.append(0.0)
        
        # Recall (how many relevant items are selected)
        if cm[i, :].sum() > 0:
            recall.append(float(cm[i, i] / cm[i, :].sum()))
        else:
            recall.append(0.0)
    
    # Per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': float(f1_per_class[i])
        }
    
    # Visualising confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Two-Stage Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'two_stage_confusion_matrix.png'))
    plt.close()
    
    metrics = {
        'accuracy': float(acc),
        'f1_score_macro': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_metrics': class_metrics
    }
    
    # Saving metrics to JSON
    with open(os.path.join(output_dir, 'two_stage_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Printing results
    print("\nTwo-Stage Model Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score (Macro): {f1:.4f}")
    print("  Per-class metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"    {class_name}:")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall: {metrics['recall']:.4f}")
        print(f"      F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics
