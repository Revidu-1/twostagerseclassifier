from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel 

class ASTRespiratoryModel(nn.Module):
    """
    AST-based respiratory sound classifier
    """
    def __init__(
        self,
        pretrained_model: str = Config.PRETRAINED_MODEL,
        num_classes: int = len(Config.CLASSES),
        dropout_rate: float = Config.DROPOUT_RATE,
        freeze_feature_extractor: bool = Config.FREEZE_FEATURE_EXTRACTOR
    ):
        super(ASTRespiratoryModel, self).__init__()
        
        # Load pretrained AST model
        self.ast = ASTModel.from_pretrained(pretrained_model)
        self.hidden_size = self.ast.config.hidden_size
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            self._freeze_feature_extractor()
        self.attention_pool = nn.Sequential(
        nn.Linear(self.hidden_size, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
        nn.Softmax(dim=1)
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
        
        # Storing class names
        if num_classes == len(Config.CLASSES):
            self.class_names = list(Config.CLASSES.keys())
        elif num_classes == len(Config.BINARY_CLASSES):
            self.class_names = list(Config.BINARY_CLASSES.keys())
        elif num_classes == len(Config.ABNORMAL_CLASSES):
            self.class_names = list(Config.ABNORMAL_CLASSES.keys())
        else:
            self.class_names = [str(i) for i in range(num_classes)]
    
    def _freeze_feature_extractor(self):
        """
        Freeze most layers, keeping only the last few transformer layers trainable
        """
        # Freezing all parameters initially then unfreezing the last four layers
        for param in self.ast.parameters():
            param.requires_grad = False
        
        unfreeze_layers = 4
        if hasattr(self.ast, 'encoder') and hasattr(self.ast.encoder, 'layer'):
            layers = self.ast.encoder.layer
            for i in range(len(layers) - unfreeze_layers, len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
    
    def forward(
        self,
        input_values,
        return_attention=False
    ):
        """
        Forward pass through the model
        
        Args:
            input_values: AST input features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with model outputs
        """
        # Processing AST features 
        outputs = self.ast(
            input_values,
            output_attentions=return_attention
        )
        
        # Using the [CLS] token representation for classification
        cls_representation = outputs.last_hidden_state[:, 0, :]
        
        # Apply classifier
        logits = self.classifier(cls_representation)
        
        # Return outputs
        result = {
            'logits': logits,
            'features': cls_representation
        }
        
        if return_attention and outputs.attentions:
            result['ast_attentions'] = outputs.attentions
        
        return result
    
    def focal_loss(
        self,
        logits,
        labels,
        gamma=2.5,
        alpha=None
    ):
        """
        Compute focal loss for class imbalance
        
        Args:
            logits: Predicted logits
            labels: True labels
            gamma: Focal loss gamma parameter
            alpha: Class weights
            
        Returns:
            Focal loss value
        """
        # Get class probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of true class
        batch_size = logits.size(0)
        p_t = probs[torch.arange(batch_size), labels]
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** gamma
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weight if provided
        if alpha is not None:
            alpha_weight = alpha[labels]
            focal_loss = alpha_weight * focal_loss
        
        # Return mean loss
        return focal_loss.mean()
    
    def compute_loss(self, logits, labels, class_weights=None):
        if class_weights is None:
            # Calculate class weights based on label distribution
            class_counts = torch.bincount(labels, minlength=len(self.class_names))
            class_weights = 1. / (class_counts.float() + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            class_weights = class_weights.to(logits.device)
        
        return self.focal_loss(logits, labels, gamma=2.0, alpha=class_weights)