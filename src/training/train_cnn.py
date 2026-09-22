import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# Assuming CustomCNN and OrdinalFocalLoss are imported from your respective modules
# from src.models.cnn_backbone import CustomCNN
# from src.training.losses import OrdinalFocalLoss

def train_epoch(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(dataloader.dataset)

def validate_epoch(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, 
                   device: torch.device) -> tuple:
    # Note: Spatial dropout remains active via the module definition, but standard 
    # batch normalization should be evaluated using running statistics.
    model.eval()
    running_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def execute_two_stage_training(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                               class_weights: torch.Tensor, device: torch.device, epochs_stage1: int = 10, epochs_stage2: int = 20):
    
    criterion = OrdinalFocalLoss(alpha_weights=class_weights, gamma=2.0, emd_weight=1.0).to(device)
    
    # --- STAGE 1: Train Classification Head Only ---
    print("--- Starting Stage 1: Head-Only Training ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # High learning rate for initial head adaptation
    optimizer_stage1 = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    for epoch in range(epochs_stage1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Stage 1 | Epoch {epoch+1}/{epochs_stage1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # --- STAGE 2: End-to-End Fine-Tuning ---
    print("\n--- Starting Stage 2: End-to-End Fine-Tuning ---")
    for param in model.parameters():
        param.requires_grad = True
        
    # Lower learning rate for stable full-network convergence
    optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    for epoch in range(epochs_stage2):
        train_loss = train_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Stage 2 | Epoch {epoch+1}/{epochs_stage2} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_cnn_backbone.pth")
            print(">> Saved new best model weights.")