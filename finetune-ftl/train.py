import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import EnsembleIDS, ModelEvaluator
import numpy as np

def train_ensemble(train_loader, val_loader, input_dim, num_classes, device, epochs=30):
    # Initialize model and optimizer
    model = EnsembleIDS(input_dim, num_classes, device).to(device)
    
    # Train traditional models first
    print("Training traditional models...")
    X_train_list = []
    y_train_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < 100:  # Use first 100 batches to train traditional models
            X_train_list.append(data.cpu().numpy())
            y_train_list.append(target.cpu().numpy())
    
    if X_train_list:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        model.train_traditional_models(X_train, y_train)
    
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Log metrics locally
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Batch Loss: {loss.item():.4f}")
        
        # Evaluate on validation set
        val_metrics = ModelEvaluator.evaluate_model(model, val_loader, device)
        avg_roc_auc = np.mean([score for score in val_metrics['roc_auc'].values()])
        print(f"Epoch {epoch}, Average Validation ROC AUC: {avg_roc_auc:.4f}, Epoch Loss: {total_loss / len(train_loader):.4f}")
        # Calculate and print accuracy
        accuracy = val_metrics.get('accuracy', None)
        if accuracy is not None:
            print(f"Epoch {epoch}, Validation Accuracy: {accuracy:.4f}")
        # Optionally print per-class ROC-AUC scores
        for class_idx, score in val_metrics['roc_auc'].items():
            print(f"  {class_idx} ROC-AUC: {score:.4f}")
    
    return model
