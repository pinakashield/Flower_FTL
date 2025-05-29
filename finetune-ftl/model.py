import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
import numpy as np

class IntrusionModel(nn.Module):
    def __init__(self, input_dim, num_classes, freeze_base=False, dropout_rate=0.3):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256, track_running_stats=False),  # Modified BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, track_running_stats=False),  # Modified BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, track_running_stats=False),   # Modified BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, x):
        if x.size(0) == 1:  # Handle single sample case
            # Switch to eval mode temporarily for BatchNorm
            self.base.eval()
            with torch.no_grad():
                x = self.base(x)
            self.base.train()
        else:
            x = self.base(x)
        return self.head(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class EnsembleIDS(nn.Module):
    def __init__(self, input_dim, num_classes, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.is_traditional_models_fitted = False
        self.label_mapping = None
        
        # Deep Learning Models
        self.dnn = IntrusionModel(input_dim, num_classes)
        self.lstm = LSTMModel(input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes)
        
        # Traditional ML Models
        self.rf = RandomForestClassifier(n_estimators=100)
        self.xgb = xgb.XGBClassifier()
        
        # Ensemble weights
        self.weights = nn.Parameter(torch.ones(4) / 4)
        
    def setup_label_mapping(self, y):
        """Create mapping between original and consecutive labels"""
        unique_labels = np.unique(y)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        return self.label_mapping

    def train_traditional_models(self, X_train, y_train):
        try:
            # Map labels to consecutive integers
            if self.label_mapping is None:
                self.setup_label_mapping(y_train)
            y_mapped = np.array([self.label_mapping[y] for y in y_train])
            
            self.rf.fit(X_train, y_mapped)
            self.xgb.fit(X_train, y_mapped)
            # Only set classes_ for RandomForest
            self.rf.classes_ = np.arange(self.num_classes)
            self.is_traditional_models_fitted = True
            print("Traditional models successfully trained")
        except Exception as e:
            print(f"Error training traditional models: {e}")
            self.is_traditional_models_fitted = False
    
    def forward(self, x):
        # DNN prediction
        dnn_pred = torch.softmax(self.dnn(x), dim=1)
        lstm_pred = torch.softmax(self.lstm(x.unsqueeze(1)), dim=1)
        
        if not self.is_traditional_models_fitted:
            # If traditional models aren't trained, use only deep learning models
            return (self.weights[0] * dnn_pred + self.weights[1] * lstm_pred) / (self.weights[0] + self.weights[1])
            
        # Use all models if traditional models are trained
        x_np = x.cpu().numpy()
        rf_pred = torch.tensor(self.rf.predict_proba(x_np), dtype=torch.float32).to(self.device)
        xgb_pred = torch.tensor(self.xgb.predict_proba(x_np), dtype=torch.float32).to(self.device)
        # Pad rf_pred and xgb_pred if needed
        if rf_pred.shape[1] != self.num_classes:
            pad = self.num_classes - rf_pred.shape[1]
            rf_pred = torch.cat([rf_pred, torch.zeros(rf_pred.shape[0], pad, device=self.device)], dim=1)
        if xgb_pred.shape[1] != self.num_classes:
            pad = self.num_classes - xgb_pred.shape[1]
            xgb_pred = torch.cat([xgb_pred, torch.zeros(xgb_pred.shape[0], pad, device=self.device)], dim=1)
        # Weighted ensemble
        weighted_pred = (self.weights[0] * dnn_pred +
                         self.weights[1] * lstm_pred +
                         self.weights[2] * rf_pred +
                         self.weights[3] * xgb_pred)
        
        return weighted_pred

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, test_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(predictions)
                all_labels.extend(labels.numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        
        # Calculate per-class ROC-AUC
        n_classes = len(np.unique(all_labels))
        roc_aucs = {}
        for i in range(n_classes):
            if len(np.unique(all_labels)) == 2:
                roc_aucs[f'class_{i}'] = roc_auc_score((all_labels == i), all_probs[:, i])
            else:
                roc_aucs[f'class_{i}'] = roc_auc_score((all_labels == i), all_probs[:, i], multi_class='ovr')
        
        # Calculate precision-recall curves
        precisions = {}
        recalls = {}
        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
            precisions[f'class_{i}'] = prec
            recalls[f'class_{i}'] = rec
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_aucs,
            'precision': precisions,
            'recall': recalls,
            'probabilities': all_probs,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    @staticmethod
    def plot_metrics(metrics, save_path=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(f'{save_path}_confusion_matrix.png')
        plt.close()
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for class_idx, auc_score in metrics['roc_auc'].items():
            plt.plot(metrics['recall'][class_idx], metrics['precision'][class_idx], 
                    label=f'{class_idx} (AUC = {auc_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        if save_path:
            plt.savefig(f'{save_path}_pr_curves.png')
        plt.close()
