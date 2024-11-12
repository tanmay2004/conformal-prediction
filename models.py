import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ConformalGCN(torch.nn.Module):
    """
    Conformal GCN that supports both regression (CQR) and classification (APS) tasks.
    regression:     outputs [predictions, lower bounds, upper bounds]
    classification: outputs [probabilities]
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, task: str = 'regression'):
        """
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            task: Either 'regression' or 'classification'
        """
        super().__init__()
        self.task = task
        
        # For regression: multiply output by 3 for [pred, lower, upper]
        # For classification: directly use class probabilities
        final_dim = out_channels * 3 if task == 'regression' else out_channels
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, final_dim)
        
    def forward(self, x, edge_index):
        """Forward pass through the network"""
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        if self.task == 'regression':
            # Split into prediction and bounds
            pred  = x[:, :x.size(1) // 3]
            lower = x[:, x.size(1) // 3: 2 * x.size(1) // 3]
            upper = x[:, 2 * x.size(1) // 3:]
            return torch.cat([pred, lower, upper], dim=1)
        else: return x

class ConformalPredictor:
    """
    A conformal prediction wrapper supporting both CQR and APS.
    
    CQR (Conformalized Quantile Regression):
        - Used for regression tasks
        - Provides prediction intervals [lower, upper]
    
    APS (Adaptive Prediction Sets):
        - Used for classification tasks
        - Provides set of possible classes with guaranteed coverage
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        task: str = 'regression',
        alpha: float = 0.1,
        lr: float = 0.01,
        epochs: int = 200
    ):
        """
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of outputs (1 for regression, num_classes for classification)
            task: Either 'regression' or 'classification'
            alpha: Miscoverage level (1 - confidence level)
            lr: Learning rate
            epochs: Number of training epochs
        """
        self.task = task
        self.model = ConformalGCN(in_channels, hidden_channels, out_channels, task)
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def _train_regression(self, data, train_idx):
        """Train the model for regression using quantile losses"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            
            # Split outputs
            pred  = out[:, :out.size(1) // 3]
            lower = out[:, out.size(1) // 3: 2 * out.size(1) // 3]
            upper = out[:, 2 * out.size(1) // 3:]

            # Quantile losses for bounds
            low_quantile = self.alpha / 2
            up_quantile = 1 - self.alpha / 2
            
            # MSE loss for predictions
            loss = F.mse_loss(pred[train_idx], data.y[train_idx])
            
            # Pinball loss for lower bound
            low_loss = torch.mean(torch.max(
                (low_quantile-1) * (data.y[train_idx]-lower[train_idx]),
                low_quantile * (data.y[train_idx]-lower[train_idx])
            ))
            
            # Pinball loss for upper bound
            up_loss = torch.mean(torch.max(
                (up_quantile-1) * (data.y[train_idx]-upper[train_idx]),
                up_quantile * (data.y[train_idx]-upper[train_idx])
            ))

            # Aggregate loss
            loss = loss + low_loss + up_loss
            loss.backward()
            optimizer.step()

    def _train_classification(self, data, train_idx):
        """Train the model for classification using cross entropy loss"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

    def _calibrate_regression(self, data, calib_idx):
        """Calibrate regression model using CQR"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out[:, :out.size(1)//3]
            lower = out[:, out.size(1)//3:2*out.size(1)//3]
            upper = out[:, 2*out.size(1)//3:]
            
            # Compute conformity scores
            cal_scores = torch.maximum(
                data.y[calib_idx] - upper[calib_idx],
                lower[calib_idx] - data.y[calib_idx]
            )
            
            # Compute correction
            n_cal = len(calib_idx)
            quantile = torch.tensor(
                min(max((n_cal + 1) * (1-self.alpha) / n_cal, 0), 1),
                dtype=cal_scores.dtype,
                device=self.device
            )
            self.qhat = torch.quantile(
                cal_scores,
                quantile
            )

    def _calibrate_classification(self, data, calib_idx):
        """Calibrate classification model using APS"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)
            
            # Sort probabilities in descending order
            sorted_probs, _ = torch.sort(probs[calib_idx], descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=1)
            
            # Compute conformity scores
            cal_scores = cumsum_probs[torch.arange(len(calib_idx)), data.y[calib_idx]]
            
            # Compute threshold
            n_cal = len(calib_idx)
            quantile = torch.tensor(
                min(max((n_cal + 1) * (1-self.alpha) / n_cal, 0), 1),
                dtype=cal_scores.dtype,
                device=self.device
            )
            self.threshold = torch.quantile(cal_scores, quantile)

    def fit(self, data, train_idx=None, calib_idx=None):
        """
        Train and calibrate the model.
        
        Args:
            data: PyG Data object
            train_idx: Training indices
            calib_idx: Calibration indices
        """
        data = data.to(self.device)
        
        # Create split if not provided
        n = data.x.size(0)
        if train_idx is None or calib_idx is None:
            perm = torch.randperm(n)
            train_idx = perm[:int(0.7*n)]
            calib_idx = perm[int(0.7*n):int(0.85*n)]
        
        # Train
        if self.task == 'regression':
            self._train_regression(data, train_idx)
        else:
            self._train_classification(data, train_idx)
            
        # Calibrate
        if self.task == 'regression':
            self._calibrate_regression(data, calib_idx)
        else:
            self._calibrate_classification(data, calib_idx)

    def predict(self, data, idx=None):
        """
        Make predictions with uncertainty quantification.
        
        Args:
            data: PyG Data object
            idx: Indices to predict on (optional)
            
        Returns:
            For regression: (predictions, lower bounds, upper bounds)
            For classification: (predictions, prediction sets)
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if self.task == 'regression':
                out = self.model(data.x, data.edge_index)
                pred = out[:, :out.size(1)//3]
                lower = out[:, out.size(1)//3:2*out.size(1)//3]
                upper = out[:, 2*out.size(1)//3:]
                
                # Apply conformal correction
                lower = lower - self.qhat
                upper = upper + self.qhat
                
                if idx is not None:
                    pred = pred[idx]
                    lower = lower[idx]
                    upper = upper[idx]
                    
                return pred, lower, upper
            
            else:
                logits = self.model(data.x, data.edge_index)
                probs = F.softmax(logits, dim=1)
                
                # Get prediction sets
                cumsum_probs = torch.cumsum(torch.sort(probs, dim=1, descending=True)[0], dim=1)
                pred_sets = cumsum_probs <= self.threshold
                
                # Get predictions (class with highest probability)
                pred = torch.argmax(probs, dim=1)
                
                if idx is not None:
                    pred = pred[idx]
                    pred_sets = pred_sets[idx]
                
                return pred, pred_sets