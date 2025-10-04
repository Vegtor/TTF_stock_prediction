import torch
import numpy as np
from torch.utils.data import DataLoader

class Early_Stopping:
    """
    Class for early stopping in training of models.
    """
    def __init__(self, patience=10):
        """
        Initialize the Early_Stopping object.

        Args:
            patience (int): Number of consecutive epochs without improvement in validation loss before stopping.
                Default is 10.
        """
        self.patience = patience
        self.counter = 0
        self.best_val = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        """
        Call this method after each epoch to check whether training should stop.

        Args:
            val_loss (float): Current epoch's validation loss.

        Returns:
            bool: Bool value indicating whether training should stop.
        """
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

class Stock_Predictor_TFT:
    def __init__(self, model, device, lr=0.001):
        """
        Initialize the Stock Predictor with a model, device, and learning rate.

        Args:
            model: The PyTorch model to train.
            device (torch.device): Device on which computations will be performed (e.g., 'cuda' or 'cpu').
            lr (float): Learning rate for the optimizer.
                Defaults to 0.001.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = None
        self.best_model = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lambda_smooth=0.3):
        """
        Train the model with early stopping and optional output smoothness regularization.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Number of training epochs.
            lambda_smooth (float): Weight for the smoothness loss component.

        Returns:
            dict: History of training and validation losses.
        """
        early_stopping = Early_Stopping(patience=15)
        best_val_loss = float('inf')
        self.best_model = None
        history = {'train_loss': [], 'val_loss': []}
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_X)

                output_squeezed = output.squeeze()

                criterion_loss = self.criterion(output_squeezed, batch_y)
                smoothness_loss = torch.mean(torch.abs(output_squeezed[1:] - output_squeezed[:-1]))
                loss = criterion_loss + lambda_smooth * smoothness_loss

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            val_loss = self.evaluate(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = self.model.state_dict().copy()

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)

            if early_stopping.step(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        self.model.load_state_dict(self.best_model)
        return history

    def save_best_model(self, name: str):
        """
        Save the best model to a file.

        Args:
            name (str): File name to save the model (without extension).
        """
        torch.save(self.best_model, './models/' + name + '.pth')

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on a given dataset in form of DataLoader.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation set.

        Returns:
            float: Average loss over the dataset.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                output = self.model(batch_X)
                loss = self.criterion(output.squeeze(), batch_y)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Generate predictions for a given input tensor.

        Args:
            X (torch.Tensor): Input tensor to predict on.

        Returns:
            np.ndarray: Model predictions as a NumPy array.
        """
        model = self.model
        model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = model(X)

        return predictions.cpu().numpy()
