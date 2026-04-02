# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:34:33 2026

@author: ezzsy2
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class SurrogateRNN(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len, output_dim,
                 num_layers=2, dropout=0.2, cell_type="LSTM", bidirectional=False):

        super(SurrogateRNN, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cell_type = cell_type.upper()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Select RNN type
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, latent_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(input_dim, latent_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif self.cell_type == "RNN":
            self.rnn = nn.RNN(input_dim, latent_dim, num_layers=num_layers,
                              batch_first=True, nonlinearity="tanh",
                              dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("cell_type must be 'LSTM', 'GRU', or 'RNN'")

        
        # One head per sensor (output_dim sensors total)
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim * self.num_directions, 1) for _ in range(output_dim)
        ])
        

    def forward(self, x):
        """
        x: [batch, input_dim]
        out: [batch, seq_len, output_dim]
        """
        # Repeat input across seq_len to generate sequential output
        x_seq = x.unsqueeze(1).repeat(1, self.seq_len, 1)   # [batch, seq_len, input_dim]
        rnn_out, _ = self.rnn(x_seq)                        # [batch, seq_len, hidden]

        # Apply each head separately
        outs = [head(rnn_out) for head in self.heads]       # list of [batch, seq_len, 1]
        out = torch.cat(outs, dim=-1)                       # [batch, seq_len, output_dim]
        return out


class SurrogateTrainer:
    def __init__(self, model, lr, patience, max_epochs, train_loader, val_loader, 
                 scaler_X, scaler_Y, scaled_thresh_array, device=None):
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
        
        self.scaled_thresh_array = scaled_thresh_array.to(self.device)
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # self.criterion = nn.MSELoss()
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # MSE per sensor
        self.train_losses_per_sensor = []
        self.val_losses_per_sensor = []
        
        # Global MSE
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Compute surrogate covariance
        self.surr_cov = "Training required first"
        self.surr_error = "Training required first"
        
        

    def masked_weighted_mse_scaled(self, y_pred, y_true_scaled, sensor_weights=None):
    
        # broadcast threshold to match batch & seq_len
        mask = (y_true_scaled.abs() > self.scaled_thresh_array[None, None, :]).float()  # [batch, seq_len, sensors]
    
        masked_sq_error = ((y_pred - y_true_scaled)**2) * mask
    
        # Avoid division by zero
        count_per_sensor = mask.sum(dim=(0,1)).clamp(min=1.0)
        per_sensor_loss = masked_sq_error.sum(dim=(0,1)) / count_per_sensor
    
        if sensor_weights is not None:
            sensor_weights = sensor_weights.to(self.device)
            per_sensor_loss = per_sensor_loss * sensor_weights
    
        total_loss = per_sensor_loss.mean()
        return total_loss, per_sensor_loss
    
    
    def train_one_epoch(self):
        self.model.train()
        per_sensor_losses = []
        total_losses = []

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)

            loss, per_sensor_loss = self.masked_weighted_mse_scaled(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_losses.append(loss.item())
            per_sensor_losses.append(per_sensor_loss.detach().cpu().numpy())

        return np.mean(total_losses), np.mean(per_sensor_losses, axis=0)

    
    def validate_one_epoch(self):
        self.model.eval()
        per_sensor_losses = []
        total_losses = []
        # Add these new lists to store mask counts
        masked_counts = []
        total_counts = []

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(x_batch)

                # Create the mask as you did before
                mask = (y_batch.abs() > self.scaled_thresh_array[None, None, :]).float()
                
                # Calculate and append counts for the current batch
                masked_counts.append((1 - mask).sum(dim=(0, 1)).cpu().numpy())
                total_counts.append(torch.ones_like(mask).sum(dim=(0, 1)).cpu().numpy())

                loss, per_sensor_loss = self.masked_weighted_mse_scaled(y_pred, y_batch)
                total_losses.append(loss.item())
                per_sensor_losses.append(per_sensor_loss.detach().cpu().numpy())

        # Calculate total counts and masked percentages after the loop
        total_masked_counts = np.sum(masked_counts, axis=0)
        total_data_points = np.sum(total_counts, axis=0)
        # Avoid division by zero
        masked_percentages = (total_masked_counts / total_data_points) * 100
        
        return np.mean(total_losses), np.mean(per_sensor_losses, axis=0), masked_percentages
    
    # surrogate training
    def fit(self,save_dir):
        
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            train_loss, train_loss_per_sensor = self.train_one_epoch()
            # capture the new masked_percentages return value
            val_loss, val_loss_per_sensor, masked_percentages = self.validate_one_epoch()

            # Store histories
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_losses_per_sensor.append(train_loss_per_sensor)
            self.val_losses_per_sensor.append(val_loss_per_sensor)

            print(f"Epoch {epoch+1}/{self.max_epochs} "
                  f"- Train Loss (mean): {train_loss:.6f}, Val Loss (mean): {val_loss:.6f}")
            
            
            # # Print the mse per sensor
            # if (epoch + 1) % 50 == 0:
            #     self.print_per_sensor_mse(epoch, val_loss_per_sensor)
                
            # Early stopping on global validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        
        # Save training history to local
        latent_dim = getattr(self.model, "latent_dim", "NA")
        num_layers = getattr(self.model, "num_layers", "NA")
        cell_type = getattr(self.model, "cell_type", "NA")
        
        # Build filename
        filename = f"training_history_ld{latent_dim}_nl{num_layers}_{str(cell_type).lower()}.pt"
        save_path = os.path.join(save_dir, filename)
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_losses_per_sensor": self.train_losses_per_sensor,
            "val_losses_per_sensor": self.val_losses_per_sensor,
            "best_val_loss": self.best_val_loss,
            "model_info": {
                "latent_dim": latent_dim,
                "num_layers": num_layers,
                "cell_type": cell_type,
            },
        }
        
        torch.save(history, save_path)
        print(f"Training history saved to: {save_path}")
        
        # Load best model before returning
        # so now trainer.model is the best model
        self.load_best_model()
        

    def load_best_model(self):
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model (Val Loss = {self.best_val_loss:.6f})")
        else:
            print("No best model found.")
            
            
    def plot_losses(self, save_dir):
        plt.figure(figsize=(8, 6))
        plt.plot(np.log10(self.train_losses), label="Train Loss (log MSE)")
        plt.plot(np.log10(self.val_losses), label="Validation Loss (log MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("log(MSE)")
        plt.title("Training vs Validation Loss (Global)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,"global_loss.png"), dpi=150)
        plt.show()
        plt.close()

    def plot_per_sensor_losses(self, save_dir):
        sensors = np.arange(len(self.train_losses_per_sensor[0]))

        # Training
        plt.figure(figsize=(10,6))
        for s in sensors:
            plt.plot([epoch[s] for epoch in self.train_losses_per_sensor], label=f"Sensor {s}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training MSE per Sensor")
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,"train_mse_per_sensor.png"), dpi=150)
        plt.close()

        # Validation
        plt.figure(figsize=(10,6))
        for s in sensors:
            plt.plot([epoch[s] for epoch in self.val_losses_per_sensor], label=f"Sensor {s}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Validation MSE per Sensor")
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,"val_mse_per_sensor.png"), dpi=150)
        plt.close()

    def inverse_scale_dataset(self, dataset, scaler_X, scaler_Y):
        # Inverse transform the scaled dataset back to ground truth values.
        
        X_scaled, Y_scaled = dataset.tensors  # unpack tensors
        X_np = X_scaled.cpu().numpy()
        Y_np = Y_scaled.cpu().numpy()
        
        # Inverse transform
        X_raw = scaler_X.inverse_transform(X_np)
    
        N, T, S = Y_np.shape
        Y_flat = Y_np.reshape(-1, S)  # (N*T, S)
        Y_raw_flat = scaler_Y.inverse_transform(Y_flat)
        Y_raw = Y_raw_flat.reshape(N, T, S)
        
        return X_raw, Y_raw

        # ---- Compute Surrogate Covariance ----
    def compute_surrogate_covariance(self, save_dir=None):
        # calculate the surrogate covariance and error using the trained model
        # on the validation dataset.
        
        X_val_scaled, _ = self.val_loader.dataset.tensors
        X_val_scaled = X_val_scaled.to(self.device)
        with torch.no_grad():
            Y_pred_scaled = self.model(X_val_scaled).cpu().numpy()
            
        # Inverse scaling
        pred_dataset = TensorDataset(self.val_loader.dataset.tensors[0],
                                     torch.tensor(Y_pred_scaled, dtype=torch.float32))
        _, Y_pred_raw = self.inverse_scale_dataset(pred_dataset, self.scaler_X, self.scaler_Y)
        _, Y_val_raw = self.inverse_scale_dataset(self.val_loader.dataset, self.scaler_X, self.scaler_Y)

        # Reshape to (N, T*S) and compute covariance
        Y_pred_flat = Y_pred_raw.reshape(Y_pred_raw.shape[0], -1)
        Y_val_flat = Y_val_raw.reshape(Y_val_raw.shape[0], -1)
        self.surr_cov = np.cov((Y_pred_flat - Y_val_flat).T)
        self.surr_error = np.mean(Y_pred_flat-Y_val_flat, axis=0)
        print("Surrogate covariance and error computed.")
        
        # Optional saving
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "surrogate_cov_error.pt")
            torch.save(
                {"surr_cov": self.surr_cov, "surr_error": self.surr_error},
                save_path
            )
            print(f"Saved surrogate covariance and error to: {save_path}")
            
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


##############################################################################

def scale_and_split(X_raw, Y_raw, test_size, batch_size, random_state, eki_percentage):
    
    # 1. Fit scalers on full data
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    scaler_Y = StandardScaler()
    Y_reshaped = Y_raw.reshape(-1, Y_raw.shape[2])  # (N*T, S)
    Y_scaled = scaler_Y.fit_transform(Y_reshaped)
    Y_scaled = Y_scaled.reshape(Y_raw.shape)  # back to (N, T, S)
    
    # 2. Convert to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)
    
    # 3. Create EKI dataset from a random subset of the whole dataset
    n_samples = len(X_tensor)
    n_eki = max(1, int(eki_percentage * n_samples))
    np.random.seed(random_state)
    eki_indices = np.random.choice(n_samples, n_eki, replace=False)
    eki_dataset = TensorDataset(X_tensor[eki_indices], Y_tensor[eki_indices])
    
    # 4. Remove EKI indices from data for train/val split
    all_indices = np.arange(n_samples)
    train_val_indices = np.setdiff1d(all_indices, eki_indices)
    
    X_train_val = X_tensor[train_val_indices]
    Y_train_val = Y_tensor[train_val_indices]
    
    # 5. Train-test split on remaining data
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=test_size, random_state=random_state
    )
    
    # 6. Create datasets and loaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler_X, scaler_Y, train_dataset, val_dataset, eki_dataset
