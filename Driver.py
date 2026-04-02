# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:31:12 2026

@author: ezzsy2
"""

import os
import time
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from RNN import SurrogateRNN, SurrogateTrainer
from RNN import scale_and_split

###############################################################################
########################     Initialisation     ###############################
###############################################################################
# Load data
Inputs = np.load('Inputs_10466.npy')
Outputs = np.load('Output_10466_s20.npy')

# read observation times and number of sensors from the data
obs_time = Outputs.shape[1]
num_sensor = Outputs.shape[2]

X_raw = Inputs
Y_raw = Outputs

# Split data to three sets: train, validation, eki_test
train_loader, val_loader, scaler_X, scaler_Y, train_dataset, val_dataset, eki_dataset = scale_and_split(
    X_raw, Y_raw, test_size=0.2, batch_size=64, random_state=42, eki_percentage=0.02)

# calculate the threshold for sensor activation in scaled space
raw_thresh = 1000
raw_thresh_array = np.full(scaler_Y.scale_.shape, raw_thresh)
scaled_thresh = torch.tensor(raw_thresh_array/scaler_Y.scale_, dtype=torch.float32)


###############################################################################
######################     Build Neural network     ###########################
###############################################################################

# Initialise model ============================================================
latent_dim = 1024
num_layers = 1
cell_type="LSTM"

model = SurrogateRNN(
    input_dim=71,   # number of parameters
    latent_dim=latent_dim,
    seq_len=obs_time, # number of observation time
    output_dim=num_sensor,  # number of sensors
    num_layers=num_layers,
    dropout=0,  #only 1 layer, no drop-out happening, if 2 layers, use dropout
    cell_type=cell_type,    # LSTM, RNN or GRU
    bidirectional=True  # <-- Switch on/off
)

trainer = SurrogateTrainer(
    model, lr=1e-3, patience=50, max_epochs=1000, 
    train_loader=train_loader, val_loader=val_loader, 
    scaler_X=scaler_X, scaler_Y=scaler_Y, scaled_thresh_array=scaled_thresh, device='cpu')


###############################################################################
#########################     Training on HPC   ###############################
###############################################################################
save_dir = os.path.join(os.getcwd(), "surrogate")
os.makedirs(save_dir, exist_ok=True)
# Train surrogate =============================================================
start_time = time.time()
trainer.fit(save_dir)
elapsed = time.time()-start_time
print(f"Training finished in {elapsed:.1f}s")

# Save best model after training ==============================================
best_model_name = f"best_model_ld{latent_dim}_nl{num_layers}_{cell_type.lower()}.pth"
torch.save(trainer.best_model_state, os.path.join(save_dir, best_model_name))
