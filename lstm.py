import torch.nn as nn
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import tqdm

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float = 0.1):

        """
        Inicializa una red LSTM para regresión con soporte para Dropout.

        Parámetros:
            input_size (int): Tamaño de la entrada.
            hidden_size (int): Tamaño del estado oculto.
            n_layers (int): Número de capas de la LSTM.
            dropout (float): Proporción de dropout (por defecto, 0.1).

        """

        super().__init__() 

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = n_layers, dropout = dropout, batch_first = True)
        
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define el paso hacia adelante de la red.

        Parámetros:
            x (torch.Tensor): Entrada con forma [batch_size, seq_len, input_size].

        Retorna:
            out (torch.Tensor): Predicción con forma [batch_size, 1].
        """

        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)

        _, (h_n, _) = self.lstm(x, (h_0, c_0))

        last_hidden_state = h_n[-1]

        out = self.fc(last_hidden_state)

        return out
    
class SequenceDataset(Dataset):
    """
    Dataset para manejar secuencias generadas por DataCreator.
    """
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_loop(
    model, 
    train_seq, 
    val_seq, 
    training_params,
    criterion = nn.L1Loss()
):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = SequenceDataset(train_seq)
    val_data = SequenceDataset(val_seq)

    optimizer = torch.optim.Adam(model.parameters(), lr = training_params["learning_rate"])


    #(d) Dataloaders con batch size determinado

    train_dataloader = DataLoader(train_data, batch_size = training_params["batch_size"], shuffle = False, pin_memory = True, num_workers = 6, drop_last = False)
    val_dataloader = DataLoader(val_data, batch_size = training_params["batch_size"], shuffle = False, pin_memory = True, num_workers = 6)
    

    train_metric = torchmetrics.MeanSquaredError(squared = False, num_outputs = 1).to(device)
    val_metric = torchmetrics.MeanSquaredError(squared = False, num_outputs = 1).to(device)

    train_losses = []
    val_losses = []

    model.to(device)

        #(e) Training y Validation Loop
    for e in range(training_params["num_epochs"]):

        start_time = time.time()

        train_batch_losses = []
        test_batch_losses = []

        model.train()
        # (e) Training
        for batch in train_dataloader:

            X, y = batch[0].to(device), batch[1].unsqueeze(1).to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            tm = train_metric(y_pred, y)
            train_batch_losses.append(loss.item())
    
        #Obtenemos el promedio del loss en el entrenamiento de cada batch        
        tm = train_metric.compute()
        train_epoch_loss = np.mean(train_batch_losses)

        model.eval()
        #(e) Validation
        with torch.no_grad():

            for batch in val_dataloader:

                X, y = batch[0].to(device), batch[1].unsqueeze(1).to(device)

                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_m = val_metric(y_pred, y)
                test_batch_losses.append(loss.item())
        
        #Obtenemos el promedio del loss de cada batch
        val_m = val_metric.compute()
        test_epoch_loss = np.mean(test_batch_losses)
        end_time = time.time()

        train_losses.append(train_epoch_loss)
        val_losses.append(test_epoch_loss)

        epoch_time = end_time - start_time

        #(f) Reporte final 
        print(f"Epoch: {e + 1}- Time: {epoch_time: .2f} - Train L1 Loss: {train_epoch_loss: .4f} - Validation L1 Loss: {test_epoch_loss:.4f} - Train RMSE: {tm:.4f} - Validation RMSE: {val_m:.4f}")

    model.train()

    return model, train_losses, val_losses
