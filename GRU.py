import torch
import torch.nn as nn
from torchinfo import summary

class EncoderGRU(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: int = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_size, 
                          num_layers = num_layers, batch_first = True, dropout = dropout)
        self.num_layers = num_layers
        self.print_shapes = True 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)

        if self.print_shapes:
            print(f"\nEncoder - Input shape: {x.shape}")
            print(f"Encoder - Embedded shape: {embedded.shape}")
            print(f"Encoder - Outputs shape: {outputs.shape}")
            print(f"Encoder - Hidden shape: {hidden.shape}\n")
            self.print_shapes = False
        return outputs, hidden


class DecoderGRU(nn.Module):
    def __init__(self, output_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: int = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_size, 
                          num_layers = num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.print_shapes = True 

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = self.fc(outputs[:, -1, :])

        if self.print_shapes:
            print(f"\nDecoder - Input shape: {x.shape}")
            print(f"Decoder - Embedded shape: {embedded.shape}")
            print(f"Decoder - Outputs shape (después de proyección): {outputs.shape}")
            print(f"Decoder - Hidden shape: {hidden.shape}\n")
            self.print_shapes = False
        return outputs, hidden


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder: EncoderGRU, decoder: DecoderGRU):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor = None, train: bool = True) -> torch.Tensor:
        batch_size = src.size(0)
        output_size = self.decoder.fc.out_features

        # Codificación
        _, hidden = self.encoder(src)

        # Preparar tensor de salidas
        if trg is not None:
            trg_length = trg.size(1)
        else:
            trg_length = 50  # Longitud máxima en predicción

        outputs = torch.zeros(batch_size, trg_length, output_size).to(src.device)

        # Inicializar entrada al Decoder. REEMPLAZAR 2 POR EL ÍNDICE DEL TOKEN ESPECIAL CON EL QUE COMIENZA LA SECUENCIA. POR EJEMPLO <sos>
        input = trg[:, 0].unsqueeze(1) if trg is not None else torch.tensor([[2]] * batch_size).to(src.device)

        # Decodificación
        for t in range(1, trg_length):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output

            if train and trg is not None:
                # Usar siempre Teacher Forcing en entrenamiento
                input = trg[:, t].unsqueeze(1)
            else:
                # Autoregresión en validación y prueba
                input = output.argmax(1).unsqueeze(1)

        return outputs
    
    def count_parameters(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        fc_params = sum(p.numel() for p in self.decoder.fc.parameters())
        total_params = encoder_params + decoder_params
        print(f"\nParámetros del Encoder: {encoder_params:,}".replace(",", "."))
        print(f"Parámetros del Decoder: {decoder_params:,}".replace(",", "."))
        print(f"Parámetros de la Proyección Lineal (Linear Projection): {fc_params:,}".replace(",", "."))
        print(f"Parámetros totales: {total_params:,}\n".replace(",", "."))


if __name__ == "__main__":
    input_size = 10000
    output_size = 1000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    batch_size = 3
    src_length = 6
    trg_length = 6
    dropout = 0.2

    # Inicializar Encoder, Decoder y modelo Seq2Seq
    encoder = EncoderGRU(input_size, embedding_dim, hidden_size, num_layers, dropout)
    decoder = DecoderGRU(output_size, embedding_dim, hidden_size, num_layers, dropout)
    model = Seq2SeqGRU(encoder, decoder)

    # Mostrar parámetros del modelo
    model.count_parameters()

    # Dummy data
    src = torch.randint(0, input_size, (batch_size, src_length))
    trg = torch.randint(0, output_size, (batch_size, trg_length))

    # Entrenamiento
    print("\n--- Entrenamiento ---\n")
    outputs_train = model(src, trg, train = True)
    print("Dimensión final de las predicciones (entrenamiento):", outputs_train.shape)

    # Validación
    print("\n--- Validación ---\n")
    outputs_val = model(src, trg, train = False)
    print("Dimensión final de las predicciones (validación):", outputs_val.shape)

    # Prueba
    print("\n--- Prueba ---\n")
    outputs_test = model(src, train = False)
    print("Dimensión final de las predicciones (prueba):", outputs_test.shape)

    # Mostrar detalles del modelo con torchinfo
    summary(
        model,
        input_data = (src, trg, True),  # Cambiamos "mode" a True para entrenamiento
        col_names = ["input_size", "output_size", "num_params", "trainable"],
        dtypes = [torch.int64, torch.int64, torch.bool],  # `train` ahora es booleano
    )
