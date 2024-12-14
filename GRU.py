import torch
import torch.nn as nn
from torchinfo import summary

class EncoderGRU(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: int = 0.3) -> None:
        """
        Encoder basado en GRU con soporte para múltiples capas.

        Parámetros:
            input_size (int): Tamaño del vocabulario de entrada.
            embedding_dim (int): Dimensión de los embeddings de entrada.
            hidden_size (int): Tamaño del estado oculto de la GRU.
            num_layers (int): Número de capas de la GRU.
            dropout (float): Tasa de dropout aplicada entre capas GRU.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_size, 
                          num_layers = num_layers, batch_first = True, dropout = dropout)
        self.num_layers = num_layers
        self.print_shapes = True 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del Encoder.

        Parámetros:
            x (torch.Tensor): Entrada con forma [batch_size, seq_length].

        Retorna:
            outputs (torch.Tensor): Salidas de cada paso de tiempo [batch_size, seq_length, hidden_size].
            hidden (torch.Tensor): Últimos estados ocultos [num_layers, batch_size, hidden_size].
        """
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
    def __init__(self, output_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: int = 0.3):
        """
        Decoder basado en GRU con soporte para múltiples capas.

        Parámetros:
            output_size (int): Tamaño del vocabulario de salida.
            embedding_dim (int): Dimensión de los embeddings de salida.
            hidden_size (int): Tamaño del estado oculto de la GRU.
            num_layers (int): Número de capas de la GRU.
            dropout (float): Tasa de dropout aplicada entre capas GRU.
        """
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_size, 
                          num_layers = num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.print_shapes = True 

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del Decoder.

        Parámetros:
            x (torch.Tensor): Entrada con forma [batch_size, 1].
            hidden (torch.Tensor): Estados ocultos iniciales [num_layers, batch_size, hidden_size].

        Retorna:
            outputs (torch.Tensor): Predicción de la próxima palabra [batch_size, output_size].
            hidden (torch.Tensor): Nuevos estados ocultos [num_layers, batch_size, hidden_size].
        """
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
        """
        Modelo Seq2Seq con Encoder y Decoder basados en GRU con múltiples capas.

        Parámetros:
            encoder (EncoderGRU): El Encoder.
            decoder (DecoderGRU): El Decoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor = None, train: bool = True) -> torch.Tensor:
        """
        Paso hacia adelante del modelo completo.

        Parámetros:
            src (torch.Tensor): Secuencia de entrada [batch_size, src_length].
            trg (torch.Tensor, opcional): Secuencia de salida esperada [batch_size, trg_length].
            train (bool): Indica si el modelo está en modo entrenamiento (True) o validación/prueba (False).

        Retorna:
            outputs (torch.Tensor): Predicciones para cada paso de la secuencia [batch_size, trg_length, output_size].
        """
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
        """
        Calcula y muestra el número de parámetros del Encoder, Decoder,
        y la proyección lineal (Linear Projection) del Decoder.
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        fc_params = sum(p.numel() for p in self.decoder.fc.parameters())
        total_params = encoder_params + decoder_params
        print(f"\nParámetros del Encoder: {encoder_params:,}".replace(",", "."))
        print(f"Parámetros del Decoder: {decoder_params:,}".replace(",", "."))
        print(f"Parámetros de la Proyección Lineal (Linear Projection): {fc_params:,}".replace(",", "."))
        print(f"Parámetros totales: {total_params:,}\n".replace(",", "."))


if __name__ == "__main__":
    # Dimensiones
    input_size = 10000 + 3
    output_size = 1000 + 2
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    batch_size = 3
    src_length = 6
    trg_length = 6
    dropout = 0.3

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
    # print("\n--- Entrenamiento ---\n")
    outputs_train = model(src, trg, train = True)
  
    print("Dimensión final de las predicciones (entrenamiento):", outputs_train.shape)
    # Validación
    # print("\n--- Validación ---\n")
    outputs_val = model(src, trg, train = False)
    print("Dimensión final de las predicciones (validación):", outputs_val.shape)

    # Prueba
    # print("\n--- Prueba ---\n")
    outputs_test = model(src, train = False)
    print("Dimensión final de las predicciones (prueba):", outputs_test.shape)

    # Mostrar detalles del modelo con torchinfo
    summary(
        model,
        input_data = (src, trg, True),
        col_names = ["input_size", "output_size", "num_params", "trainable"],
        dtypes = [torch.int64, torch.int64, torch.bool],
    )
