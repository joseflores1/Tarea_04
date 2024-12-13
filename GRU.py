import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_len=10003, embedding_dim=256, hidden_dim=512, n_layers=2, dropout_prob=0.3):
        super().__init__()
 
        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
 
    def forward(self, input_batch):
        embed = self.dropout(self.embedding(input_batch))
        outputs, hidden = self.rnn(embed)
        return hidden
    
class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim=1002, embedding_dim=256, hidden_dim=512, n_layers=2, dropout_prob=0.3):
        super().__init__()
        self.input_output_dim = input_output_dim
 
        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_output_dim)
        self.dropout = nn.Dropout(dropout_prob)
 
    def forward(self, target_token, hidden):
        target_token = target_token.unsqueeze(1)  
        embedding_layer = self.dropout(self.embedding(target_token))
        output, hidden = self.rnn(embedding_layer, hidden)
 
        linear = self.fc(output.squeeze(1))
        return linear, hidden
    

class Decoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device
 
    def forward(self, target, hidden):
        target_len, batch_size = target.shape[1], target.shape[0]
        target_vocab_size = self.one_step_decoder.input_output_dim
        predictions = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        input = target[:, 0]  
        
        for t in range(1, target_len):
            predict, hidden = self.one_step_decoder(input, hidden)
            predictions[:, t, :] = predict
            input = predict.argmax(1)        
        
        return predictions

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder        
 
    def forward(self, source, target):
        hidden = self.encoder(source)
        outputs = self.decoder(target, hidden)
        return outputs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    input_vocab_size = 10003
    output_vocab_size = 1002
    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout_prob = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(input_vocab_size, embedding_dim, hidden_dim, n_layers, dropout_prob)
    one_step_decoder = OneStepDecoder(output_vocab_size, embedding_dim, hidden_dim, n_layers, dropout_prob)
    decoder = Decoder(one_step_decoder, device)
    model = EncoderDecoder(encoder, decoder).to(device)

    print(f"Total parameters: {count_parameters(model)}")
    print(f"Encoder parameters: {count_parameters(encoder)}")
    print(f"Decoder parameters: {count_parameters(decoder)}")