import torch.nn as nn


class GruNet(nn.Module):
    
    def __init__(self, hidden_dim, input_dim, output_dim, bidirectional):
        
        super(GruNet,self).__init__()
        
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        self.n_layers = num_directions
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, bidirectional = bidirectional, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)
        
    
    def forward(self, x):
        pred, _ = self.rnn(x, None)
        pred = self.fc(pred[:, -1])

        return pred
