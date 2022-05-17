import torch
import torch.nn as nn
from torchsummary import summary


class DenseN(nn.Module):
    def __init__(self, in_dim, n_hidden_layers, hidden_dims, out_dim, dropout = 0., return_logits=True):
        super().__init__()
        self.in_dim = in_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout
        self.return_logits = return_logits

        assert n_hidden_layers >= 0, "Number of hidden layers cannot be negative number"
        if n_hidden_layers == 0:
            self.net = nn.ModuleList[nn.Linear(self.in_dim, self.out_dim)]
        else:
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]*n_hidden_layers
            if isinstance(dropout, float):
                dropout = [dropout]*n_hidden_layers
            assert n_hidden_layers == len(hidden_dims)    
            layers = []    
            # first layer
            layers.append(nn.Linear(self.in_dim, hidden_dims[0])) 
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout[0]))
            # hidden layers
            for i in range(self.n_hidden_layers-1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1])) 
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout[i+1]))
            # output layer
            layers.append(nn.Linear(hidden_dims[-1], self.out_dim))
            self.net = nn.ModuleList(layers)
            

    def forward(self, x):
        for l in self.net:
            x = l(x)
        if self.return_logits == True:
            return x
        else:
            return nn.Softmax(dim=1)(x)
    @staticmethod
    def re_init_last_layer(DenseN, out=10):
        pre_last_layer_size = DenseN.net[-1].in_features
        DenseN.net[-1] = nn.Linear(pre_last_layer_size, out)
        torch.nn.init.xavier_uniform_(DenseN.net[-1].weight)
        return DenseN
        

if __name__ == "__main__":
    dense_net = DenseN(in_dim=512, n_hidden_layers=2, hidden_dims=[1024,512], out_dim=35, dropout =0.1, return_logits=True).to('cuda:0')
    input = torch.rand(10, 512).to('cuda:0')
    logits = dense_net(input)
    print(logits.shape)
    dense_net = DenseN.re_init_last_layer(dense_net, 5).to('cuda:0')
    print(summary(dense_net, (10, 512)))
    logits = dense_net(input)
    print(logits.shape)

