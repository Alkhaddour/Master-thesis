import torch.nn as nn
import torch


import torch
from torch import device, nn

from einops import rearrange, repeat

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, num_classes):
        super().__init__()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x)[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
    
    @staticmethod
    def train_last_only(model):
        model.transformer = model.transformer.eval()
        model.to_latent = model.to_latent.eval()
        model.mlp_head = model.mlp_head.train()
        return model


        
    @staticmethod
    def re_init_last_layer(TransformerClassifier, out=10):
        pre_last_dim = TransformerClassifier.mlp_head[1].in_features
        TransformerClassifier.mlp_head[1] = nn.Linear(pre_last_dim, out)
        torch.nn.init.xavier_uniform_(TransformerClassifier.mlp_head[1].weight)
        return TransformerClassifier

class TransformerSED(nn.Module):
    def __init__(self, n_mels, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pos_embedding = nn.Parameter(torch.randn(1, n_mels+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


    @staticmethod
    def re_init_last_layer(TransformerSED, out=10):
        pre_last_dim = TransformerSED.mlp_head[1].in_features
        TransformerSED.mlp_head[1] = nn.Linear(pre_last_dim, out)
        torch.nn.init.xavier_uniform_(TransformerSED.mlp_head[1].weight)
        return TransformerSED

if __name__ == '__main__':
    # n_mels=128
    # n_feat = 256
    # dim=n_feat
    # inp = torch.randn(1, n_mels, n_feat)
    # model = TransformerSED(n_mels, num_classes=2, dim=dim, depth=2, heads=12, mlp_dim=512)
    # model = TransformerSED.re_init_last_layer(model,out=10)
    # print(model)
    # print(model(inp))
    device = 'cuda:1'
    clf = TransformerClassifier(dim=512, depth=2, heads=4, dim_head=64, mlp_dim=1024, dropout=0.0, num_classes=10).to(device)
    inp = torch.randn(3, 512).to(device)
    out = clf(inp)
    print(out.shape)
    clf = TransformerClassifier.re_init_last_layer(clf, 5).to(device)
    out = clf(inp)
    print(out.shape)


