import torch
import torch.nn as nn


class ResNet1DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, stride=1, bias=False):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding="same"),
            nn.InstanceNorm1d(embed_dim),
            nn.ELU(inplace=True),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding="same"),
            nn.InstanceNorm1d(embed_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.conv_net(x)
        x = x + residual

        return x

class ResNet1D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet1DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class RibosomeLoadingPredictionHead(nn.Module):
    def __init__(self, c_in, embed_dim, num_blocks, dropout=0.2):
        super().__init__()

        self.linear_in = nn.Linear(c_in, embed_dim)

        self.resnet = ResNet1D(embed_dim, num_blocks)
        self.dropout = nn.Dropout(p=dropout)

        self.linear_out = nn.Linear(embed_dim, 1)

    def forward(self, x, padding_mask=None):
        x = self.linear_in(x)

        x = x.permute(0, 2, 1) # B x L x E => B x E x L
        x = self.resnet(x)
        x = x.permute(0, 2, 1) # B x E x L => B x L x E

        # Global pooling (B x L x E => B x E)
        if padding_mask is not None:
            x[padding_mask, :] = 0.0
            x = x.sum(dim=-2) / (~padding_mask).sum(dim=-1)[:, None]
        else:
            x = x.mean(dim=-2)

        x = self.dropout(x)
        x = self.linear_out(x).squeeze(-1)

        return x