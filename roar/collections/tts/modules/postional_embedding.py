import torch
import torch.nn as nn


# Implements sinusoidal positional encoding
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        return self._forward(pos_seq, bsz)

    def _forward(self, pos_seq, bsz=None):
        #        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        sinusoid_inp = torch.matmul(
            torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0)
        )

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].repeat(bsz, 1, 1)
        else:
            return pos_emb[None, :, :]


class RelativePositionalEncoding(PositionalEmbedding):
    def __init__(self, demb):
        super(RelativePositionalEncoding, self).__init__(demb)

    def forward(self, pos_seq, bsz=None):
        pass
