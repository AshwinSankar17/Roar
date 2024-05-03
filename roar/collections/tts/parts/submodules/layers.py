import torch

@torch.jit.script
def swiglu(alpha, beta):
    return torch.nn.functional.silu(alpha) * beta


class SwiGLU(torch.nn.Module):
    
    def __init__(self, in_size: int, hidden_size: int):
        super(SwiGLU, self).__init__()
        self.w1 = torch.nn.Linear(in_size, hidden_size * 2)
        self.proj = torch.nn.Linear(hidden_size, in_size)
    
    def forward(self, inp: torch.Tensor):
        alpha, beta = self.w1(inp).chunk(2, dim=-1)
        out = swiglu(alpha, beta)
        return self.proj(out)