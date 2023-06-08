import torch
from s4 import S4Block


batch, length, input_channels = 20, 501, 126
hidden_channels = 32
inp = torch.rand(batch, length, input_channels)

model = S4Block(d_model=126, final_act='id', transposed=False, dropout=0.1)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
outp1, outp2 = model.forward(inp)
print(outp1.shape, outp2)