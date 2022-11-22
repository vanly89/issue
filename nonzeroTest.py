import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, index, x):
        y = x[index == 1.0].view(2, -1)
        return y


model = Model()
model.eval()

index = torch.zeros((2, 20), dtype=torch.float)
index[:, 2:5] = 1.0
x = torch.range(1, 40).view(2, 20)
torch.onnx.export(model,
                  (index, x),
                  'index.onnx',
                  opset_version=16,
                  input_names=['index', 'x'],
                  output_names=['y'])

y = model(index, x)
print(y)
