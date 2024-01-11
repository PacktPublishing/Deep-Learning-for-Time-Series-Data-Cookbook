import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, seq_length):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.fc = nn.Linear(hidden_size * (seq_length - kernel_size + 1), output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # swap sequence and feature dimensions
        out = torch.relu(self.conv1(x))
        out = out.view(out.size(0), -1)  # flatten the tensor
        out = self.fc(out)
        return out


# 5 input channels, 20 output channels, 1 output feature, kernel size 3, sequence length 10
convnet = ConvNet(5, 20, 1, 3, 10)
print(convnet)

# 100 samples, 10 time steps, 5 features
X = torch.randn(100, 10, 5)
Y = torch.randn(100, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(convnet.parameters(), lr=0.01)

for epoch in range(100):
    output = convnet(X)
    loss = loss_fn(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
