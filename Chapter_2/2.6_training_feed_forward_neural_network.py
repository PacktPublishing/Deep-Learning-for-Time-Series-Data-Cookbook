import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 10 input units, 5 units in the hidden layer
        self.fc1 = nn.Linear(10, 5)
        # 5 units in the hidden layer, 1 output unit
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
print(net)

# Create a synthetic dataset
X = torch.randn(100, 10)
Y = torch.randn(100, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(100):
    # Forward pass: compute predicted outputs by passing
    # inputs to the model
    output = net(X)
    # Compute loss
    loss = loss_fn(output, Y)
    # Zero the gradients before running the backward pass
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss
    # with respect to model parameters
    loss.backward()
    # Calling the step function on an Optimizer performs
    # an update on its parameters
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
