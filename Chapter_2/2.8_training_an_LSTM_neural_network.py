import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # initial cell state
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # get LSTM output
        # pass last output to Fully Connected layer
        out = self.fc(out[:, -1, :])

        return out


lstm = LSTM(10, 20, 1) # 10 features, 20 hidden units, 1 output
print(lstm)

# 100 samples, 5 time steps, 10 features
X = torch.randn(100, 5, 10)
Y = torch.randn(100, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.01)

for epoch in range(100):
    output = lstm(X)
    loss = loss_fn(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')