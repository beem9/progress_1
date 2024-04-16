import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.linspace(-10, 10, steps=200).unsqueeze(1)
y = torch.where(X > 0, torch.tensor(1.0), torch.tensor(0.0))  # 1 if positive, 0 if negative



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x



model = SimpleNN()


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


test_numbers = torch.tensor([-5, 2, 0, -3, 4.5]).unsqueeze(1)
with torch.no_grad():
    predictions = model(test_numbers)
    predicted_labels = torch.round(predictions)
    print("Predictions:", predicted_labels.flatten().numpy())
