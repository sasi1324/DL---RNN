# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 
Convert data to tensors and set up DataLoader.



### STEP 3: 
Define the RNN model architecture.



### STEP 4: 
Summarize, compile with loss and optimizer



### STEP 5: 
Train the model with loss tracking.



### STEP 6:
Predict on test data, plot actual vs. predicted prices.






## PROGRAM

### Name:SASINTHARA S

### Register Number:212223110045

```python
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
df_train = pd.read_csv('/content/drive/MyDrive/19AI413- Deep Learning -20251029T030542Z-1-001/19AI413- Deep Learning/Lab Exp 2025/Exp-05/trainset.csv')
df_test = pd.read_csv('/content/drive/MyDrive/19AI413- Deep Learning -20251029T030542Z-1-001/19AI413- Deep Learning/Lab Exp 2025/Exp-05/testset.csv')
# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)
# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class RNNModel(nn.Module):
    # write your code here
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take the last output
        out = self.fc(out)
        return out
from torchinfo import summary
summary(model, input_size=(64, 60, 1))
model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
num_epochs = 50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) # Move model to the selected device


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    train_losses.append(average_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="775" height="287" alt="image" src="https://github.com/user-attachments/assets/a0de4bb5-a7ac-4757-a9c7-3f8f13d86069" />

<img width="407" height="735" alt="image" src="https://github.com/user-attachments/assets/6c423968-a25e-45b2-84c7-e6bc6650e308" />

## True Stock Price, Predicted Stock Price vs time
<img width="562" height="402" alt="image" src="https://github.com/user-attachments/assets/205623cc-681c-4419-b0aa-42cb44c2266c" />

### Predictions
<img width="748" height="507" alt="image" src="https://github.com/user-attachments/assets/54189d11-0bb1-49da-9f16-bf094864db32" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
