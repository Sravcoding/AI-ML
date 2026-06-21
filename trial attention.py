import torch
import torch.nn as nn
import random

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data Generation (Increased samples for better digit recognition)
A_vals, B_vals, targets, data = [], [], [], []
for i in range(1000):
    a = random.randint(1, 49)
    b = random.randint(1, 49)
    
    u_sum = (a % 10 + b % 10) % 10
    t_sum = (a // 10 + b // 10) % 10
    carry = 1 if (a % 10 + b % 10) >= 10 else 0
    
    A_vals.append(a); B_vals.append(b)
    targets.append([float(t_sum), float(u_sum), float(carry)])
    data.append([[0, a%10, 1], [carry, a//10, -1], [0, b%10, 1], [carry, b//10, -1], [0, 0, 0]])

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
target_tensor = torch.tensor(targets, dtype=torch.float32).to(device)

# 3. Model
class AdditionAttention(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        K, Q, V = self.Wk(x), self.Wq(x), self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1)**0.5)
        weights = self.softmax(scores)
        context = torch.matmul(weights, V)
        return context[:, 4, :] # Extract the 0,0,0 layer result

# 4. Training
model = AdditionAttention().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

for epoch in range(1201):
    optimizer.zero_grad()
    preds = model(data_tensor)
    loss = criterion(preds, target_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 5. Final Display
print("\n--- Final Learned Weights ---")
print("Wv (Value Matrix):\n", model.Wv.weight.data)
print("Wk (Key Matrix):\n", model.Wk.weight.data)
print("Wq (Query Matrix):\n", model.Wq.weight.data)

print("\n--- Final Sum Examples ---")
model.eval()
with torch.no_grad():
    for i in range(5):
        sample = data_tensor[i].unsqueeze(0)
        # Get the 3 output components
        t_out, u_out, c_out = model(sample).squeeze().cpu().tolist()
        
        # Reconstruct the final sum
        # (Tens * 10) + Units + (Carry * 10)
        final_sum = (round(t_out) * 10) + round(u_out) + (round(c_out) * 10)
        
        print(f"Input: {A_vals[i]} + {B_vals[i]} | Model Predicted Sum: {final_sum} | Actual: {A_vals[i]+B_vals[i]}")