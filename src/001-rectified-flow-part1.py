import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

X, _ = make_moons(n_samples=2000, noise=0.05)
plt.scatter(X[:,0], X[:,1])
plt.title("Real Two Moons Data")
plt.show()


class TimeConditionedMLP(nn.Module):
    """
    A simple time-conditioned MLP that represents
    the velocity field v_theta(x_t, t)
    """
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()

        # 输入是 [x_t, t]，因此维度是 data_dim + 1
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t, t):
        """
        x_t : [batch_size, data_dim]
        t   : [batch_size, 1]
        """
        # 拼接时间条件
        model_input = torch.cat([x_t, t], dim=-1)
        return self.net(model_input)


num_epochs = 100
batch_size = 128

dataset = torch.utils.data.TensorDataset(
    torch.tensor(X, dtype=torch.float)
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

model = TimeConditionedMLP().cuda()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

model.train()
# 用于调整时间采样分布（靠近 t=0 更多）
time_shift = 0.1
for epoch in range(num_epochs):
    for (x_data,) in dataloader:

        x_data = x_data.cuda() # [bs, 2]

        # 1. Sample time t from logit-normal distribution
        t = torch.randn(x_data.shape[0], 1).cuda()
        t = torch.sigmoid(t)

        # 时间重参数化（bias towards small t）
        t = time_shift * t / (1 + (time_shift - 1) * t)

        # 2. Sample noise x_1 ~ N(0, I)
        x_noise = torch.randn_like(x_data)

        # 3. Interpolation: x_t = (1 - t) x_0 + t x_1
        x_t = (1 - t) * x_data + t * x_noise

        # 4. Predict velocity field
        v_pred = model(x_t, t)

        # 5. Flow Matching objective
        # v*(x_t, t) = x_1 - x_0
        loss = F.mse_loss(v_pred, x_noise - x_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.6f}")



model.eval()

num_steps = 100
num_samples = 2000

# 初始点：纯噪声
x = torch.randn(num_samples, 2).cuda()

# 从 t=1 → t=0
time_grid = torch.linspace(1, 0, num_steps + 1).cuda()

with torch.no_grad():
    for i in range(num_steps):
        t = time_grid[i]
        t_next = time_grid[i + 1]

        # 预测当前速度场
        v = model(x, t.repeat(x.shape[0], 1))

        # Euler integration
        x = x - v * (t - t_next)

samples = x.cpu().numpy()
plt.scatter(samples[:, 0], samples[:, 1])
plt.title("Generated Samples (Rectified Flow)")
plt.show()
