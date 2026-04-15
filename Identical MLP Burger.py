import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

nu = 0.01 / np.pi
print(f"nu     : {nu:.6f}")

# ------------------------------------------------------------------
# Model architecture  (ORIGINAL — unchanged)
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),  nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

model = MLP().to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------------
# Training data  (EQUALIZED)
# ------------------------------------------------------------------
N_RES = 10_000

# Collocation points — random sampling, fixed throughout training
def new_colloc():
    pts = torch.rand(N_RES, 2, device=device)
    pts[:, 0] = pts[:, 0] * 2.0 - 1.0   # x: [-1, 1]
    # pts[:, 1] stays in [0, 1]           # t: [0, 1]
    return pts

col = new_colloc()
x_f = col[:, 0:1].requires_grad_(True)
t_f = col[:, 1:2].requires_grad_(True)

# IC  (t = 0)
x_ic = torch.linspace(-1, 1, 512).reshape(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# BC  (x = ±1, all t)
t_bc = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
x_left  = -torch.ones_like(t_bc)
x_right =  torch.ones_like(t_bc)

# ------------------------------------------------------------------
# Loss function  (EQUALIZED — all weights = 1.0)
# ------------------------------------------------------------------
def compute_loss():
    u    = model(x_f, t_f)
    u_t  = torch.autograd.grad(u.sum(),  t_f, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(),  x_f, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0]

    residual = u_t + u * u_x - nu * u_xx
    loss_res = torch.mean(residual ** 2)

    loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)

    loss_bc = (torch.mean(model(x_left,  t_bc) ** 2) +
               torch.mean(model(x_right, t_bc) ** 2))

    # Equal weights = 1.0 for all terms
    return loss_res + loss_ic + loss_bc

# ------------------------------------------------------------------
# Training  (EQUALIZED)
# ------------------------------------------------------------------
start_time = time.time()

# Stage 1 : Adam  — 10,000 steps, lr=1e-3, no scheduler
print("\nStage 1/2 — Adam (10,000 steps) ...")
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in tqdm(range(10_000)):
    optimizer_adam.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer_adam.step()
    if (step + 1) % 2000 == 0:
        tqdm.write(f"  [{step+1:5d}] loss = {loss.item():.4e}")

# Stage 2 : L-BFGS — max_iter=5000, strong_wolfe
print("\nStage 2/2 — L-BFGS (max_iter=5000) ...")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), max_iter=5000,
    line_search_fn="strong_wolfe"
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print(f"  L-BFGS done.")

total_time = time.time() - start_time
print(f"\nTotal training time : {total_time:.1f} s")

# ------------------------------------------------------------------
# FDM ground truth
# ------------------------------------------------------------------
def get_fdm_truth(nu_val, nx=512, nt=200):
    xg  = np.linspace(-1, 1, nx)
    dxg = xg[1] - xg[0]
    tg  = np.linspace(0, 1, nt)
    u0  = -np.sin(np.pi * xg[1:-1])

    def rhs(u, t_val):
        uf   = np.concatenate(([0.0], u, [0.0]))
        u_xx = (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dxg**2
        u_x  = (uf[2:] - uf[:-2]) / (2 * dxg)
        return -uf[1:-1] * u_x + nu_val * u_xx

    sol = odeint(rhs, u0, tg)
    u   = np.zeros((nt, nx))
    u[:, 1:-1] = sol
    return xg, tg, u

x_true, t_true, u_true = get_fdm_truth(nu)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
u_pred = np.zeros_like(u_true)
model.eval()
with torch.no_grad():
    for i in range(len(t_true)):
        tt = torch.full((len(x_true), 1), t_true[i], device=device)
        xx = torch.tensor(x_true, dtype=torch.float32,
                          device=device).reshape(-1, 1)
        u_pred[i] = model(xx, tt).cpu().numpy().flatten()

rl1 = np.sum(np.abs(u_true - u_pred)) / (np.sum(np.abs(u_true)) + 1e-12)
rl2 = np.sqrt(np.sum((u_true - u_pred)**2) /
              (np.sum(u_true**2) + 1e-12))

print(f"\n{'='*50}")
print(f"  MLP-PINN  —  FAIR COMPARISON RESULT")
print(f"  Relative L1 Error : {rl1:.6f}  ({rl1*100:.4f}%)")
print(f"  Relative L2 Error : {rl2:.6f}  ({rl2*100:.4f}%)")
print(f"{'='*50}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kw = dict(aspect='auto', extent=[0, 1, -1, 1], origin='lower', cmap='turbo')

im0 = axes[0].imshow(u_true, **kw)
axes[0].set_title("FDM Ground Truth"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u_pred, **kw)
axes[1].set_title("MLP-PINN (Fair)"); axes[1].set_xlabel("t")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(u_true - u_pred),
                     aspect='auto', extent=[0, 1, -1, 1],
                     origin='lower', cmap='inferno')
axes[2].set_title("Absolute Error"); axes[2].set_xlabel("t")
plt.colorbar(im2, ax=axes[2], label="|Error|")

plt.suptitle(f"", fontsize=11)
plt.tight_layout()
plt.savefig("fair_MLP_PINN_result.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: fair_MLP_PINN_result.png")
