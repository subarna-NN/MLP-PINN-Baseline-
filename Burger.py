import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import time
torch.cuda.synchronize()
# --- 1. Physics & Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nu = 0.01 / np.pi

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

# --- 2. RIGOROUS Ground Truth (Finite Difference) ---
def burgers_rhs(u, t, k, nu, dx):
    # Finite Difference discretization for u_t = -u*u_x + nu*u_xx
    # u is vector of size N-2 (internal points)
    # Boundary conditions u[0]=0, u[-1]=0 are enforced implicitly
    
    # Pad with boundaries (0.0)
    u_full = np.concatenate(([0.0], u, [0.0]))
    
    # Central Difference for Viscosity (2nd derivative)
    u_xx = (u_full[2:] - 2*u_full[1:-1] + u_full[:-2]) / dx**2
    
    # Central Difference for Advection (1st derivative)
    # Note: For shocks, Upwind is often safer, but Central is fine for ground truth 
    # if grid is fine enough (N=512 is sufficient here).
    u_x = (u_full[2:] - u_full[:-2]) / (2*dx)
    
    du_dt = -u_full[1:-1] * u_x + nu * u_xx
    return du_dt

def get_fdm_truth(nu, nx=512, nt=200):
    """
    Finite Difference Solver using scipy.integrate.odeint.
    This respects the Dirichlet BCs: u(-1)=0, u(1)=0.
    """
    x = np.linspace(-1, 1, nx)
    dx = x[1] - x[0]
    t = np.linspace(0, 1, nt)
    
    # Initial Condition: -sin(pi*x)
    u0 = -np.sin(np.pi * x[1:-1]) # Internal points only
    
    # Solve ODE system
    sol = odeint(burgers_rhs, u0, t, args=(nx, nu, dx))
    
    # Add boundaries back (0.0)
    u_final = np.zeros((nt, nx))
    u_final[:, 1:-1] = sol
    u_final[:, 0] = 0.0
    u_final[:, -1] = 0.0
    
    return x, t, u_final

# --- 3. Loss Function (Standardized) ---
def compute_loss(model, x_f, t_f, x_ic, t_ic, u_ic, x_left, t_left, x_right, t_right):
    # PDE Residual
    u = model(x_f, t_f)
    u_t = torch.autograd.grad(u.sum(), t_f, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x_f, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    mse_f = torch.mean(f**2)
    
    # Initial Condition Loss
    mse_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
    
    # Boundary Condition Loss (Continuous Time)
    mse_bc_left = torch.mean(model(x_left, t_left)**2)
    mse_bc_right = torch.mean(model(x_right, t_right)**2)
    
    # NO MAGIC WEIGHTS (Standard 1.0)
    # If MLP fails here, it proves MLP is weak (Good for your paper!)
    return mse_f + mse_ic + mse_bc_left + mse_bc_right

# --- 4. Data Setup (Continuous Boundaries) ---
# Residual Points (Grid)
x = torch.linspace(-1, 1, 128).to(device)
t = torch.linspace(0, 1, 100).to(device)
X, T = torch.meshgrid(x, t, indexing='ij')
x_f = X.reshape(-1, 1).requires_grad_(True)
t_f = T.reshape(-1, 1).requires_grad_(True)

# IC Points (t=0)
x_ic = torch.linspace(-1, 1, 512).reshape(-1, 1).to(device)
u_ic = -torch.sin(np.pi * x_ic)
t_ic = torch.zeros_like(x_ic)

# BC Points (Fixed Walls for all t)
t_bc_vals = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
x_left = torch.ones_like(t_bc_vals) * -1.0
x_right = torch.ones_like(t_bc_vals) * 1.0
t_left = t_bc_vals
t_right = t_bc_vals
start_time = time.time()
# --- 5. Training ---
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training MLP-PINN (Corrected Setup)...")
# Adam Stage
for i in tqdm(range(10000)):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, t_f, x_ic, t_ic, u_ic, x_left, t_left, x_right, t_right)
    loss.backward()
    optimizer.step()

# L-BFGS Stage (Standard polish)
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=5000, line_search_fn="strong_wolfe")
def closure():
    optimizer_lbfgs.zero_grad()
    loss = compute_loss(model, x_f, t_f, x_ic, t_ic, u_ic, x_left, t_left, x_right, t_right)
    loss.backward()
    return loss
optimizer_lbfgs.step(closure)

# --- 6. Evaluation & Plotting ---
print("Evaluating against FDM Ground Truth...")
x_true, t_true, u_true = get_fdm_truth(nu) # Now using the correct Finite Difference solver

u_pred = np.zeros_like(u_true)
model.eval()
with torch.no_grad():
    for i in range(len(t_true)):
        t_v = torch.full((len(x_true), 1), t_true[i], device=device)
        x_v = torch.tensor(x_true, device=device).reshape(-1, 1).float()
        u_pred[i, :] = model(x_v, t_v).cpu().numpy().flatten()

rl1 = np.sum(np.abs(u_true - u_pred)) / np.sum(np.abs(u_true))
rl2 = np.sqrt(np.sum((u_true - u_pred)**2) / np.sum(u_true**2))

print(f"\n--- RIGOROUS MLP-PINN BASELINE ---")
print(f"Relative L1 Error: {rl1:.6f}")
print(f"Relative L2 Error: {rl2:.6f}")
print("(Note: If this error is higher than before (e.g. >10%), it confirms standard MLP is weak.)")

# Plotting
plt.figure(figsize=(12, 4))

# common color scale for first two plots
vmin = min(u_true.min(), u_pred.min())
vmax = max(u_true.max(), u_pred.max())

plt.subplot(1, 3, 1)
plt.title("FDM Ground Truth")
plt.imshow(u_true, aspect='auto', extent=[-1,1,1,0],
           cmap='RdBu_r', vmin=vmin, vmax=vmax)   # ✅ changed
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("MLP-PINN")
plt.imshow(u_pred, aspect='auto', extent=[-1,1,1,0],
           cmap='RdBu_r', vmin=vmin, vmax=vmax)   # ✅ changed
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Absolute Error")
plt.imshow(np.abs(u_true - u_pred), aspect='auto',
           extent=[-1,1,1,0], cmap='Reds')        # ✅ better for error
plt.colorbar()

plt.tight_layout()
plt.savefig("result.png", dpi=600, bbox_inches='tight')
plt.show()

print("Total Time:", time.time() - start_time, "seconds")
