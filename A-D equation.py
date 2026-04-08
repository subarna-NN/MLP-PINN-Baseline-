import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import time
torch.cuda.synchronize()
# ======================================================
# 1. DEVICE & PHYSICS (SAME AS TRANS-PINN)
# ======================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = 0.1
nu = 0.05
pi = np.pi

print("Using device:", device)
print(f"c={c}, nu={nu}")

# ======================================================
# 2. PURE MLP (BASELINE MODEL)
# ======================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x,t], dim=-1))

# ======================================================
# 3. FDM GROUND TRUTH (SAME AS TRANS-PINN)
# ======================================================
def get_fdm_truth(c, nu, nx=512, nt=120):

    x = np.linspace(0,1,nx)
    dx = x[1]-x[0]
    t = np.linspace(0,1,nt)

    u0 = np.sin(np.pi*x[1:-1])

    def rhs(u,t):
        uf = np.concatenate(([0.0],u,[0.0]))
        u_xx = (uf[2:] - 2*uf[1:-1] + uf[:-2])/dx**2
        u_x  = (uf[2:] - uf[:-2])/(2*dx)
        return -c*u_x + nu*u_xx

    sol = odeint(rhs,u0,t)

    u = np.zeros((nt,nx))
    u[:,1:-1] = sol
    return x,t,u

# ======================================================
# 4. LOSS FUNCTION (A-D EQUATION)
# ======================================================
def compute_loss(model,
                 x_f,t_f,
                 x_ic,t_ic,u_ic,
                 x_l,t_l,x_r,t_r):

    u = model(x_f,t_f)

    u_t = torch.autograd.grad(u.sum(), t_f, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x_f, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0]

    # Linear Advection-Diffusion residual
    f = u_t + c*u_x - nu*u_xx

    mse_f = torch.mean(f**2)

    mse_ic = torch.mean((model(x_ic,t_ic)-u_ic)**2)

    mse_bc_l = torch.mean(model(x_l,t_l)**2)
    mse_bc_r = torch.mean(model(x_r,t_r)**2)

    # STANDARD baseline weights (no tricks)
    return mse_f + mse_ic + mse_bc_l + mse_bc_r

# ======================================================
# 5. DATA SETUP (MATCH TRANS-PINN)
# ======================================================
x = torch.linspace(0,1,128).to(device)
t = torch.linspace(0,1,100).to(device)

X,T = torch.meshgrid(x,t,indexing='ij')

x_f = X.reshape(-1,1).requires_grad_(True)
t_f = T.reshape(-1,1).requires_grad_(True)

# Initial condition
x_ic = torch.linspace(0,1,512).reshape(-1,1).to(device)
t_ic = torch.zeros_like(x_ic)
u_ic = torch.sin(torch.tensor(np.pi,device=device)*x_ic)

# Boundary points
t_bc = torch.linspace(0,1,200).reshape(-1,1).to(device)

x_l = torch.zeros_like(t_bc)
x_r = torch.ones_like(t_bc)

t_l = t_bc
t_r = t_bc
start_time = time.time()
# ======================================================
# 6. TRAINING
# ======================================================
model = MLP().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training MLP-PINN...")

for _ in tqdm(range(8000)):
    optimizer.zero_grad()
    loss = compute_loss(model,
                        x_f,t_f,
                        x_ic,t_ic,u_ic,
                        x_l,t_l,x_r,t_r)
    loss.backward()
    optimizer.step()

# LBFGS polish
opt_lbfgs = torch.optim.LBFGS(model.parameters(),
                              max_iter=500,
                              line_search_fn="strong_wolfe")

def closure():
    opt_lbfgs.zero_grad()
    loss = compute_loss(model,
                        x_f,t_f,
                        x_ic,t_ic,u_ic,
                        x_l,t_l,x_r,t_r)
    loss.backward()
    return loss

print("LBFGS...")
opt_lbfgs.step(closure)

# ======================================================
# 7. EVALUATION
# ======================================================
print("Evaluating against FDM...")

x_true,t_true,u_true = get_fdm_truth(c,nu)

u_pred = np.zeros_like(u_true)

model.eval()
with torch.no_grad():
    for i in range(len(t_true)):
        tt = torch.full((len(x_true),1), t_true[i], device=device)
        xx = torch.tensor(x_true,device=device).reshape(-1,1).float()
        u_pred[i,:] = model(xx,tt).cpu().numpy().flatten()

l1 = np.sum(np.abs(u_true-u_pred))/np.sum(np.abs(u_true))
l2 = np.sqrt(np.sum((u_true-u_pred)**2)/np.sum(u_true**2))

print("\nMLP-PINN A-D BASELINE")
print(f"L1={l1:.2e}, L2={l2:.2e}")

# ======================================================
# 8. PLOT (MATCH TRANS-PINN STYLE)
# ======================================================
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.imshow(u_true,aspect='auto',extent=[0,1,0,1],
           origin='lower',cmap='RdBu_r')
plt.colorbar()
plt.title("FDM Ground Truth")
plt.xlabel("t")
plt.ylabel("x")

plt.subplot(1,3,2)
plt.imshow(u_pred,aspect='auto',extent=[0,1,0,1],
           origin='lower',cmap='RdBu_r')
plt.colorbar()
plt.title("MLP-PINN")
plt.xlabel("t")

plt.subplot(1,3,3)
plt.imshow(np.abs(u_true-u_pred),aspect='auto',
           extent=[0,1,0,1],origin='lower',cmap='Reds')
plt.colorbar(label="|Error|")
plt.title("Absolute Error")
plt.xlabel("t")

plt.tight_layout()
plt.savefig("result.png", dpi=600, bbox_inches='tight')
plt.show()
print("Total Time:", time.time() - start_time, "seconds")
