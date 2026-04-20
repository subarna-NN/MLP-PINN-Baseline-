import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
# ======================================================
# 1. Physics & Device
# ======================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 0.01
pi = torch.tensor(np.pi, device=device)

print("Using device:", device)
print("epsilon =", eps)

# ======================================================
# 2. MLP-PINN Model (2D input)
# ======================================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=-1)
        return self.net(inp)

# ======================================================
# 3. Finite Difference Ground Truth for 2D Allen–Cahn
# ======================================================
def allen_cahn_rhs(u_flat, t_val, eps, Nx, Ny, dx, dy):

    u = u_flat.reshape(Nx, Ny)

    u[0,:] = 0.0
    u[-1,:] = 0.0
    u[:,0] = 0.0
    u[:,-1] = 0.0

    u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2

    u_xx[0,:] = u_xx[-1,:] = 0
    u_yy[:,0] = u_yy[:,-1] = 0

    du_dt = eps*(u_xx + u_yy) - u**3 + u

    du_dt[0,:] = du_dt[-1,:] = 0
    du_dt[:,0] = du_dt[:,-1] = 0

    return du_dt.reshape(-1)

def get_fdm_truth(eps, Nx=32, Ny=32, Nt=30):

    x = np.linspace(0,1,Nx)
    y = np.linspace(0,1,Ny)
    t = np.linspace(0,1,Nt)

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    X,Y = np.meshgrid(x,y,indexing='ij')
    u0 = 0.05*np.sin(np.pi*X)*np.sin(np.pi*Y)

    u0[0,:] = u0[-1,:] = 0
    u0[:,0] = u0[:,-1] = 0

    u0_flat = u0.reshape(-1)

    sol = odeint(allen_cahn_rhs, u0_flat, t, args=(eps, Nx, Ny, dx, dy))

    u = sol.reshape(Nt, Nx, Ny)

    return x, y, t, u

# ======================================================
# 4. Loss Function (Standard MLP-PINN)
# ======================================================
def compute_loss(model, x_f, y_f, t_f,
                 x_ic, y_ic, t_ic, u_ic,
                 bc_x, bc_y, bc_t):

    u = model(x_f, y_f, t_f)

    grads = torch.autograd.grad(u.sum(), [x_f, y_f, t_f], create_graph=True)

    u_x = grads[0]
    u_y = grads[1]
    u_t = grads[2]

    u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y_f, create_graph=True)[0]

    f = u_t - eps*(u_xx + u_yy) + u**3 - u
    loss_res = torch.mean(f**2)

    u_ic_pred = model(x_ic, y_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic)**2)

    u_bc_pred = model(bc_x, bc_y, bc_t)
    loss_bc = torch.mean(u_bc_pred**2)

    return loss_res + loss_ic + loss_bc

# ======================================================
# 5. Training Data
# ======================================================
Nx, Ny, Nt = 32, 32, 30

x = torch.linspace(0,1,Nx).to(device)
y = torch.linspace(0,1,Ny).to(device)
t = torch.linspace(0,1,Nt).to(device)

X,Y,T = torch.meshgrid(x,y,t, indexing="ij")

x_f = X.reshape(-1,1).requires_grad_(True)
y_f = Y.reshape(-1,1).requires_grad_(True)
t_f = T.reshape(-1,1).requires_grad_(True)

X_ic, Y_ic = torch.meshgrid(x,y,indexing="ij")

x_ic = X_ic.reshape(-1,1).to(device)
y_ic = Y_ic.reshape(-1,1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)

u_ic = 0.05*torch.sin(pi*x_ic)*torch.sin(pi*y_ic)

Nt_bc = 50
Ny_bc = 50

t_bc = torch.linspace(0,1,Nt_bc).to(device)
y_vals = torch.linspace(0,1,Ny_bc).to(device)

Yb,Tb = torch.meshgrid(y_vals,t_bc,indexing="ij")

x0 = torch.zeros_like(Yb)
x1 = torch.ones_like(Yb)

bc_x = torch.cat([x0.reshape(-1,1), x1.reshape(-1,1)], dim=0)
bc_y = torch.cat([Yb.reshape(-1,1), Yb.reshape(-1,1)], dim=0)
bc_t = torch.cat([Tb.reshape(-1,1), Tb.reshape(-1,1)], dim=0)

bc_x = bc_x.to(device)
bc_y = bc_y.to(device)
bc_t = bc_t.to(device)

# ======================================================
# 6. Training
# ======================================================
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training MLP-PINN on 2D Allen–Cahn...")

for i in tqdm(range(8000)):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, y_f, t_f,
                        x_ic, y_ic, t_ic, u_ic,
                        bc_x, bc_y, bc_t)
    loss.backward()
    optimizer.step()

print("Polishing with L-BFGS...")

optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),
                                    max_iter=1200,
                                    line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    l = compute_loss(model, x_f, y_f, t_f,
                     x_ic, y_ic, t_ic, u_ic,
                     bc_x, bc_y, bc_t)
    l.backward()
    return l

optimizer_lbfgs.step(closure)

# ======================================================
# 7. Evaluation
# ======================================================
print("Evaluating against FDM Ground Truth...")

xg, yg, tg, u_true = get_fdm_truth(eps, Nx, Ny, Nt)

upred = np.zeros_like(u_true)

model.eval()
with torch.no_grad():
    for i in range(len(tg)):
        tt = torch.full((Nx*Ny,1), tg[i], device=device)
        Xg,Yg = torch.meshgrid(torch.tensor(xg,device=device).float(),
                               torch.tensor(yg,device=device).float(),
                               indexing="ij")
        xx = Xg.reshape(-1,1)
        yy = Yg.reshape(-1,1)

        upred[i] = model(xx, yy, tt).cpu().numpy().reshape(Nx,Ny)

l1 = np.sum(np.abs(u_true-upred)) / np.sum(np.abs(u_true))
l2 = np.sqrt(np.sum((u_true-upred)**2) / np.sum(u_true**2))

print("\n--- MLP-PINN Baseline for 2D Allen–Cahn ---")
print(f"Relative L1 Error: {l1:.6e}")
print(f"Relative L2 Error: {l2:.6e}")

# ======================================================
# 8. Plotting
# ======================================================
idx = -1   # final time level

# Create meshgrid for plotting
X_plot, Y_plot = np.meshgrid(xg, yg, indexing="ij")

fig = plt.figure(figsize=(18,5))

# -------- FDM Exact (3D) --------
ax1 = fig.add_subplot(1,3,1, projection='3d')
ax1.plot_surface(X_plot, Y_plot, u_true[idx], cmap='viridis')
ax1.set_title("FDM Ground Truth")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")
ax1.grid(False)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# -------- MLP-PINN Prediction (3D) --------
ax2 = fig.add_subplot(1,3,2, projection='3d')
ax2.plot_surface(X_plot, Y_plot, upred[idx], cmap='viridis')
ax2.set_title("MLP-PINN")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("u")
ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

# -------- Absolute Error (3D) --------
ax3 = fig.add_subplot(1,3,3, projection='3d')
ax3.plot_surface(X_plot, Y_plot,
                 np.abs(u_true[idx] - upred[idx]),
                 cmap='hot')
ax3.set_title("Absolute Error")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("|Error|")
ax3.grid(False)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig("MLP_PINN_AllenCahn_3D.png", dpi=600)
plt.show()
