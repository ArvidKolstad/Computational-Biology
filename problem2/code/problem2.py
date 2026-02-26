import numpy as np
import matplotlib.pyplot as plt

class BelousovDiffusionModel:
    def __init__(self,
        L: int, a: float, b: float, Du: float, Dv: float, dt: float,
    ) -> None:
        self.L: int = L
        self.a: float = a
        self.b: float = b
        self.Du: float = Du
        self.Dv: float = Dv
        self.dt: float = dt
        
        self.h = 1 # Step size in grid
        
        self.init_grid() 
        
    def init_grid(self) -> None:
        initial_state = self.analytic_stable_state()
        
        initial_grid = np.ones(shape=(self.L, self.L, 2)) * initial_state[None, None, :]
        
        perturbation = np.zeros(initial_grid.shape)

        bound1 = initial_state[0]*0.1
        bound2 = initial_state[1]*0.1
        
        perturbation[:, :, 0] = np.random.uniform(-bound1, bound1, size=(self.L, self.L))
        perturbation[:, :, 1] = np.random.uniform(-bound2, bound2, size=(self.L, self.L))
        
        self.state = initial_grid + perturbation
        
    def analytic_stable_state(self) -> np.ndarray:
        """
        Analytically derived solution when diffusion term is neglected
        """
        return np.array([self.a, self.b / self.a])
    
    def calc_laplace_discrete(self, state: np.ndarray) -> np.ndarray:
        """
        Laplace discretation:
        
        Nabla^2(u) = (u_(i+1,j) + u_(i-1,j) + u_(i,j+1) + u_(i,j-1) - 4u_(i, j)) / eps^2
        
        np.roll() ensures periodic boundaries
        """
        return (
            (np.roll(state, 1, axis=0) + np.roll(state, -1, axis=0) + 
            np.roll(state, 1, axis=1) + np.roll(state, -1, axis=1) - 
            4*state) / self.h**2
        )
        
    def F(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Dynamics for u without diffusion
        """
        return self.a - (self.b + 1)*u + u**2*v
    
    def G(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Dynamics for v without diffusion
        """
        return self.b*u - u**2*v
    
    def update(self) -> None:
        """
        Euler explicit numerical integration:
        
        y_(i+1) = y_i + dt*f'(t, y_i)
        """
        u = self.state[:, :, 0]
        v = self.state[:, :, 1]
             
        Lu = self.calc_laplace_discrete(u)
        Lv = self.calc_laplace_discrete(v)
        
        u_new = u + self.dt*(self.F(u, v) + self.Du*Lu)
        v_new = v + self.dt*(self.G(u, v) + self.Dv*Lv)
        
        self.state[:, :, 0] = u_new
        self.state[:, :, 1] = v_new

# Constants
L = 128
A = 3
B = 8
DU = 1
dt = 0.01

Dvs = [2.3, 3, 5, 9]

N_tot = 50000
N_transient = 500

starting_states = []
transient_states = []
final_states = []

# Run simulations
for ax_i, Dv in enumerate(Dvs):
    model = BelousovDiffusionModel(L, A, B, DU, Dv, dt)
    starting_states.append(model.state[:, :, 0].copy())
    
    for i in range(N_tot):
        model.update()
        
        if i == N_transient:
            transient_states.append(model.state[:, :, 0].copy())
    
    final_states.append(model.state[:, :, 0].copy())
    
    
# Find max and min values to ensure same color scaling
for state in starting_states + transient_states + final_states:
    v_max = np.max(state)
    v_min = np.min(state)

# Plotting
fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    
for starting_state, transient_state, final_state, Dv, axs_row in zip(starting_states, transient_states, final_states, Dvs, axes):
    axs_row[0].imshow(starting_state, cmap="viridis", vmin=v_min, vmax=v_max)
    axs_row[1].imshow(transient_state, cmap="viridis", vmin=v_min, vmax=v_max)
    axs_row[2].imshow(final_state, cmap="viridis", vmin=v_min, vmax=v_max)

row_labels = [r"$D_v=2.3$", r"$D_v=3$", r"$D_v=5$", r"$D_v=9$"]
col_labels = ["Starting state (i=0)", f"Starting state (i={N_transient})", f"Final state (i={N_tot})"]

for ax, row_label in zip(axes[:, 0], row_labels):
    ax.annotate(row_label, xy=(0, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14, rotation=90)
    
for ax, col_label in zip(axes[0, :], col_labels):
    ax.set_title(col_label)
    
for ax in axes.flatten():
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
im = axes[ax_i, 2].imshow(final_state, cmap="viridis", vmin=v_min, vmax=v_max)
cbar = fig.colorbar(im, ax=axes, shrink=0.7)
cbar.set_label(r"Concentration of $u$", fontsize=14)

fig.savefig("2_2_c_test.pdf")