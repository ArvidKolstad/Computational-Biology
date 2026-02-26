import numpy as np
import matplotlib.pyplot as plt

class KuramotoModelSimulation:
    def __init__(self, N: int, K: float, dt: float) -> None:
        self.N: int = N
        self.K: float = K
        self.dt: float = dt
        
        self.thetas: np.ndarray = np.random.uniform(-np.pi/2, np.pi/2, size=self.N)
        self.omegas: np.ndarray = np.random.standard_cauchy(size=self.N) # Mean=0 and gamma=1
        self.omegas = np.clip(self.omegas, -10, 10) # Prevent really large omegas
    
    def get_r_vec(self) -> np.ndarray:
        """
        r = 1/N * sum(e^(i*theta_j))
        """
        xs = np.cos(self.thetas)
        ys = np.sin(self.thetas)
        
        r_x = np.sum(xs) / self.N
        r_y = np.sum(ys) / self.N
        
        r_vec = np.array([r_x, r_y])
        
        return r_vec
    
    def calc_r(self) -> np.floating:
        r_vec = self.get_r_vec()
        return np.linalg.norm(r_vec)
    
    def dtheta_dt(self) -> np.ndarray:
        theta_diffs = self.thetas[None, :] - self.thetas[:, None]
        
        return self.omegas + self.K / self.N * np.sum(np.sin(theta_diffs), axis=1)
        
    def update(self) -> None:
        """
        Euler explicit numerical integration:
        
        y_(i+1) = y_i + dt*f'(t, y_i)
        """
        self.thetas += self.dt * self.dtheta_dt()
        self.thetas = np.mod(self.thetas, 2*np.pi)

# Run simulation
dt = 0.01
T_tot = 50
Ks = [1, 2.1, 4]
Ns = [20, 100, 300]

data_K_N = []

for K in Ks:
    
    data_K = []
    
    for N in Ns:
        model = KuramotoModelSimulation(N, K, dt)
        
        ts = np.linspace(0, T_tot, int(T_tot/dt))
        rs = [model.calc_r()]
        
        for i in range(len(ts)-1):
            model.update()
            rs.append(model.calc_r())
        
        data_K.append(rs)
        
    data_K_N.append(data_K)

# Find mean r and variance
r_means = []
r_vars = []

for data_K, K in zip(data_K_N, Ks):
    print(f"K={K}:")
    
    r_means_K = []
    r_vars_K = []
    
    for data_N, N in zip(data_K, Ns):
        # Include data from after 10s simulation time
        r_mean = np.mean(data_N[int(10/dt):])
        r_var = np.var(data_N[int(10/dt):])
        
        print(f"    N={N}: r_mean={r_mean:.2}, r_var={r_var:.2}")
        
        r_means_K.append(r_mean)
        r_vars_K.append(r_var)
        
    r_means.append(r_means_K)
    r_vars.append(r_vars_K)
        
# Plotting
fig, axes = plt.subplots(3, 1, sharex=True, figsize=[8, 12])

for data_K, r_means_K, K, ax in zip(data_K_N, r_means, Ks, axes):
    
    for data_N, r_mean, N in zip(data_K, r_means_K, Ns):
        ax.plot(ts, data_N, label=f"$N={N}$, $r_m={r_mean:.2}$")
    
    ax.set_title(f"Kuramoto Model Simulation with K={K}")
    ax.set_ylabel("r")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid()
    
ax.set_xlabel("t (s)")

plt.tight_layout()
plt.savefig("2_3.pdf")
plt.show()