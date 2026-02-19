import numpy as np
import matplotlib.pyplot as plt


def get_next_pertubation(eta, k: float, r: float, b: float, stable=False):
    if stable:

        eta_next = (1 - b * r / (1 + r)) * eta
    else:
        eta_next = (r + 1) * eta

    return eta_next


def get_exact_step(n, k: float, r: float, b: float) -> float:
    n_next = (r + 1) * n / (1 + (n / k) ** b)
    return n_next


def get_exact_trajectory(
    time_steps: int, n_0, k: float, r: float, b: float
) -> np.ndarray:
    trajectory = np.zeros(time_steps + 1)
    trajectory[0] = n_0

    for idx in range(time_steps):
        trajectory[idx + 1] = get_exact_step(trajectory[idx], k, r, b)

    return trajectory


def get_approx_trajectory(
    time_steps: int,
    k: float,
    r: float,
    b: float,
    stable_point: float,
    pertubation,
    stable=False,
) -> np.ndarray:
    eta = np.zeros((time_steps + 1))
    trajectory = np.zeros((time_steps + 1))
    trajectory[0] = pertubation + stable_point
    eta[0] = pertubation

    for idx in range(time_steps):
        eta[idx + 1] = get_next_pertubation(eta[idx], k, r, b, stable=stable)
        trajectory[idx + 1] = stable_point + eta[idx + 1]

    return trajectory


def part1():
    number_of_timesteps = 100
    stable_point = 0
    pertubation = np.array([1, 2, 3, 10])
    start_pos = stable_point + pertubation
    k = 1e3
    r = 0.1
    b = 1

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
    ax = ax.flatten()

    time = np.linspace(1, number_of_timesteps, number_of_timesteps + 1)

    for idx, pert in enumerate(pertubation):
        traj_exact = get_exact_trajectory(number_of_timesteps, start_pos[idx], k, r, b)
        traj_approx = get_approx_trajectory(
            number_of_timesteps, k, r, b, stable_point, pert
        )
        ax[idx].plot(time, traj_exact, label="Exact trajectory")
        ax[idx].plot(time, traj_approx, label="Approximate trajectory")
        ax[idx].scatter(1, start_pos[idx], label="Starting value", color="black")
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("Population")
        ax[idx].set_title(f"Pertubation: {pert}")
        ax[idx].set_xscale("log")
        ax[idx].set_yscale("log")
        ax[idx].legend()

    fig.tight_layout()
    fig.savefig("../figures/d.pdf")


def part2():
    number_of_timesteps = 100
    pertubation = np.array([-10, -3, -2, -1, 1, 2, 3, 10])
    k = 1e3
    r = 0.1
    b = 1

    stable_point = k * (r ** (1 / b))
    start_pos = stable_point + pertubation

    fig, ax = plt.subplots(4, 2, figsize=(10, 16), sharey=False, sharex=True)
    ax = ax.flatten()

    time = np.linspace(1, number_of_timesteps, number_of_timesteps + 1)

    for idx, pert in enumerate(pertubation):
        traj_exact = get_exact_trajectory(number_of_timesteps, start_pos[idx], k, r, b)
        traj_approx = get_approx_trajectory(
            number_of_timesteps,
            k,
            r,
            b,
            stable_point,
            pert,
            stable=True,
        )
        ax[idx].plot(time, traj_exact, label="Exact trajectory")
        ax[idx].plot(time, traj_approx, label="Approximate trajectory")
        ax[idx].scatter(1, start_pos[idx], label="Starting value", color="black")
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("Population")
        ax[idx].set_title(f"Pertubation: {pert:.0f}")
        ax[idx].set_xscale("log")
        ax[idx].set_yscale("log")
        ax[idx].legend()

    fig.tight_layout()
    fig.savefig("../figures/e.pdf")


def main():
    part1()
    part2()


if __name__ == "__main__":
    main()
