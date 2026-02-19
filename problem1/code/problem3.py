import numpy as np
import matplotlib.pyplot as plt


def next_population(n, r: np.ndarray, alpha: float) -> float:
    n_next = r * n * np.exp(-alpha * n)
    return n_next


def simulate_population(
    time_steps: int, n_0: float, r: np.ndarray, alpha: float
) -> np.ndarray:

    trajectories = np.zeros((r.shape[0], time_steps + 1))
    trajectories[:, 0] = n_0

    for idx in range(time_steps):
        trajectories[:, idx + 1] = next_population(trajectories[:, idx], r, alpha)
    return trajectories


def part1():
    time_steps = 300
    n_0 = 900
    alpha = 0.01
    r_values = np.arange(
        1,
        30.1,
        0.1,
    )
    trajectories = simulate_population(time_steps, n_0, r_values, alpha)
    fig, ax = plt.subplots()

    for idx, r in enumerate(r_values):
        ax.scatter(r * np.ones(100), trajectories[idx, -100:], s=1)
    ax.set_xlabel("R")
    ax.set_ylabel("Population")

    fig.tight_layout()
    fig.savefig("../figures/3a.pdf")


def part2():
    time_steps = 40
    n_0 = 900
    alpha = 0.01
    r_values = np.array([5.0, 10.0, 13.0, 23.0])
    trajectories = simulate_population(time_steps, n_0, r_values, alpha)
    fig, ax = plt.subplots()

    for idx, r in enumerate(r_values):
        ax.plot(trajectories[idx, :], label=f"R = {r}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population")
    ax.legend()

    fig.tight_layout()
    fig.savefig("../figures/3b.pdf")


def part3():
    time_steps = 300
    n_0 = 900
    alpha = 0.01
    r_values = np.arange(
        12,
        13.1,
        0.1,
    )
    trajectories = simulate_population(time_steps, n_0, r_values, alpha)
    fig, ax = plt.subplots()

    for idx, r in enumerate(r_values):
        ax.scatter(r * np.ones(100), trajectories[idx, -100:])
    ax.set_xlabel("R")
    ax.set_ylabel("Population")

    fig.tight_layout()
    fig.savefig("../figures/3c2.pdf")


def part4():
    time_steps = 1000
    n_0 = 900
    alpha = 0.01
    r_values = np.arange(14.74, 14.79, 0.0005)
    trajectories = simulate_population(time_steps, n_0, r_values, alpha)
    fig, ax = plt.subplots()

    for idx, r in enumerate(r_values):
        ax.scatter(r * np.ones(800), trajectories[idx, -800:], s=0.1)

    ax.set_ylabel("R")
    ax.set_ylabel("Population")

    ax.grid()

    fig.tight_layout()
    fig.savefig("../figures/3d.pdf")


def main():
    # part1()
    # part2()
    # part3()
    part4()


if __name__ == "__main__":
    main()
