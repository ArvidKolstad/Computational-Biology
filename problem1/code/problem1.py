import numpy as np
import matplotlib.pyplot as plt
import scipy


def n(n_trajectory: np.ndarray, t: float, time_step: float, n_0: float) -> float:
    if t < 0:
        return n_0
    else:
        index = np.round(t / time_step)
        return float(n_trajectory[index])


def n_prim(
    n_trajectory: np.ndarray,
    t: float,
    time_step: float,
    T: float,
    A: float,
    K: float,
    r: float,
    n_0: float,
):
    n_prim = r * n(n_trajectory, t, time_step, n_0)


def main():

    T = np.arange(0.1, 5.1, 0.1)
    A = 20.0
    K = 100.0
    r = 0.1
    N_0 = 50.0


if __name__ == "__main__":
    main()
