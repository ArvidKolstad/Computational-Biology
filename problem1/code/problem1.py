import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint


def n_prim(
    n,
    t: float,
    T: float,
    A: float = 20.0,
    K: float = 100.0,
    r: float = 0.1,
) -> float:
    n_prim = r * n(t) * (1 - n(t - T) / K) * (n(t) / A - 1)
    return n_prim


def before_zero(t, n_0=50.0):
    return n_0


def check_if_oscillation(
    trajectory: np.ndarray, mid_point: float, thresh_hold: float = 1e-3
) -> bool:
    if np.max(trajectory) - mid_point > thresh_hold:
        index = np.argmax(trajectory)
        print(np.min(trajectory[index:]))
        if mid_point - np.min(trajectory[index:]) > thresh_hold:
            return True
        else:
            return False
    else:
        return False


def check_if_limit_cycle(trajectory: np.ndarray, thresh_hold: float = -1e-3) -> bool:
    index = []
    peaks = []

    for idx, point in enumerate(trajectory):
        if (idx != 0) and (idx != trajectory.shape[0] - 1):
            if (trajectory[idx - 1] < point) and (trajectory[idx + 1] < point):
                index.append(idx)
                peaks.append(point)
    k = np.polyfit(index, peaks, 1)[0]
    print(k)

    if k < thresh_hold:
        return True
    else:
        return False


def run_integration(
    different_T: np.ndarray,
    total_time: float,
    check_oscillation=False,
    check_limit_cycle=False,
):
    tt = np.linspace(0, total_time, 5 * int(total_time))
    fig, ax = plt.subplots()
    # ax.axhline(100.0, color="black", label="Midpoint")
    for T in different_T:
        print(T)
        yy = ddeint(n_prim, before_zero, tt, fargs=(T,))

        linestyle = "solid"

        if check_oscillation:
            ax.set_ylim(95, 105)
            ax.set_xlim(7, 20)
            oscillation = check_if_oscillation(yy, 100.0)
            if oscillation:
                linestyle = "dotted"
            else:
                linestyle = "dashed"

        if check_limit_cycle:
            limit_cycle = check_if_limit_cycle(yy)
            if limit_cycle:
                linestyle = "dashdot"
            else:
                linestyle = "dotted"

        ax.plot(tt, yy, label=f"T = {T:.1f}", linestyle=linestyle)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population (N)")
    ax.set_title("Limit cycle")
    fig.legend()
    fig.savefig("../report/figures/problem1/stable_oscillation.pdf")


def main():

    T = np.arange(5.0, 5.05, 0.1)
    total_runtime = 200.0
    run_integration(T, total_runtime, check_limit_cycle=False)


if __name__ == "__main__":
    main()
