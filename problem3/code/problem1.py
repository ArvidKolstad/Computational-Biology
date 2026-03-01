import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_model(x, a, b):
    return a * np.exp(-b * x)


def check_probability(p: float) -> bool:
    rand = np.random.rand()
    if p > rand:
        return True
    else:
        return False


def draw_outcome(p_b: float, p_d: float) -> str:
    if check_probability(p_b + p_d):
        pb_probability = p_b / (p_b + p_d)
        if check_probability(pb_probability):
            return "new infection"
        else:
            return "new recovery"
    else:
        return "nothing happened"


def get_distribution(b: float, d: float, dt: float, total_samples: int):
    distribution_infections = []
    distribution_recovery = []
    samples = 0
    time = 0.0
    p_b = b * dt
    p_d = d * dt

    saved_infection = False
    saved_recovery = False
    while total_samples > samples:
        time += dt
        outcome = draw_outcome(p_b, p_d)
        if (outcome == "new infection") and not (saved_infection):
            distribution_infections.append(time)
            saved_infection = True
            samples += 1
        elif outcome == "new recovery" and not (saved_recovery):
            saved_recovery = True
            distribution_recovery.append(time)
            samples += 1

        if saved_infection and saved_recovery:
            time = 0.0
            saved_infection = False
            saved_recovery = False

        if samples % 100 == 0 and samples != 0:
            print(samples)
    distribution_infections = np.array(distribution_infections)
    distribution_recovery = np.array(distribution_recovery)

    np.save(f"infection_distribution1.npy", distribution_infections)
    np.save("recovery_distribution1.npy", distribution_recovery)


def plot_distribution(distr_inf, distr_rec: np.ndarray, file_name: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    hist_inf, bins_inf = np.histogram(distr_inf, bins=1000)
    hist_rec, bins_rec = np.histogram(distr_rec, bins=1000)
    t_inf = np.linspace(bins_inf[0], bins_inf[-1], 1000)
    t_rec = np.linspace(bins_rec[0], bins_rec[-1], 1000)

    params_inf, covariance_inf = curve_fit(exp_model, bins_inf[1:], hist_inf)
    params_rec, covariance_rec = curve_fit(exp_model, bins_rec[1:], hist_rec)
    a_inf, lambda_inf = params_inf
    a_rec, lambda_rec = params_rec

    ax[0].stairs(hist_inf, bins_inf)
    ax[0].plot(t_inf, exp_model(t_inf, a_inf, lambda_inf))
    ax[0].set_xticks(np.arange(0, 50, 3))
    ax[0].set_yscale("log")
    ax[0].text(24, 10, f"lambda = {lambda_inf:.3f}")
    ax[0].grid()

    ax[1].stairs(hist_rec, bins_rec)
    ax[1].plot(t_rec, exp_model(t_rec, a_rec, lambda_rec))
    ax[1].set_xticks(np.arange(0, 50, 3))
    ax[1].set_yscale("log")
    ax[1].text(24, 10, f"lambda = {lambda_rec:.3f}")
    ax[1].grid()

    fig.tight_layout()
    fig.savefig(file_name)


def c():
    b = 0.1
    d = 0.2
    time_step = 1e-4
    total_samples = 10000
    get_distribution(b, d, time_step, total_samples)
    infection_dis = np.load("infection_distribution1.npy")
    recovery_dis = np.load("recovery_distribution1.npy")

    plot_distribution(infection_dis, recovery_dis, "distribution_plot1.pdf")


def main():
    c()


if __name__ == "__main__":
    main()
