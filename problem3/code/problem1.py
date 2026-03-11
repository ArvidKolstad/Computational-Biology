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

    np.save(f"infection_distribution3.npy", distribution_infections)
    np.save("recovery_distribution3.npy", distribution_recovery)


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

    mean_low_inf = np.average(bins_inf[:-1], weights=hist_inf)
    mean_high_inf = np.average(bins_inf[1:], weights=hist_inf)

    mean_inf = (mean_low_inf + mean_high_inf) / 2

    mean_low_rec = np.average(bins_rec[:-1], weights=hist_rec)
    mean_high_rec = np.average(bins_rec[1:], weights=hist_rec)
    mean_rec = (mean_low_rec + mean_high_rec) / 2

    ax[0].stairs(hist_inf, bins_inf, label="Distribution Infected")
    ax[0].plot(t_inf, exp_model(t_inf, a_inf, lambda_inf), label="Exponential fit")
    ax[0].set_xticks(np.arange(0, 1, 0.1))
    ax[0].set_yscale("log")
    ax[0].text(0.7, 15, rf"$\lambda$ = {lambda_inf:.3f}")
    ax[0].text(0.7, 30, rf"$\mu$ = {mean_inf:.3f}")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0.9, 80)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Frequency")
    ax[0].grid()
    ax[0].legend()

    ax[1].stairs(hist_rec, bins_rec, label="Distribution Recovered")
    ax[1].plot(t_rec, exp_model(t_rec, a_rec, lambda_rec), label="Exponential fit")
    ax[1].set_xticks(np.arange(0, 1, 0.1))
    ax[1].set_yscale("log")
    ax[1].text(0.7, 15, rf"$\lambda$ = {lambda_rec:.3f}")
    ax[1].text(0.7, 30, rf"$\mu$ = {mean_rec:.3f}")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0.9, 80)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Frequency")
    ax[1].grid()
    ax[1].legend()
    fig.suptitle(rf"$b_n = {lambda_inf:.1f}$, and $d_n = {lambda_rec:.1f}$")
    fig.tight_layout()
    fig.savefig(file_name)


def c():
    b = 10
    d = 5
    time_step = 1e-6
    total_samples = 10000
    # get_distribution(b, d, time_step, total_samples)
    infection_dis = np.load("infection_distribution3.npy")
    recovery_dis = np.load("recovery_distribution3.npy")

    plot_distribution(
        infection_dis, recovery_dis, "../report1/figures/distribution_plot3.pdf"
    )


def get_next_event(dist_infected: np.ndarray, dist_recovery: np.ndarray):
    rand_index_infected = np.random.randint(0, dist_infected.shape[0])
    rand_index_recovery = np.random.randint(0, dist_recovery.shape[0])

    time_infected = dist_infected[rand_index_infected]
    time_recovery = dist_infected[rand_index_recovery]
    if time_infected > time_recovery:
        event_type = "recovery"
        time = time_recovery
        return event_type, time
    else:
        event_type = "infection"
        time = time_infected
        return event_type, time


def print_calculations(alpha: float, beta: float, N: float):
    r_0 = alpha / beta
    s_0 = np.log(r_0) - (1 - 1 / r_0)
    t_ext = np.exp(N * s_0)
    I_star = N * (1 - 1 / r_0)
    print(f"T_exit = {t_ext}")
    print(f"I_star = {I_star}")


def run_simulation(
    samples: int,
    record_times: np.ndarray,
    starting_infected: int,
    total_population: int,
    alpha: float,
    beta: float,
    total_simulation_time: int,
):
    n = np.zeros(samples, dtype=int)
    n[:] = starting_infected
    time = np.zeros(samples)
    has_recorded = np.zeros((samples, 3), dtype=bool)
    t_exit = []
    saved_n = np.zeros((samples, 3))
    n_zero = n <= 0
    n_max = n >= total_population
    t_b = np.zeros_like(time)
    t_d = np.zeros_like(time)

    for i in range(total_simulation_time):
        if i % 1000 == 0:
            print(i)
        b_n = alpha * n * (1 - n / total_population)
        d_n = beta * n
        t_b[n_zero] = 1e5
        t_b[~n_zero] = np.random.exponential(1 / b_n[~n_zero])
        t_b[n_max] = 1e5

        t_d[n_zero] = 1e5
        t_d[~n_zero] = np.random.exponential(1 / d_n[~n_zero])

        t_b_smaller = (t_b - t_d) < 0
        t_d_smaller = (t_d - t_b) < 0

        time[t_d_smaller] += t_d[t_d_smaller]
        time[t_b_smaller] += t_b[t_b_smaller]

        n[t_b_smaller] += 1
        n[t_d_smaller] -= 1

        n_zero = n <= 0

        n_max = n == total_population

        to_save = (~has_recorded) & (
            (time[:, None] >= record_times[None, :]) | (n_zero[:, None])
        )
        saved_n[to_save] = np.broadcast_to(n[:, None], (n.shape[0], 3))[to_save]
        has_recorded[to_save] = True

    # assert has_recorded.all() == True
    print(np.mean(time[n_zero]))
    print(np.sum(n_zero) / samples)
    print(np.sum(has_recorded) / (samples * 3))

    return saved_n


def gaussian_func(x: np.ndarray, mean: float, variance: float):
    return (
        1
        / (np.sqrt(2 * np.pi * variance))
        * np.exp(-((x - mean) ** 2) / (2 * variance))
    )


def d():
    alpha = 0.112
    beta = 0.1
    N = 1000
    samples = 5000
    simulation_time = 10000000
    t_record = np.array([5000, 37125, 100000])

    print_calculations(alpha, beta, N)

    n_start = 107
    variance = N * beta / alpha
    n = run_simulation(samples, t_record, n_start, N, alpha, beta, simulation_time)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    for idx, ax in enumerate(axes):
        hist, bins = np.histogram(n[:, idx], bins=100, density=True)
        bin_min = bins[0]
        bin_max = bins[-1]
        x = np.linspace(bin_min, bin_max, 1000)

        ax.stairs(-np.log(hist), bins, label="sampled simulations")
        ax.plot(
            x, -np.log(gaussian_func(x, n_start, variance)), label="gaussian function"
        )

        ax.axvline(n_start, label=r"$I^*_2$", c="black")
        ax.set_title(rf" $t = {t_record[idx]}$")
        ax.set_xlabel(r"$n_t$")
        ax.set_ylabel(r"$-Log\left(p(n_t)\right)$")
        ax.legend()

    fig.tight_layout()
    fig.savefig("../report1/figures/plot_dist_pn.pdf")


def main():
    # c()
    d()
    # test()


if __name__ == "__main__":
    main()
