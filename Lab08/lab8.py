import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# valorile pe care le cere exercitiul
Y_values = [0, 5, 10]          # cati clienti au cumparat
theta_values = [0.2, 0.5]      # probabilitatea ca un client sa cumpere

# aici tin rezultatele pt fiecare combinatie (Y, theta)
results = {}

for Y in Y_values:
    for theta in theta_values:
        with pm.Model() as model:
            # prior pt n: nu stim cati clienti au venit, dar in medie cam 10
            n = pm.Poisson("n", mu=10)

            # verosimilitatea: din cei n clienti, fiecare cumpara cu prob theta
            # Y e ce am observat, theta il stim deci e doar un numar fix
            y_obs = pm.Binomial("y_obs", n=n, p=theta, observed=Y)

            # n e discret deci pymc nu poate folosi NUTS, alege singur Metropolis
            idata = pm.sample(draws=2000, tune=2000, chains=4, random_seed=42)

        results[(Y, theta)] = idata

        # ne uitam si la diagnostice (r_hat ~ 1, ess destul de mare)
        print(f"Y = {Y}, theta = {theta}")
        print(az.summary(idata, var_names=["n"]))
        print()

# toate graficele intr-o singura fereastra: randuri = Y, coloane = theta
fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(12, 10))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        az.plot_posterior(results[(Y, theta)], var_names=["n"], ax=axes[i, j])
        axes[i, j].set_title(f"Y = {Y}, theta = {theta}")

plt.tight_layout()
plt.savefig("posterior_n.png")
plt.show()
