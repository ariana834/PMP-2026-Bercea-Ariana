# lambda =rata unei distributii exponentiale
# In problema: viteza frizerului, exprimata in clienti/ora
# X=timpul de servire al unui client

# Daca X ~ Exp(lambda): E[X] = 1 / lambda, adica timpul mediu de servire este 1/lambda
# cu cat lambda este mai mare,cu atat frizerul este mai rapid si timpul mediu este mai mic

# np.random.choice(...)alege aleator o valoare dintre mai multe posibilitati

# p = probabilitatile cu care sunt alese posibilitatile
# np.random.exponential(scale=...) genereaza o valoare din distributia exponentiala
# scale = 1 / lambda

# mean() = media valorilor simulate
# std() = deviatia standard masoara cat de imprastiate sunt valorile in jurul mediei

# histograma = arata aproximativ distributia valorilor simulate
# density=True transforma histograma intr-o aproximare a densitatii de probabilitate

import numpy as np
import matplotlib.pyplot as plt


lambdas = [3, 6, 4]
probabilitati = [3/13, 6/13, 4/13]
nr_clienti = 10000

#pentru fiecare client aleg frizerul (0, 1 sau 2)
frizer = np.random.choice([0, 1, 2], size=nr_clienti, p=probabilitati)

X = np.zeros(nr_clienti)
for i in range(nr_clienti):
    lam = lambdas[frizer[i]]
    X[i] = np.random.exponential(scale=1/lam)

#media si deviația standard
print(f"Media lui X: {X.mean():.4f} ore  (= {X.mean() * 60:.1f} minute)")
print(f"Deviația standard: {X.std():.4f} ore")

#graficul densitatii
plt.hist(X, bins=60, density=True, alpha=0.4, label="histograma")
try:
    import arviz as az
    az.plot_kde(X)
except Exception:
    from scipy.stats import gaussian_kde
    xs = np.linspace(0, X.max(), 500)
    plt.plot(xs, gaussian_kde(X)(xs), label="densitate (KDE)")
plt.xlabel("Timpul de servire X (ore)")
plt.ylabel("Densitate")
plt.title("Densitatea timpului de servire la frizerie")
plt.legend()
plt.show()