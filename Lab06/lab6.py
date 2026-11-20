import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# datele observate: 180 de apeluri in 10 ore
n_ore = 10
total_apeluri = 180

# prior-ul: Gamma(alpha, beta)
# nu avem cifre exacte din istoric, stim doar ca rata variaza mult,
# asa ca aleg un prior slab informativ:
#   media = alpha / beta = 2 / 0.1 = 20 apeluri pe ora (ordinul de marime e plauzibil)
#   dispersia = alpha / beta^2 = 200, deci deviatia ~14 -> foarte larg
# un prior larg lasa datele sa decida aproape tot
alpha_prior = 2
beta_prior = 0.1

# de ce Gamma: e conjugata pentru Poisson, adica posterior-ul iese tot Gamma
# si il putem scrie direct, fara simulari.
#
# demonstratia pe scurt:
#   verosimilitatea pentru k_1, ..., k_n observatii Poisson:
#     L(lambda) = produs e^(-lambda) * lambda^k_i / k_i!
#               ~ lambda^(suma k_i) * e^(-n * lambda)
#   prior-ul:
#     p(lambda) ~ lambda^(alpha - 1) * e^(-beta * lambda)
#   posterior = prior * verosimilitate (pana la o constanta):
#     ~ lambda^(alpha + suma k_i - 1) * e^(-(beta + n) * lambda)
#   asta e exact forma unei Gamma(alpha + suma k_i, beta + n)
#
# nu avem nevoie de fiecare ora separat, conteaza doar suma (180) si n (10)

alpha_post = alpha_prior + total_apeluri   # 2 + 180 = 182
beta_post = beta_prior + n_ore             # 0.1 + 10 = 10.1

print("Posterior: Gamma(alpha =", alpha_post, ", beta =", beta_post, ")")

# media si modul posterior-ului, cu formulele din indicatii
media_post = alpha_post / beta_post
modul_post = max(alpha_post - 1, 0) / beta_post   # valoarea cea mai probabila
dev_post = np.sqrt(alpha_post) / beta_post

print("media posterior :", round(media_post, 3))
print("modul posterior :", round(modul_post, 3), " <- cea mai probabila valoare a lui lambda")
print("deviatie std    :", round(dev_post, 3))

# atentie la scipy: gamma are parametrul scale = 1 / beta, nu beta direct
posterior = stats.gamma(a=alpha_post, scale=1 / beta_post)

# HDI de 94% = cel mai scurt interval care contine 94% din probabilitate
# il calculam din esantioane, ca la arviz:
#   sortam esantioanele, incercam toate intervalele care contin 94% din ele
#   si il pastram pe cel mai ingust
esantioane = posterior.rvs(size=100000, random_state=42)

def hdi(x, prob=0.94):
    x = np.sort(x)
    n = len(x)
    k = int(np.floor(prob * n))          # cate puncte trebuie sa intre in interval
    latimi = x[k:] - x[:n - k]           # latimea fiecarui interval posibil
    i = np.argmin(latimi)                # cel mai ingust
    return x[i], x[i + k]

st, dr = hdi(esantioane)
print("HDI 94%         : [", round(st, 3), ",", round(dr, 3), "]")

# cu arviz e un singur apel, si plot_posterior deseneaza tot (HDI + mod)
try:
    import arviz as az
    print("HDI arviz       :", az.hdi(esantioane, hdi_prob=0.94))
    az.plot_posterior(esantioane, hdi_prob=0.94, point_estimate="mode")
    plt.savefig("posterior_arviz_lab6.png")
except ImportError:
    print("arviz nu e instalat, desenez cu matplotlib")

# grafic: prior vs posterior, ca sa se vada cat de mult au mutat datele
x = np.linspace(0, 40, 1000)
prior = stats.gamma(a=alpha_prior, scale=1 / beta_prior)

plt.figure(figsize=(8, 4))
plt.plot(x, prior.pdf(x), label="prior Gamma(2, 0.1)")
plt.plot(x, posterior.pdf(x), label=f"posterior Gamma({alpha_post}, {beta_post})")
plt.axvspan(st, dr, alpha=0.2, label="HDI 94%")
plt.axvline(modul_post, linestyle="--", label=f"mod = {modul_post:.2f}")
plt.xlabel("lambda (apeluri pe ora)")
plt.legend()
plt.tight_layout()   # ca sa nu taie eticheta de jos
plt.savefig("posterior_lab6.png")
# plt.show()

# observatie: modul posterior-ului (17.92) e foarte aproape de rata observata (18),
# pentru ca prior-ul e slab si avem 10 ore de date. un prior mai puternic
# (alpha si beta mari) ar trage rezultatul spre media lui.
