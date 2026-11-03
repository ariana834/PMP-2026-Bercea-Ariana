# Ex. 2 - Jocul cu moneda și zarul
#
# a)
# prima stema = succes
# N=numarul de pasi pana la prima stema(succes)
# S=banii rezultati la finalul jocului
# E[N]=media lui N, E[S]=media lui S (se numeste valoarea asteptata)
#    => distributie geometrica, P(N = n) = (1 - p)^(n-1) * p, n = 1, 2, 3, ... => numarul de incercari pana la primul succes
#   avem nevoie de E[N] - valoarea asteptata = 1/p
#    Pentru moneda corecta = 0.5, deci E[N]=1/p
#
#    Suma totala: S =(z-3)-0.5*(N-1)
#    Teoretic: E[S] = E[z-3] - 0.5 * E[N - 1] = E[Z]-3 - 0.5 * (1 - p) / p = 0.5-0.5*(1-p)/p

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
# rng = generator de numere aleatoare
# np.random.default_rng() creeaza generatorul
# rng.random()
# genereaza un numar aleator intre 0 si 1
# rng.random() < p este True cu probabilitatea p, deci poate fi folosit pentru a simula aparitia stemei


# b) Simularea unui singur joc
def joaca_un_joc(p=0.5):
    N = 0
    S = 0.0
    while True:
        N += 1
        stema = rng.random() < p          # True cu probabilitatea p
        if stema:
            z = rng.integers(1, 7)
            S += z-3                    # al doilea ii da primului z-3
            return N, S                   # jocul se termina
        else:
            S -= 0.5                      # primul îi dă celui de-al doilea 0.5


def simuleaza(p=0.5, nr_jocuri=10000):
    rezultate = [joaca_un_joc(p) for _ in range(nr_jocuri)]
    N = np.array([r[0] for r in rezultate])
    S = np.array([r[1] for r in rezultate])
    return N, S


def media_teoretica_S(p):
    return 0.5 - 0.5 * (1 - p) / p

N1, S1 = joaca_un_joc()
print(f"Un joc: N = {N1}, S = {S1}\n")

valori_p = [0.5, 0.3, 0.7]
fig, axe = plt.subplots(2, len(valori_p), figsize=(15, 8))

for j, p in enumerate(valori_p):
    N, S = simuleaza(p)

    print(f"p = {p}:")
    print(f"  media lui N: {N.mean():.3f}   (teoretic 1/p = {1 / p:.3f})")
    print(f"  media lui S: {S.mean():.3f}   (teoretic {media_teoretica_S(p):.3f})")

    # Histograma lui S: S ia valori discrete, deci număr fiecare valoare separat
    valori, frecvente = np.unique(S, return_counts=True)
    axe[0, j].bar(valori, frecvente / len(S), width=0.4)
    axe[0, j].axvline(S.mean(), color="red", linestyle="--", label=f"media = {S.mean():.2f}")
    axe[0, j].set_title(f"Distribuția lui S (p = {p})")
    axe[0, j].set_xlabel("S ($)")
    axe[0, j].set_ylabel("frecvență relativă")
    axe[0, j].set_xlim(-8, 4)
    axe[0, j].legend()

    n_max = 12
    n = np.arange(1, n_max + 1)
    frecv_N = [(N == k).mean() for k in n]
    axe[1, j].bar(n, frecv_N, alpha=0.6, label="simulat")
    axe[1, j].plot(n, (1 - p) ** (n - 1) * p, "ro-", label="Geometrică teoretic")
    axe[1, j].set_title(f"Distribuția lui N (p = {p})")
    axe[1, j].set_xlabel("N (număr de pași)")
    axe[1, j].set_ylabel("probabilitate")
    axe[1, j].legend()

plt.tight_layout()
plt.show()